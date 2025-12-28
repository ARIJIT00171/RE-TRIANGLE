import os
import torch.distributed as dist
from tqdm import tqdm
from collections import defaultdict
import torch
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import autocast, GradScaler
import wandb

from evaluation import evaluation_registry
from utils.save import ModelSaver
from utils.sched import get_lr_sched
from utils.utils import NoOp
from utils.logger import LOGGER, RunningMeter

str_to_bool_mapper = {
    "yes": True,
    "no": False
}


def train(model, optimizer, train_loader, val_loaders, final_config, start_step=0):
    run_cfg = final_config.run_cfg
    if dist.get_rank() == 0:
        LOGGER.info(f'Total Steps: {run_cfg.num_train_steps}')
        pbar = tqdm(total=run_cfg.num_train_steps, initial=start_step)
        model_saver = ModelSaver(os.path.join(run_cfg.output_dir, 'ckpt'),
                                 remove_before_ckpt=str_to_bool_mapper[run_cfg.remove_before_ckpt])
    else:
        pbar = NoOp()
        model_saver = NoOp()

    loss_moving_averagetors = {}
    metric_logger_dict = defaultdict(dict)
    global_step = start_step

    scaler = GradScaler()

    best_indicator = {}
    evaluate_fn = evaluation_registry[model.config.evaluation_type]

    for step, (name, batch) in enumerate(train_loader):
        task = name.split('--')[0]

        optimizer.zero_grad(set_to_none=True)
        if run_cfg.load_dtype == "fp16":
            with autocast(dtype=torch.float16):
                loss_dict = model(batch, tasks=task, compute_loss=True, global_step=global_step)
                loss = sum(list(loss_dict.values()))
                loss_dict['total_loss'] = loss
                loss_dict = {k: v.item() for k, v in loss_dict.items()}
                if dist.get_rank() == 0:
                    wandb.log(loss_dict)
        else:
            loss_dict = model(batch, tasks=task, compute_loss=True, global_step=global_step)
            loss = sum(list(loss_dict.values()))
            loss_dict['total_loss'] = loss
            loss_dict = {k: v.item() for k, v in loss_dict.items()}
            if dist.get_rank() == 0:
                wandb.log(loss_dict)

        if name not in loss_moving_averagetors:
            # first time initialize
            for k in loss_dict.keys():
                loss_moving_averagetors[f'loss_{name}/{k}'] = RunningMeter()

        # accumulate loss
        for k, v in loss_dict.items():
            loss_moving_averagetors[f'loss_{name}/{k}'](v)

        global_step += 1
        # learning rate scheduling
        lr_ratio = get_lr_sched(global_step, run_cfg)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['init_lr'] * lr_ratio

        if global_step % 50 == 0:
            LOGGER.info({name: averagetor.val for name, averagetor in loss_moving_averagetors.items()})
            # update model params

        if dist.get_rank() == 0 and global_step % 50 == 0:
            for i, pg in enumerate(optimizer.param_groups):
                LOGGER.info(
                    f"step={global_step} group={i} lr={pg['lr']:.3e} "
                    f"init_lr={pg.get('init_lr', -1):.3e} lr_ratio={lr_ratio:.3e}"
                )

        if run_cfg.load_dtype == "fp16":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if run_cfg.grad_norm != -1:
                clip_grad_norm_(model.parameters(), run_cfg.grad_norm)
        else:
            loss.backward()
            if run_cfg.grad_norm != -1:
                clip_grad_norm_(model.parameters(), run_cfg.grad_norm)

        # ---- DEBUG: grad norm per param-group (add here) ----
        def group_grad_norm(pg):
            s = 0.0
            for p in pg["params"]:
                if p.grad is not None:
                    s += p.grad.data.float().norm(2).item() ** 2
            return s ** 0.5

        if dist.get_rank() == 0 and global_step % 100 == 0:
            for i, pg in enumerate(optimizer.param_groups):
                LOGGER.info(f"step={global_step} group={i} lr={pg['lr']:.3e} grad_norm={group_grad_norm(pg):.3e}")

        if not str_to_bool_mapper[run_cfg.use_ddp]:
            works = []
            for p in model.parameters():
                # to speed it up, you can also organize grads to larger buckets to make allreduce more efficient
                if p.grad is not None:
                    works.append(dist.all_reduce(p.grad, async_op=True))
            for work in works:
                work.wait()

        if run_cfg.load_dtype == "fp16":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        pbar.update(1)

        if (global_step + 1) % run_cfg.valid_steps == 0:
            if dist.is_initialized():
                dist.barrier()

            del loss
            torch.cuda.empty_cache()

            model.eval()
            with torch.inference_mode():
                eval_log = evaluate_fn(model, val_loaders, final_config, global_step=global_step)
            model.train()

            if dist.get_rank() == 0:
                for task_name, val_log in eval_log.items():
                    for eval_name, metric in val_log.items():
                        eval_name = task_name + '_' + eval_name
                        metric_logger_dict[eval_name][str(global_step)] = metric
                        LOGGER.info(f"====-evaluation--{eval_name}=====step {global_step}--===========\n")
                        LOGGER.info(metric)
                        best_name = get_best_name(eval_name, metric)
                        if best_name is not None:
                            if ('best_step' not in metric_logger_dict[eval_name]) or \
                                    (metric[best_name] >= metric_logger_dict[eval_name]['best_value']):
                                metric_logger_dict[eval_name]['best_step'] = global_step
                                metric_logger_dict[eval_name]['best_value'] = metric[best_name]
                                best_indicator[eval_name] = True
                            else:
                                best_indicator[eval_name] = False
                            best_step = metric_logger_dict[eval_name]['best_step']
                            LOGGER.info(f"======evaluation--{eval_name}====history best step: {best_step}=======\n")
                            LOGGER.info(metric_logger_dict[eval_name][str(best_step)])

                model_saver.save(model, global_step, optimizer, best_indicator, str_to_bool_mapper[run_cfg.save_best])

        if global_step >= run_cfg.num_train_steps:
            break
    pbar.close()


def test(model, test_loader, final_config):
    run_cfg = final_config.run_cfg
    evaluate_fn = evaluation_registry[model.config.evaluation_type]
    eval_log = evaluate_fn(model, test_loader, final_config, global_step=0)
    if dist.get_rank() == 0:
        for task_name, val_log in eval_log.items():
            for eval_name, metric in val_log.items():
                eval_name = task_name + '_' + eval_name
                LOGGER.info(f"==== evaluation--{eval_name}========\n")
                LOGGER.info(metric)


def get_best_name(eval_name, metric):
    if eval_name.startswith('cap'):
        return 'CIDEr'
    elif eval_name.startswith('qa'):
        return 'accuracy'
    elif eval_name.startswith('ret'):
        if 'video_r1' in metric:
            return 'video_r1'
        if 'area_T2D_r1' in metric:
            return 'area_T2D_r1'
        if 'triangleian_value' in metric:
            return 'triangleian_value'
    elif eval_name.startswith('pt'):
        return None
    else:
        raise NotImplementedError
