import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from model.main_models import model_registry
from utils.logger import LOGGER

str_to_bool_mapper = {
    "yes": True,
    "no": False
}


class DDP_modify(DDP):
    def __init__(self, model, device_ids, output_device, find_unused_parameters):
        super().__init__(model, device_ids=device_ids, output_device=output_device,
                         find_unused_parameters=find_unused_parameters)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except Exception:
            return getattr(self.module, name)


def build_model(final_config, device):
    model = model_registry[final_config.model_cfg.model_type](final_config.model_cfg, final_config.run_cfg)
    checkpoint = {}

    # load ckpt from a pretrained_dir
    if final_config.run_cfg.pretrain_dir:
        checkpoint = load_from_pretrained_dir(final_config)
        LOGGER.info("load from pretrained dir {} successful".format(final_config.run_cfg.pretrain_dir))

    # load ckpt from specific path
    if final_config.run_cfg.checkpoint:
        checkpoint = torch.load(final_config.run_cfg.checkpoint, map_location='cpu')

    # resume training
    if str_to_bool_mapper[final_config.run_cfg.resume]:
        checkpoint, checkpoint_optim, start_step = load_from_resume(final_config.run_cfg)
    else:
        checkpoint_optim, start_step = None, 0

    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

    if checkpoint != {}:
        checkpoint = model.modify_checkpoint(checkpoint)
        if "model" in checkpoint.keys():
            checkpoint = checkpoint["model"]

        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
        LOGGER.info(f"Unexpected keys {unexpected_keys}")
        LOGGER.info(f"missing_keys  {missing_keys}")

    model.to(device)
    if str_to_bool_mapper[final_config.run_cfg.use_ddp]:
        model = DDP_modify(model, device_ids=[final_config.local_rank], output_device=final_config.local_rank,
                           find_unused_parameters=True)
    else:
        pass

    return model, checkpoint_optim, start_step


def load_from_pretrained_dir(final_config):
    try:  # huggingface trainer
        checkpoint_dir = final_config.run_cfg.pretrain_dir
        checkpoint_ls = [i for i in os.listdir(checkpoint_dir) if i.startswith('checkpoint')]
        checkpoint_ls = [int(i.split('-')[1]) for i in checkpoint_ls]
        checkpoint_ls.sort()
        step = checkpoint_ls[-1]

        try:
            checkpoint_name = f'checkpoint-{step}/pytorch_model.bin'
            ckpt_file = os.path.join(checkpoint_dir, checkpoint_name)
            checkpoint = torch.load(ckpt_file, map_location='cpu')
            LOGGER.info(f'loading from checkpoint file: {ckpt_file}')
        except Exception:
            checkpoint_name1 = f'checkpoint-{step}/pytorch_model-00001-of-00002.bin'
            ckpt_file1 = torch.load(os.path.join(checkpoint_dir, checkpoint_name1), map_location='cpu')
            checkpoint_name2 = f'checkpoint-{step}/pytorch_model-00002-of-00002.bin'
            ckpt_file2 = torch.load(os.path.join(checkpoint_dir, checkpoint_name2), map_location='cpu')
            ckpt_file1.update(ckpt_file2)
            checkpoint = ckpt_file1
            LOGGER.info(f'loading from checkpoint file: {os.path.join(checkpoint_dir, checkpoint_name1)} and '
                        f'{os.path.join(checkpoint_dir, checkpoint_name2)}')
        # checkpoint = {k.replace('module.',''):v for k,v in checkpoint.items()}
    except Exception:
        checkpoint_dir = os.path.join(final_config.run_cfg.pretrain_dir, 'ckpt')
        checkpoint_ls = [i for i in os.listdir(checkpoint_dir) if i.startswith('model_step')]
        checkpoint_ls = [int(i.split('_')[2].split('.')[0]) for i in checkpoint_ls]
        checkpoint_ls.sort()
        step = checkpoint_ls[-1]

        checkpoint_name = 'model_step_' + str(step) + '.pt'
        ckpt_file = os.path.join(checkpoint_dir, checkpoint_name)
        checkpoint = torch.load(ckpt_file, map_location='cpu')
        LOGGER.info(f'loading from checkpoint file: {ckpt_file}')
    return checkpoint


def load_from_resume(run_cfg):
    ckpt_dir = os.path.join(run_cfg.output_dir, 'ckpt')
    previous_optimizer_state = [i for i in os.listdir(ckpt_dir) if i.startswith('optimizer')]
    steps = [i.split('.pt')[0].split('_')[-1] for i in previous_optimizer_state]
    steps = [int(i) for i in steps]
    steps.sort()
    previous_step = steps[-1]
    previous_optimizer_state = f'optimizer_step_{previous_step}.pt'
    previous_model_state = f'model_step_{previous_step}.pt'
    previous_step = int(previous_model_state.split('.')[0].split('_')[-1])
    previous_optimizer_state = os.path.join(ckpt_dir, previous_optimizer_state)
    previous_model_state = os.path.join(ckpt_dir, previous_model_state)

    assert os.path.exists(previous_optimizer_state) and os.path.exists(previous_model_state)
    LOGGER.info("choose previous model: {}".format(previous_model_state))
    LOGGER.info("choose previous optimizer: {}".format(previous_optimizer_state))
    previous_model_state = torch.load(previous_model_state, map_location='cpu')
    previous_optimizer_state = torch.load(previous_optimizer_state, map_location='cpu')
    return previous_model_state, previous_optimizer_state, previous_step
