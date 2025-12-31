import torch.distributed as dist
import warnings
import wandb

from dataset_utils.build_dataloader import create_train_dataloaders, create_val_dataloaders
from model.build_model import build_model
from model.build_optimizer import build_optimizer
from utils.args import get_args
from utils.config import update_default_config_with_parse_args
from utils.initialize import initialize_process, log_final_cfg
from utils.pipeline import train, test
from utils.logger import LOGGER

warnings.filterwarnings("ignore")

str_to_bool_mapper = {
    "yes": True,
    "no": False
}


def run_main():
    args = get_args()
    final_config = update_default_config_with_parse_args(args)
    device = initialize_process(final_config)

    if dist.get_rank() == 0:
        n_gpu = log_final_cfg(final_config)

        wandb.init(
            # TODO IMP SET RIGHT BEFORE RUN
            project="Triangle_Reproduce",
            # track hyperparameters and run metadata
            config={
                "desc": f"Train_{final_config.data_cfg.val[0]['name']}",
                "batch-size-train": final_config.data_cfg.train[0]['batch_size'],
                "batch-size-val": final_config.data_cfg.val[0]['batch_size'],
                "ngpus": n_gpu,
                "architecture": "TRIANGLE",
                "dataset": final_config.data_cfg.val[0]['name'],
                "epochs": final_config.data_cfg.train[0]['epoch'],
                "name": final_config.run_cfg.mode + "_finetuning_" + final_config.data_cfg.val[0]['name'] +
                        "_valFrame=" + str(final_config.data_cfg.train[0]['vision_sample_num']),
                "val_frame": str(final_config.data_cfg.val[0]['vision_sample_num']),
                "train_frame": str(final_config.data_cfg.train[0]['vision_sample_num']),
            }
        )

    if final_config.run_cfg.mode == 'training':
        train_loader = create_train_dataloaders(final_config, device)
        val_loaders = create_val_dataloaders(final_config, device)

        for name, loader in val_loaders.items():
            print(f"val_loader: {name} has {len(loader)} batches")

        model, optimizer_ckpt, start_step = build_model(final_config, device)
        optimizer = build_optimizer(model, final_config, optimizer_ckpt)
        if dist.get_rank() == 0:
            for i, pg in enumerate(optimizer.param_groups):
                n = sum(p.numel() for p in pg["params"])
                req = sum(p.numel() for p in pg["params"] if p.requires_grad)
                LOGGER.info(f"group={i} init_lr={pg['init_lr']} n_params={n} req_grad_params={req}")

        if str_to_bool_mapper[final_config.run_cfg.first_eval] or str_to_bool_mapper[final_config.run_cfg.zero_shot]:
            test(model, val_loaders, final_config)
            if str_to_bool_mapper[final_config.run_cfg.zero_shot]:
                return
        train(model, optimizer, train_loader, val_loaders, final_config, start_step=start_step)

    elif final_config.run_cfg.mode == 'testing':
        val_loaders = create_val_dataloaders(final_config, device)

        model, _, _ = build_model(final_config, device)
        print("TESTING MODE")

        test(model, val_loaders, final_config)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    run_main()
