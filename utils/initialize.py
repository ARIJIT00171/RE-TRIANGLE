import os
import json
import random
import torch
import numpy as np
import torch.distributed as dist

from utils.logger import LOGGER, add_log_to_file


def initialize_process(final_config):
    os.makedirs(os.path.join(final_config.run_cfg.output_dir), exist_ok=True)
    os.makedirs(os.path.join(final_config.run_cfg.output_dir, 'log'), exist_ok=True)
    os.makedirs(os.path.join(final_config.run_cfg.output_dir, 'ckpt'), exist_ok=True)

    local_rank = final_config.local_rank
    print(local_rank)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    print("DEVICE SET")
    dist.init_process_group(backend='nccl')
    if final_config.run_cfg.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(final_config.run_cfg.gradient_accumulation_steps))
    set_random_seed(final_config.run_cfg.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if dist.get_rank() == 0:
        add_log_to_file(os.path.join(final_config.run_cfg.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
    return device


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def log_final_cfg(final_config):
    with open(os.path.join(final_config.run_cfg.output_dir, 'log', 'hps.json'), 'w') as writer:
        json.dump(vars(final_config), writer, indent=4)

    n_gpu = dist.get_world_size()
    return n_gpu
