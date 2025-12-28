import sys
import os
import json
from easydict import EasyDict as edict

from utils.utils import (compute_max_vision_sample_num_for_position_embeddings,
                         compute_max_audio_sample_num_for_position_embeddings)

str_to_bool_mapper = {
    "yes": True,
    "no": False
}


def update_default_config_with_parse_args(parse_args):
    file_cfg = edict(json.load(open(parse_args.config)))

    # Args passed while running (not default)
    cmd_cfg_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:] if arg.startswith('--')}

    # Loading default run_config and updating
    run_cfg = edict(json.load(open(file_cfg.run_cfg.default)))
    run_cfg.update(file_cfg.run_cfg)

    for k in cmd_cfg_keys:
        if k in run_cfg:
            run_cfg[k] = getattr(parse_args, k)

    # Loading default model_config and updating
    model_cfg = edict(json.load(open(file_cfg.model_cfg.default)))
    model_cfg.update(file_cfg.model_cfg)

    if parse_args.pretrain_dir:
        pretrain_model_cfg = edict(json.load(open(os.path.join(parse_args.pretrain_dir, 'log', 'hps.json')))).model_cfg
        global_inherit_keys = ['vision_encoder_type', 'pool_video']
        inherit_keys = list(set(global_inherit_keys).union(set(model_cfg.inherit_keys)))
        inherit_model_cfg = edict({k: v for k, v in pretrain_model_cfg.items() if k in inherit_keys})
        model_cfg.update(inherit_model_cfg)

    for k in cmd_cfg_keys:
        if k in model_cfg:
            model_cfg[k] = getattr(parse_args, k)

    # Loading and updating data_config (only possible with single dataset)
    data_cfg = file_cfg['data_cfg']
    for k in cmd_cfg_keys:
        if k.startswith('train_'):
            assert len(data_cfg.train) == 1 or k in ['train_batch_size', 'train_task']
            if k == 'train_epoch':
                data_cfg.train[0].epoch = parse_args.train_epoch
            elif k == 'train_steps':
                data_cfg.train[0].steps = parse_args.train_steps
            elif k == 'train_vision_sample_num':
                data_cfg.train[0].vision_sample_num = parse_args.train_vision_sample_num
            elif k == 'train_batch_size':
                for i in range(len(data_cfg.train)):
                    data_cfg.train[i].batch_size = parse_args.train_batch_size
            elif k == 'train_task':
                for i in range(len(data_cfg.train)):
                    data_cfg.train[i].task = parse_args.train_task
        elif k.startswith('test'):
            # assert len(data_cfg.val)==1
            for i in range(len(data_cfg.val)):
                if k == 'test_batch_size':
                    data_cfg.val[i].batch_size = parse_args.test_batch_size
                elif k == 'test_vision_sample_num':
                    data_cfg.val[i].vision_sample_num = parse_args.test_vision_sample_num
                elif k == 'test_task':
                    data_cfg.val[i].task = parse_args.test_task
        elif k == 'vision_transforms':
            assert len(data_cfg.train) == 1
            assert len(data_cfg.val) == 1
            data_cfg.train[0]['vision_transforms'] = parse_args.vision_transforms
            data_cfg.val[0]['vision_transforms'] = parse_args.vision_transforms

    if str_to_bool_mapper[model_cfg.checkpointing]:
        run_cfg.use_ddp = "no"

    data_cfg.concatenated_nums = getattr(model_cfg, 'concatenated_nums', 1)

    max_vision_sample_num = compute_max_vision_sample_num_for_position_embeddings(data_cfg)
    max_audio_sample_num = compute_max_audio_sample_num_for_position_embeddings(data_cfg)

    model_cfg.max_vision_sample_num = max_vision_sample_num
    model_cfg.max_audio_sample_num = max_audio_sample_num

    final_cfg = edict({'run_cfg': run_cfg,
                       'model_cfg': model_cfg,
                       'data_cfg': data_cfg,
                       'local_rank': parse_args.local_rank})

    return final_cfg
