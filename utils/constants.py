# Pretrain Weights for models
bert_pretrain_dir = '/home/scur1693/Triangle_Refactor/pretrained_weights/bert/bert-base-uncased'

beats_pretrain_dir = '/home/scur1693/Triangle_Refactor/pretrained_weights/beats/BEATs_iter3_plus_AS2M.pt'

evaclip_pretrain_dir_mapper = {
    "evaclip02_base": {
        "model_name": "EVA02-CLIP-B-16",
        "pretrained": "/home/scur1693/Triangle_Refactor/pretrained_weights/clip/EVA02_CLIP_B_psz16_s8B.pt",
        "vision_dim": 768
    },
    "evaclip02_base_self": {
        "model_name": "EVA02-CLIP-B-16",
        "pretrained": "/home/scur1693/Triangle_Refactor/pretrained_weights/clip/EVA02_B_psz14to16.pt",
        "vision_dim": 768
    },
    "evaclip02_large": {
            "model_name": "EVA02-CLIP-L-14",
            "pretrained": "/home/scur1693/Triangle_Refactor/pretrained_weights/clip/EVA02_CLIP_L_psz14_s4B.pt",
            "vision_dim": 1024
        },
    "evaclip02_bige": {
            "model_name": "EVA02-CLIP-bigE-14-plus",
            "pretrained": "/home/scur1693/Triangle_Refactor/pretrained_weights/clip/EVA02_CLIP_E_psz14_plus_s9B.pt",
            "vision_dim": 1792
        },
    "evaclip01_giant": {
            "model_name": "EVA01-CLIP-g-14",
            "pretrained": "/home/scur1693/Triangle_Refactor/pretrained_weights/eva_clip/EVA01_CLIP_g_14_psz14_s11B.pt",
            "vision_dim": 1408
        },
}
