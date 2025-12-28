import math
import torch
import random
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import LayerNorm as LayerNorm

from model.vision_encoders.evaclip import create_model
from model.vision_encoders.clip.clip import build_model
from model.audio_encoders.beats.beats import BEATsConfig, BEATs
from utils.constants import beats_pretrain_dir, evaclip_pretrain_dir_mapper

str_to_bool_mapper = {
    "yes": True,
    "no": False
}


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_):
        output = gelu(input_)
        return output


class Contra_head(nn.Module):
    def __init__(self, input_dim, contra_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, contra_dim, bias=False)

    def forward(self, cls_token):
        return self.linear(cls_token)


class Match_head(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.activation = GELU()
        self.layernorm = LayerNorm(hidden_size, eps=1e-12)
        self.linear2 = nn.Linear(hidden_size, 2)

    def forward(self, cls_token):
        logits1 = self.activation.forward(self.linear1(cls_token))
        return self.linear2(self.layernorm(logits1))


def disabled_train(self, disable_mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class TokenMasker(nn.Module):
    def __init__(self, mask_token=-1, range_start=-1, range_end=-1):
        super().__init__()
        self.mask_token = mask_token
        self.range = [range_start, range_end]

    def forward(self, tokens, mask_prob):
        tokens = tokens.clone()  # important, must have
        tokens, labels = self.perform_mask(tokens, mask_prob)
        return tokens, labels

    def perform_mask(self, tokens, mask_prob):
        tokens = np.array(tokens.cpu().numpy())

        # generate indicator first:
        mask_indicator = np.zeros(tokens.shape, dtype=np.int64)
        for i in range(len(mask_indicator)):
            while all(mask_indicator[i] == 0):
                for j in range(1, len(mask_indicator[0])):
                    if tokens[i][j] != 0 and random.random() < mask_prob:
                        mask_indicator[i][j] = 1

        labels = -np.ones(tokens.shape, dtype=np.int64) * 100  # -100 ignore idx for nn.CrossEntropyLoss used in BERT
        for i in range(tokens.shape[0]):
            for j in range(tokens.shape[1]):

                if mask_indicator[i][j] == 1:
                    src_token = tokens[i][j]
                    prob = random.random()  # e-6 too much time
                    if prob < 0.8:
                        tokens[i][j] = self.mask_token  # e-6 have no idea why too much
                    elif prob < 0.9:
                        tokens[i][j] = random.choice(list(range(*self.range)))
                        # tokens[i][j] = self.mask_token
                    labels[i][j] = src_token

        tokens = torch.from_numpy(tokens).long().cuda()
        labels = torch.from_numpy(labels).long().cuda()

        return tokens, labels


class MMGeneralModule(nn.Module):
    def __init__(self):
        super().__init__()

    def modify_checkpoint(self, checkpoint):
        new_ckpt = {}
        for k, v in checkpoint.items():
            if 'video' in k:
                new_ckpt[k.replace('video', 'vision')] = v
            elif 'evaclip_model' in k:
                new_ckpt[k.replace('evaclip_model', 'vision_encoder')] = v
            elif 'clip_model' in k:
                new_ckpt[k.replace('clip_model', 'vision_encoder')] = v
            else:
                new_ckpt[k] = v.float()

        checkpoint = new_ckpt

        if self.config.frame_embedding_type == 'adaptive':
            if 'vision_frame_embedding' in checkpoint:
                pretrain_embed = checkpoint['vision_frame_embedding']
                if pretrain_embed.shape[1] != self.config.max_vision_sample_num:
                    pretrain_embed = F.interpolate(pretrain_embed.permute(0, 2, 1), self.config.max_vision_sample_num,
                                                   mode='nearest').permute(0, 2, 1)
                    checkpoint['vision_frame_embedding'] = pretrain_embed

            else:
                pretrain_embed = checkpoint['vision_perceiver.vision_frame_embedding']
                if pretrain_embed.shape[1] != self.config.max_vision_sample_num:
                    pretrain_embed = F.interpolate(pretrain_embed.permute(0, 2, 1), self.config.max_vision_sample_num,
                                                   mode='nearest').permute(0, 2, 1)
                    checkpoint['vision_perceiver.vision_frame_embedding'] = pretrain_embed

            if 'audio_frame_embedding' in checkpoint:
                pretrain_embed_a = checkpoint['audio_frame_embedding']
                if pretrain_embed_a.shape[1] != self.config.max_audio_sample_num:
                    pretrain_embed_a = F.interpolate(pretrain_embed_a.permute(0, 2, 1),
                                                     self.config.max_audio_sample_num, mode='nearest').permute(0, 2, 1)
                    checkpoint['audio_frame_embedding'] = pretrain_embed_a

        if self.config.vision_encoder_type.startswith('clip'):
            vision_width = checkpoint["vision_encoder.visual.positional_embedding"].shape[1]
            vision_layers = len(
                [k for k in checkpoint.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = checkpoint["vision_encoder.visual.conv1.weight"].shape[-1]

            grid_size = round((checkpoint["vision_encoder.visual.positional_embedding"].shape[0] - 1) ** 0.5)

            src = checkpoint["vision_encoder.visual.positional_embedding"]
            src_cls = src[0:1]
            src_oth = src[1:]
            new_grid_size = self.config.vision_resolution // vision_patch_size
            if new_grid_size != grid_size:
                src_oth = F.interpolate(
                    src_oth.reshape(grid_size, grid_size, vision_width).permute(2, 0, 1).unsqueeze(0),
                    (new_grid_size, new_grid_size), mode='bilinear')
                src_oth = src_oth[0].permute(1, 2, 0).reshape(-1, src.shape[-1])
                tgt = torch.cat((src_cls, src_oth), dim=0)
                checkpoint["vision_encoder.visual.positional_embedding"] = tgt

        elif self.config.vision_encoder_type.startswith('evaclip'):
            vision_width = checkpoint["vision_encoder.visual.pos_embed"].shape[2]
            vision_layers = len(
                [k for k in checkpoint.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])

            vision_patch_size = checkpoint["vision_encoder.visual.patch_embed.proj.weight"].shape[-1]
            grid_size = round((checkpoint["vision_encoder.visual.pos_embed"].shape[1] - 1) ** 0.5)

            src = checkpoint["vision_encoder.visual.pos_embed"][0]
            src_cls = src[0:1]
            src_oth = src[1:]
            new_grid_size = self.config.vision_resolution // vision_patch_size
            if new_grid_size != grid_size:
                src_oth = F.interpolate(
                    src_oth.reshape(grid_size, grid_size, vision_width).permute(2, 0, 1).unsqueeze(0),
                    (new_grid_size, new_grid_size), mode='bilinear')
                src_oth = src_oth[0].permute(1, 2, 0).reshape(-1, src.shape[-1])
                tgt = torch.cat((src_cls, src_oth), dim=0)
                checkpoint["vision_encoder.visual.pos_embed"] = tgt.unsqueeze(0)
        else:
            pass

        return checkpoint

    def construct_vision_encoder(self):
        vision_encoder, vision_dim = self.load_clip_model()

        if str_to_bool_mapper[self.config.frozen_vision]:
            for k, v in vision_encoder.named_parameters():
                v.requires_grad = False

            vision_encoder.eval()
            vision_encoder.train = disabled_train
        return vision_encoder, vision_dim

    def construct_audio_encoder(self):
        audio_encoder, audio_dim = self.load_beats_model()

        if str_to_bool_mapper[self.config.frozen_audio]:
            for k, v in self.audio_encoder.named_parameters():
                v.requires_grad = False

            audio_encoder.eval()
            audio_encoder.train = disabled_train
        return audio_encoder, audio_dim

    def load_clip_model(self):
        if self.config.vision_encoder_type.startswith('evaclip'):
            evaclip_model_desc = evaclip_pretrain_dir_mapper.get(self.config.vision_encoder_type)
            vision_dim = evaclip_model_desc["vision_dim"]

            vision_encoder = create_model(evaclip_model_desc["model_name"],
                                          pretrained=evaclip_model_desc["pretrained"],
                                          force_custom_clip=True, image_size=self.config.vision_resolution)

            if str_to_bool_mapper[self.config.checkpointing]:
                vision_encoder.set_grad_checkpointing()
        else:
            # TODO Remove normal CLIP Paths. Not doing currently as not in use
            if self.config.vision_encoder_type == 'clip_vit_base_16':
                clip_weight = torch.jit.load('./pretrained_weights/clip/ViT-B-16.pt', map_location='cpu')
                vision_dim = 768
            elif self.config.vision_encoder_type == 'clip_vit_large_14_336px':
                clip_weight = torch.jit.load('./pretrained_weights/clip/ViT-L-14-336px.pt', map_location='cpu')
                vision_dim = 1024
            elif self.config.vision_encoder_type == 'clip_vit_base_32':
                clip_weight = torch.jit.load('./pretrained_weights/clip/ViT-B-32.pt', map_location='cpu')
                vision_dim = 768
            else:
                clip_weight = torch.jit.load('./pretrained_weights/clip/ViT-B-32.pt', map_location='cpu')
                vision_dim = 768

            clip_weight = clip_weight.state_dict()
            vision_encoder = build_model(clip_weight, self.config.vision_resolution,
                                         str_to_bool_mapper[self.config.checkpointing]).float()
        return vision_encoder, vision_dim

    def load_beats_model(self):
        checkpoint = torch.load(beats_pretrain_dir)
        cfg = BEATsConfig(checkpoint['cfg'])

        audio_encoder = BEATs(cfg, checkpointing=str_to_bool_mapper[self.config.checkpointing])
        audio_encoder.load_state_dict(checkpoint['model'])
        audio_dim = 768
        return audio_encoder, audio_dim

    def forward_vision_encoder(self, vision_pixels):  # b,n,3,h,w
        b, n, _, h, w = vision_pixels.shape
        vision_output = self.vision_encoder.visual(vision_pixels.reshape(b * n, 3, h, w), return_all_features=True)
        vision_output = vision_output.reshape(b, -1, *vision_output.shape[-2:])

        return vision_output  # B , n , x ,C  n = self.frame_num

    def forward_audio_encoder(self, audio_spectrotriangles):
        b, n, h, w, = audio_spectrotriangles.shape
        audio_spectrotriangles = audio_spectrotriangles.reshape(-1, *audio_spectrotriangles.shape[2:])
        audio_output = self.audio_encoder(audio_spectrotriangles)
        audio_output = audio_output.reshape(b, n, -1, audio_output.shape[-1])

        return audio_output

    def pool_vision_for_contra(self, feature):  # feature b ,n ,x ,c
        # always use frame_avg  for retrieval
        feature = feature[:, :, 0]
        feature = torch.mean(feature, dim=1)
        return feature

    def pool_text_for_contra(self, feature):  # feature b ,n ,x, c
        return feature[:, 0]

    def pool_audio_for_contra(self, feature):
        feature = feature.mean(dim=2)
        feature = torch.mean(feature, dim=1)
        return feature

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_multimodal_forward_input_vision(self, vision_output):
        b, n, x, c = vision_output.shape
        vision_output = self.hidden_trans_vision_multimodal(vision_output)

        if self.config.frame_embedding_type == 'adaptive':
            if n != self.vision_frame_embedding.shape[1]:  # testing and interpolate
                vision_frame_embedding = F.interpolate(self.vision_frame_embedding.float().permute(0, 2, 1), n,
                                                       mode='nearest').permute(0, 2, 1).to(self.vision_frame_embedding)
            else:
                vision_frame_embedding = self.vision_frame_embedding

            vision_output = vision_output + vision_frame_embedding.unsqueeze(-2)
        elif self.config.frame_embedding_type == 'none':
            pass

        vision_output = vision_output.reshape(b, -1, self.multimodal_dim)

        if hasattr(self, 'vision_type_embeddings'):  # for three modality
            vision_output = vision_output + self.vision_type_embeddings
        return vision_output

    def get_multimodal_forward_input_audio(self, audio_output):
        b, n, x, c = audio_output.shape
        if n != self.audio_frame_embedding.shape[1]:  # testing and interpolate
            audio_frame_embedding = F.interpolate(self.audio_frame_embedding.permute(0, 2, 1), n,
                                                  mode='nearest').permute(0, 2, 1)
        else:
            audio_frame_embedding = self.audio_frame_embedding
        audio_output = self.hidden_trans_audio_multimodal(audio_output)
        audio_output = audio_output + audio_frame_embedding.unsqueeze(-2)
        audio_output = audio_output.reshape(b, -1, self.multimodal_dim)
        audio_output = audio_output + self.audio_type_embeddings
        return audio_output

    def get_multimodal_forward_input_subtitle(self, subtitle_output):
        subtitle_output = self.hidden_trans_subtitle_multimodal(subtitle_output)
        subtitle_output = subtitle_output + self.subtitle_type_embeddings
        return subtitle_output
