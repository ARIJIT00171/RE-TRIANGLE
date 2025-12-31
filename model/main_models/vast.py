import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn import LayerNorm as LayerNorm
from transformers import BertTokenizer
from easydict import EasyDict as edict

from model.text_encoders.bert.bert import BertForMaskedLM, BertConfig
from model.main_models.general_module import TokenMasker, MMGeneralModule, Contra_head, Match_head
from utils.distributed import all_gather_with_grad, concat_all_gather
from utils.constants import bert_pretrain_dir
from utils.logger import LOGGER

str_to_bool_mapper = {
    "yes": True,
    "no": False
}


class VAST(MMGeneralModule):
    """ VLP pretraining """
    def __init__(self, config, run_cfg):
        super().__init__()

        self.config = config
        self.run_cfg = run_cfg
        self.vision_encoder, self.vision_dim = self.construct_vision_encoder()
        self.audio_encoder, self.audio_dim = self.construct_audio_encoder()
        self.multimodal_encoder, self.multimodal_dim, self.text_masker = self.construct_multimodal_encoder()

        self.contra_dim = self.config.contra_dim
        self.contra_head_t = Contra_head(self.multimodal_dim, self.contra_dim)
        self.contra_head_s = Contra_head(self.multimodal_dim, self.contra_dim)
        self.contra_head_v = Contra_head(self.vision_dim, self.contra_dim)
        self.contra_head_a = Contra_head(self.audio_dim, self.contra_dim)
        self.contra_head_d = Contra_head(self.vision_dim, self.contra_dim)
        self.contra_head_va = nn.Linear(self.vision_dim + self.audio_dim, self.contra_dim)
        self.contra_head_vs = nn.Linear(self.vision_dim + self.multimodal_dim, self.contra_dim)
        self.contra_head_vas = nn.Linear(self.vision_dim + self.audio_dim + self.multimodal_dim, self.contra_dim)
        self.contra_temp = nn.Parameter(torch.tensor(0.07))
        self.itm_head = Match_head(self.multimodal_dim)
        self.vision_frame_embedding = nn.Parameter(
            0.02 * torch.randn(1, self.config.max_vision_sample_num, self.multimodal_dim))
        self.audio_frame_embedding = nn.Parameter(
            0.02 * torch.randn(1, self.config.max_audio_sample_num, self.multimodal_dim))
        self.hidden_trans_vision_multimodal = nn.Sequential(nn.Linear(self.vision_dim, self.multimodal_dim),
                                                            LayerNorm(self.multimodal_dim, eps=1e-12))
        self.hidden_trans_audio_multimodal = nn.Sequential(nn.Linear(self.audio_dim, self.multimodal_dim),
                                                           LayerNorm(self.multimodal_dim, eps=1e-12))
        self.hidden_trans_subtitle_multimodal = nn.Sequential(nn.Linear(self.multimodal_dim, self.multimodal_dim),
                                                              LayerNorm(self.multimodal_dim, eps=1e-12))
        self.vision_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim))
        self.audio_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim))
        self.subtitle_type_embeddings = nn.Parameter(0.02 * torch.randn(1, 1, self.multimodal_dim))
        self.beam_size = config.beam_size
        self.itm_ratio = config.itm_ratio
        self.max_omni_caption_len = config.max_omni_caption_len
        self.max_caption_len = config.max_caption_len
        self.max_subtitle_len = config.max_subtitle_len

    def construct_multimodal_encoder(self):
        LOGGER.info("Loading BertForMaskedLM Model with Cross Attention weights")
        bertconfig = BertConfig.from_pretrained(bert_pretrain_dir)
        bertconfig.add_cross_attention = True
        bertconfig.is_decoder = True

        multimodal_encoder = BertForMaskedLM.from_pretrained(bert_pretrain_dir, config=bertconfig)
        multimodal_dim = 768

        if str_to_bool_mapper[self.config.checkpointing]:
            multimodal_encoder._set_gradient_checkpointing(multimodal_encoder.bert.encoder, True)

        multimodal_encoder.tokenizer = BertTokenizer.from_pretrained(bert_pretrain_dir)
        multimodal_encoder.tokenizer.bos_token_id = (multimodal_encoder.tokenizer.convert_tokens_to_ids(['[CLS]']))[0]
        multimodal_encoder.tokenizer.eos_token_id = (multimodal_encoder.tokenizer.convert_tokens_to_ids(['[SEP]']))[0]
        multimodal_encoder.tokenizer.pad_token_id = (multimodal_encoder.tokenizer.convert_tokens_to_ids(['[PAD]']))[0]
        multimodal_encoder.tokenizer.mask_token_id = (multimodal_encoder.tokenizer.convert_tokens_to_ids(['[MASK]']))[0]
        text_masker = TokenMasker(mask_token=multimodal_encoder.tokenizer.mask_token_id, range_start=106,
                                  range_end=30522)
        LOGGER.info("BertForMaskedLM Model with Cross Attention weights loaded successfully")

        return multimodal_encoder, multimodal_dim, text_masker

    def batch_get(self, batch, key):
        if key in batch:
            return batch[key]
        elif key == 'caption_tokens':
            caption_tokens = self.multimodal_encoder.tokenizer(batch.raw_captions,
                                                               padding="max_length",
                                                               truncation=True,
                                                               max_length=self.max_caption_len,
                                                               return_tensors="pt").to(torch.device('cuda'))
            batch[key] = caption_tokens
        elif key == 'subtitle_tokens':
            subtitle_tokens = self.multimodal_encoder.tokenizer(batch.raw_subtitles,
                                                                padding="max_length",
                                                                truncation=True,
                                                                max_length=self.max_subtitle_len,
                                                                return_tensors="pt")
            subtitle_tokens = subtitle_tokens.to(torch.device('cuda'))
            batch[key] = subtitle_tokens
        elif key == 'vision_caption_tokens':
            caption_tokens = self.multimodal_encoder.tokenizer(batch.vision_captions,
                                                               padding="max_length",
                                                               truncation=True,
                                                               max_length=self.max_caption_len,
                                                               return_tensors="pt")
            caption_tokens = caption_tokens.to(torch.device('cuda'))
            batch[key] = caption_tokens
        elif key == 'audio_caption_tokens':
            caption_tokens = self.multimodal_encoder.tokenizer(batch.audio_captions,
                                                               padding="max_length",
                                                               truncation=True,
                                                               max_length=self.max_caption_len,
                                                               return_tensors="pt")
            caption_tokens = caption_tokens.to(torch.device('cuda'))
            batch[key] = caption_tokens
        elif key == 'omni_caption_tokens':
            caption_tokens = self.multimodal_encoder.tokenizer(batch.omni_captions,
                                                               padding="max_length",
                                                               truncation=True,
                                                               max_length=self.max_omni_caption_len,
                                                               return_tensors="pt")
            caption_tokens = caption_tokens.to(torch.device('cuda'))
            batch[key] = caption_tokens
        elif key == 'caption_output':
            caption_tokens = self.batch_get(batch, 'caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_output = self.multimodal_encoder.bert(input_ids=input_ids,
                                                          attention_mask=attention_mask).last_hidden_state
            batch[key] = caption_output
        elif key == 'vision_caption_output':
            caption_tokens = self.batch_get(batch, 'vision_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_output = self.multimodal_encoder.bert(input_ids=input_ids,
                                                          attention_mask=attention_mask).last_hidden_state
            batch[key] = caption_output
        elif key == 'audio_caption_output':
            caption_tokens = self.batch_get(batch, 'audio_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_output = self.multimodal_encoder.bert(input_ids=input_ids,
                                                          attention_mask=attention_mask).last_hidden_state
            batch[key] = caption_output
        elif key == 'subtitle_output':
            subtitle_tokens = self.batch_get(batch, 'subtitle_tokens')
            input_ids = subtitle_tokens.input_ids
            attention_mask = subtitle_tokens.attention_mask
            subtitle_output = self.multimodal_encoder.bert(input_ids=input_ids,
                                                           attention_mask=attention_mask).last_hidden_state
            batch[key] = subtitle_output
        elif key == 'vision_output':
            vision_output = self.forward_vision_encoder(batch.vision_pixels)
            batch[key] = vision_output
        elif key == 'depth_output':
            depth_output = self.forward_vision_encoder(batch.depth_pixels)
            batch[key] = depth_output
        elif key == 'audio_output':
            audio_output = self.forward_audio_encoder(batch.audio_spectrotriangles)
            batch[key] = audio_output
        elif key == 'condition_feats_v':
            vision_output = self.batch_get(batch, 'vision_output')
            condition_feats_v = self.get_multimodal_forward_input_vision(vision_output)
            batch[key] = condition_feats_v
        elif key == 'condition_feats_d':
            vision_output = self.batch_get(batch, 'depth_output')
            condition_feats_d = self.get_multimodal_forward_input_vision(vision_output)
            batch[key] = condition_feats_d
        elif key == 'condition_feats_a':
            audio_output = self.batch_get(batch, 'audio_output')
            condition_feats_a = self.get_multimodal_forward_input_audio(audio_output)
            batch[key] = condition_feats_a
        elif key == 'condition_feats_s':
            subtitle_output = self.batch_get(batch, 'subtitle_output')
            condition_feats_s = self.get_multimodal_forward_input_subtitle(subtitle_output)
            batch[key] = condition_feats_s
        elif key == 'condition_feats_va':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_a = self.batch_get(batch, 'condition_feats_a')
            condition_feats_va = torch.cat((condition_feats_v, condition_feats_a), dim=1)
            batch[key] = condition_feats_va
        elif key == 'condition_feats_vs':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_s = self.batch_get(batch, 'condition_feats_s')
            condition_feats_vs = torch.cat((condition_feats_v, condition_feats_s), dim=1)
            batch[key] = condition_feats_vs
        elif key == 'condition_feats_vas':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_a = self.batch_get(batch, 'condition_feats_a')
            condition_feats_s = self.batch_get(batch, 'condition_feats_s')
            condition_feats_vas = torch.cat((condition_feats_v, condition_feats_a, condition_feats_s), dim=1)
            batch[key] = condition_feats_vas
        elif key == 'condition_feats_vasd':
            condition_feats_v = self.batch_get(batch, 'condition_feats_v')
            condition_feats_a = self.batch_get(batch, 'condition_feats_a')
            condition_feats_s = self.batch_get(batch, 'condition_feats_s')
            condition_feats_d = self.batch_get(batch, 'condition_feats_d')
            condition_feats_vas = torch.cat(
                (condition_feats_v, condition_feats_a, condition_feats_s, condition_feats_d), dim=1)
            batch[key] = condition_feats_vas
        elif key == 'feat_v':
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            feat_v = self.contra_head_v.forward(vision_output_pooled)
            feat_v = F.normalize(feat_v, dim=-1)
            batch[key] = feat_v
        elif key == 'feat_d':
            depth_output = self.batch_get(batch, 'depth_output')
            depth_output_pooled = self.pool_vision_for_contra(depth_output)
            feat_d = self.contra_head_d.forward(depth_output_pooled)  # TODO do we have to use a contra head depth?
            feat_d = F.normalize(feat_d, dim=-1)
            batch[key] = feat_d
        elif key == 'feat_a':
            audio_output = self.batch_get(batch, 'audio_output')
            audio_output_pooled = self.pool_audio_for_contra(audio_output)
            feat_a = self.contra_head_a.forward(audio_output_pooled)
            feat_a = F.normalize(feat_a, dim=-1)
            batch[key] = feat_a
        elif key == 'feat_s':
            subtitle_output = self.batch_get(batch, 'subtitle_output')
            subtitle_output_pooled = self.pool_text_for_contra(subtitle_output)
            feat_s = self.contra_head_s.forward(subtitle_output_pooled)
            feat_s = F.normalize(feat_s, dim=-1)
            batch[key] = feat_s
        elif key == 'feat_t':
            caption_output = self.batch_get(batch, 'caption_output')
            caption_output_pooled = self.pool_text_for_contra(caption_output)
            feat_t = self.contra_head_t.forward(caption_output_pooled)
            feat_t = F.normalize(feat_t, dim=-1)
            batch[key] = feat_t
        elif key == 'feat_va':
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            audio_output = self.batch_get(batch, 'audio_output')
            audio_output_pooled = self.pool_audio_for_contra(audio_output)
            feat_va = torch.cat((vision_output_pooled, audio_output_pooled), dim=1)
            feat_va = self.contra_head_va(feat_va)
            feat_va = F.normalize(feat_va, dim=-1)
            batch[key] = feat_va
        elif key == 'feat_vs':
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            subtitle_output = self.batch_get(batch, 'subtitle_output')
            subtitle_output_pooled = self.pool_text_for_contra(subtitle_output)
            feat_vs = torch.cat((vision_output_pooled, subtitle_output_pooled), dim=1)
            feat_vs = self.contra_head_vs(feat_vs)
            feat_vs = F.normalize(feat_vs, dim=-1)
            batch[key] = feat_vs
        elif key == 'feat_vas':
            vision_output = self.batch_get(batch, 'vision_output')
            vision_output_pooled = self.pool_vision_for_contra(vision_output)
            audio_output = self.batch_get(batch, 'audio_output')
            audio_output_pooled = self.pool_audio_for_contra(audio_output)
            subtitle_output = self.batch_get(batch, 'subtitle_output')
            subtitle_output_pooled = self.pool_text_for_contra(subtitle_output)
            feat_vas = torch.cat((vision_output_pooled, audio_output_pooled, subtitle_output_pooled), dim=1)
            feat_vas = self.contra_head_vas(feat_vas)
            feat_vas = F.normalize(feat_vas, dim=-1)
            batch[key] = feat_vas
        elif key == 'feat_t_omni_caption':
            caption_tokens = self.batch_get(batch, 'omni_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_tokens = self.multimodal_encoder.bert(input_ids=input_ids,
                                                          attention_mask=attention_mask).last_hidden_state
            caption_tokens_pooled = self.pool_text_for_contra(caption_tokens)
            feat_t = self.contra_head_t.forward(caption_tokens_pooled)
            feat_t = F.normalize(feat_t, dim=-1)
            batch[key] = feat_t
        elif key == 'feat_t_vision_caption':
            caption_tokens = self.batch_get(batch, 'vision_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_tokens = self.multimodal_encoder.bert(input_ids=input_ids,
                                                          attention_mask=attention_mask).last_hidden_state
            caption_tokens_pooled = self.pool_text_for_contra(caption_tokens)
            feat_t = self.contra_head_t.forward(caption_tokens_pooled)
            feat_t = F.normalize(feat_t, dim=-1)
            batch[key] = feat_t
        elif key == 'feat_t_audio_caption':
            caption_tokens = self.batch_get(batch, 'audio_caption_tokens')
            input_ids = caption_tokens.input_ids
            attention_mask = caption_tokens.attention_mask
            caption_tokens = self.multimodal_encoder.bert(input_ids=input_ids,
                                                          attention_mask=attention_mask).last_hidden_state
            caption_tokens_pooled = self.pool_text_for_contra(caption_tokens)
            feat_t = self.contra_head_t.forward(caption_tokens_pooled)
            feat_t = F.normalize(feat_t, dim=-1)
            batch[key] = feat_t
        return batch[key]

    def forward(self, batch, tasks, compute_loss=True, global_step=0):
        # other datasets pretraining or finetuning
        output_ls = []
        task_ls = tasks.split('_')

        for task in task_ls:
            if task.startswith('ret'):
                ret_dict = self.forward_ret(batch, task, compute_loss=compute_loss, global_step=global_step)
                output_ls.append(ret_dict)
            elif task.startswith('cap'):
                cap_dict = self.forward_cap(batch, task, compute_loss=compute_loss)
                output_ls.append(cap_dict)
            elif task.startswith('qa'):
                qa_dict = self.forward_qa(batch, task, compute_loss=compute_loss)
                output_ls.append(qa_dict)
            else:
                raise NotImplementedError

        output_dict = {k: v for dic in output_ls for k, v in dic.items()}
        return output_dict

    def forward_vast27m(self, batch, task):
        output_ls = []
        task_ls = task.split('_')

        for task in task_ls:
            if task.startswith('ret'):
                ret_dict = self.forward_ret_vast27m(batch, task)
                output_ls.append(ret_dict)
            elif task.startswith('cap'):
                cap_dict = self.forward_cap_vast27m(batch, task)
                output_ls.append(cap_dict)
            else:
                raise NotImplementedError

        output_dict = {k: v for dic in output_ls for k, v in dic.items()}
        return output_dict

    def compute_slice_scores(self, slice_multimodal_vision_input, slice_input_ids, slice_attention_mask):

        slice_output = self.multimodal_encoder.bert(input_ids=slice_input_ids,
                                                    attention_mask=slice_attention_mask,
                                                    encoder_hidden_states=slice_multimodal_vision_input
                                                    ).last_hidden_state
        slice_scores = F.softmax(self.itm_head.forward(slice_output[:, 0]), dim=1)[:, 1]

        return slice_scores

    def forward_ret(self, batch, task, compute_loss=True, global_step=0):
        batch = edict(batch)
        if isinstance(batch.raw_captions[0], list):  # test
            batch.raw_captions = [i for j in batch.raw_captions for i in j]
        subtasks = task.split('%')[1:]
        if compute_loss:
            loss_dict = {}
            loss_itc = []
            loss_itm = []
            loss_area = []

            # Extract text features
            feat_t = self.batch_get(batch, 'feat_t')
            feat_t_all = concat_all_gather(feat_t)

            if dist.get_rank() == 0 and global_step % 100 == 0:
                LOGGER.info(f"feat_t requires_grad={feat_t.requires_grad}, grad_fn={feat_t.grad_fn}")
                LOGGER.info(f"feat_t_all requires_grad={feat_t_all.requires_grad}, grad_fn={feat_t_all.grad_fn}")

            caption_tokens = self.batch_get(batch, 'caption_tokens')
            input_ids, attention_mask = caption_tokens.input_ids, caption_tokens.attention_mask
            input_ids_collate = concat_all_gather(input_ids)
            attention_mask_collate = concat_all_gather(attention_mask)

            for task in subtasks:
                # COMPUTE ITC

                feat_cond = self.batch_get(batch, f'feat_{task[1:]}')
                feat_cond_all = concat_all_gather(feat_cond)

                sim_cond2t = torch.matmul(feat_cond, feat_t_all.permute(1, 0))
                sim_cond2t = sim_cond2t / self.contra_temp
                sim_t2cond = torch.matmul(feat_t, feat_cond_all.permute(1, 0))
                sim_t2cond = sim_t2cond / self.contra_temp

                rank = dist.get_rank()
                bs = feat_t.size(0)
                targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(feat_cond.device)

                loss = (F.cross_entropy(sim_cond2t, targets, label_smoothing=0.1)
                        + F.cross_entropy(sim_t2cond, targets, label_smoothing=0.1)) / 2
                loss_itc.append(loss)

                # COMPUTE ITM

                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
                condition_feats_collate = all_gather_with_grad(condition_feats)

                with torch.no_grad():
                    weights_t2cond = F.softmax(sim_t2cond, dim=1) + 1e-4
                    weights_t2cond[:, rank * bs: rank * bs + bs].fill_diagonal_(0)

                    weights_cond2t = F.softmax(sim_cond2t, dim=1) + 1e-4
                    weights_cond2t[:, rank * bs: rank * bs + bs].fill_diagonal_(0)

                condition_feats_neg = []
                for b in range(bs):
                    neg_idx = torch.multinomial(weights_t2cond[b], 1).item()
                    condition_feats_neg.append(condition_feats_collate[neg_idx])
                condition_feats_neg = torch.stack(condition_feats_neg, dim=0)

                text_ids_neg = []
                text_atts_neg = []
                for b in range(bs):
                    neg_idx = torch.multinomial(weights_cond2t[b], 1).item()
                    text_ids_neg.append(input_ids_collate[neg_idx])
                    text_atts_neg.append(attention_mask_collate[neg_idx])

                text_ids_neg = torch.stack(text_ids_neg, dim=0)
                text_atts_neg = torch.stack(text_atts_neg, dim=0)

                input_ids_1 = torch.cat((input_ids, input_ids, text_ids_neg), dim=0)
                attention_mask_1 = torch.cat((attention_mask, attention_mask, text_atts_neg), dim=0)

                condition_feats = torch.cat((condition_feats, condition_feats_neg, condition_feats), dim=0)
                output = self.multimodal_encoder.bert(input_ids=input_ids_1,
                                                      attention_mask=attention_mask_1,
                                                      encoder_hidden_states=condition_feats).last_hidden_state
                batch_size = condition_feats_neg.shape[0]
                cls = output[:, 0].float()
                logits = self.itm_head.forward(cls)
                ground_truth = torch.zeros(batch_size * 3, dtype=torch.long, device=logits.device)
                ground_truth[:batch_size] = 1
                loss = F.cross_entropy(logits, ground_truth)  # itm (dtm)
                loss_itm.append(self.itm_ratio * loss)
                loss_area.append(torch.zeros((), device=logits.device))

            loss_itc = sum(loss_itc) / len(loss_itc)
            loss_dict['loss_itc'] = loss_itc
            loss_itm = sum(loss_itm) / len(loss_itm)
            loss_dict['loss_itm'] = loss_itm
            loss_area = sum(loss_area) / len(loss_area)
            loss_dict['loss_area'] = loss_area
            return loss_dict

        else:
            evaluation_dict = {}
            feat_t = self.batch_get(batch, 'feat_t')
            evaluation_dict['feat_t'] = feat_t
            caption_tokens = self.batch_get(batch, 'caption_tokens')
            evaluation_dict['input_ids'] = caption_tokens.input_ids
            evaluation_dict['attention_mask'] = caption_tokens.attention_mask
            for task in subtasks:
                assert task in ['tv', 'ta', 'tva', 'tvs', 'tvas', 'tvasd']

                # compute_itc
                feat_cond = self.batch_get(batch, f'feat_{task[1:]}')
                evaluation_dict[f'feat_cond_{task[1:]}'] = feat_cond

                # compute_itm
                condition_feats = self.batch_get(batch, f'condition_feats_{task[1:]}')
                evaluation_dict[f'condition_feats_{task[1:]}'] = condition_feats
            return evaluation_dict
