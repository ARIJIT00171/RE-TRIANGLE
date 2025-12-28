import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm
from easydict import EasyDict as edict
import wandb

from adhoc_scripts.ir_logging import save_qrels_t2v, save_qrels_v2t, save_trec_run
from utils.logger import LOGGER
from utils.distributed import all_gather_list_cpu, ddp_allgather_cpu
from utils.utils import NoOp, area_computation_cpu_chunked, area_computation_cpu_chunked_cos

str_to_bool_mapper = {
    "yes": True,
    "no": False
}


def evaluate_mm(model, val_dataloaders, final_config, global_step):
    eval_log = {}
    model.eval()
    for task, loader in val_dataloaders.items():
        LOGGER.info(f"evaluate on {task} task")
        val_log = evaluate_single(model, loader, task.split('--')[0], final_config, global_step, task.split('--')[1])
        eval_log[task] = val_log
    model.train()
    return eval_log


@torch.no_grad()
def evaluate_single(model, val_loader, task, final_config, global_step, dset_name):
    LOGGER.info("start running {} validation...".format(task))
    tasks = task.split('_')

    output_ls = []
    for task in tasks:
        ret_dict = evaluate_ret(model, task, val_loader, final_config, global_step=global_step, dset_name=dset_name)
        output_ls.append(ret_dict)

    output_dict = {k: v for dic in output_ls for k, v in dic.items()}
    return output_dict


@torch.no_grad()
def evaluate_ret(model, tasks, val_loader, final_config, global_step=0, dset_name="unknown"):
    val_log = {}
    ids = []
    ids_txt = []
    input_ids = []
    attention_mask = []

    subtasks = tasks.split('%')[1:]
    store_dict = {}
    feat_t = []
    feat_a = []
    feat_v = []
    feat_s = []
    feat_d = []

    for task in subtasks:
        if final_config.model_cfg.model_type == "vast":
            store_dict[f'feat_cond_{task[1:]}'] = []
        store_dict[f'condition_feats_{task[1:]}'] = []

    LOGGER.info(f"val_loader size on {len(val_loader)}")
    for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
        batch = edict(batch)
        evaluation_dict = model(batch, tasks=tasks, compute_loss=False, global_step=global_step)

        feat_t.append(evaluation_dict['feat_t'].detach().float().cpu())
        if 'feat_a' in evaluation_dict.keys():
            feat_a.append(evaluation_dict['feat_a'].detach().float().cpu())
        if 'feat_v' in evaluation_dict.keys():
            feat_v.append(evaluation_dict['feat_v'].detach().float().cpu())
        if 'feat_s' in evaluation_dict.keys():
            feat_s.append(evaluation_dict['feat_s'].detach().float().cpu())
        if 'feat_d' in evaluation_dict.keys():
            feat_d.append(evaluation_dict['feat_d'].detach().float().cpu())

        input_ids.append(evaluation_dict['input_ids'].detach().cpu().long())
        attention_mask.append(evaluation_dict['attention_mask'].detach().cpu().long())
        ids += batch.ids

        if 'ids_txt' in batch:
            if isinstance(batch['ids_txt'][0], list):
                ids_txt += [j for i in batch.ids_txt for j in i]
            else:
                ids_txt += batch.ids_txt
        else:
            ids_txt += batch.ids

        for task in subtasks:
            if final_config.model_cfg.model_type == "vast":
                store_dict[f'feat_cond_{task[1:]}'].append(
                    evaluation_dict[f'feat_cond_{task[1:]}'].detach().float().cpu())
            store_dict[f'condition_feats_{task[1:]}'].append(
                evaluation_dict[f'condition_feats_{task[1:]}'].detach().float().cpu())

    ids = [j for i in all_gather_list_cpu(ids) for j in i]
    ids_txt = [j for i in all_gather_list_cpu(ids_txt) for j in i]
    input_ids = torch.cat([i for i in input_ids], dim=0)
    input_ids = ddp_allgather_cpu(input_ids)
    attention_mask = torch.cat([i for i in attention_mask], dim=0)
    attention_mask = ddp_allgather_cpu(attention_mask)

    feat_t = torch.cat(feat_t, dim=0)
    feat_t = ddp_allgather_cpu(feat_t)

    if final_config.model_cfg.model_type == "vast":
        trec_dir = os.path.join(final_config.run_cfg.output_dir, "trec_runs", dset_name)

        trec_itc_t2v = {}  # text -> video
        trec_itc_v2t = {}  # video -> text
        # Compute ITC Score
        for task in subtasks:
            store_dict[f'feat_cond_{task[1:]}'] = torch.cat(store_dict[f'feat_cond_{task[1:]}'], dim=0)
            store_dict[f'feat_cond_{task[1:]}'] = ddp_allgather_cpu(store_dict[f'feat_cond_{task[1:]}'])
            score_matrix_t_cond = torch.matmul(feat_t, store_dict[f'feat_cond_{task[1:]}'].permute(1, 0))

            store_dict[f'score_matrix_t_cond_{task[1:]}'] = score_matrix_t_cond

            # For TREC: T2V = (num_txt, num_vid); V2T = transpose
            trec_itc_t2v[task[1:]] = score_matrix_t_cond.detach()
            trec_itc_v2t[task[1:]] = score_matrix_t_cond.detach().T

            log = compute_metric_ret(score_matrix_t_cond, ids, ids_txt, direction='forward')
            log = {k.replace('forward', 'video'): v for k, v in log.items()}
            # if model.config.ret_bidirection_evaluation:
            log2 = compute_metric_ret(score_matrix_t_cond, ids, ids_txt, direction='backward')
            log2 = {k.replace('backward', 'txt'): v for k, v in log2.items()}
            log.update(log2)

            val_log[f'ret_itc_{task[1:]}'] = log

        trec_itm_t2v = {}  # text -> video
        trec_itm_v2t = {}  # video -> text
        # Compute ITM Score
        for task in subtasks:
            store_dict[f'condition_feats_{task[1:]}'] = torch.cat(store_dict[f'condition_feats_{task[1:]}'], dim=0)
            itm_rerank_num = model.config.itm_rerank_num
            score_matrix = refine_score_matrix(store_dict[f'condition_feats_{task[1:]}'], input_ids, attention_mask,
                                               store_dict[f'score_matrix_t_cond_{task[1:]}'], model, itm_rerank_num,
                                               direction='forward')
            trec_itm_t2v[task[1:]] = score_matrix.detach()
            log = compute_metric_ret(score_matrix, ids, ids_txt, direction='forward')
            log = {k.replace('forward', 'video'): v for k, v in log.items()}

            # if model.config.ret_bidirection_evaluation:
            score_matrix = refine_score_matrix(store_dict[f'condition_feats_{task[1:]}'], input_ids, attention_mask,
                                               store_dict[f'score_matrix_t_cond_{task[1:]}'], model, itm_rerank_num,
                                               direction='backward')
            trec_itm_v2t[task[1:]] = score_matrix.detach()
            log2 = compute_metric_ret(score_matrix, ids, ids_txt, direction='backward')
            log2 = {k.replace('backward', 'txt'): v for k, v in log2.items()}
            log.update(log2)

            val_log[f'ret_itm_{task[1:]}'] = log

        if dist.get_rank() == 0:
            # ---------------------------------------------------
            # QRELS + TREC RUNS
            # ---------------------------------------------------
            os.makedirs(trec_dir, exist_ok=True)

            # 1) QRELS: separate files for T2V and V2T
            qrels_t2v_path = os.path.join(str(trec_dir), "qrels_t2v.txt")
            qrels_v2t_path = os.path.join(str(trec_dir), "qrels_v2t.txt")
            save_qrels_t2v(ids, ids_txt, qrels_t2v_path)
            save_qrels_v2t(ids, ids_txt, qrels_v2t_path)

            for task in subtasks:
                # ---- ITC TREC runs ----
                itc_t2v = trec_itc_t2v[task[1:]]
                itc_v2t = trec_itc_v2t[task[1:]]

                save_trec_run(
                    score_matrix=itc_t2v,
                    query_ids=ids_txt,  # T2V: text queries
                    doc_ids=ids,  # video docs
                    out_path=os.path.join(trec_dir, f"itc_{task[1:]}_T2V.txt"),
                    run_name=f"itc_{task[1:]}_T2V",
                )

                save_trec_run(
                    score_matrix=itc_v2t,
                    query_ids=ids,  # V2T: video queries
                    doc_ids=ids_txt,  # text docs
                    out_path=os.path.join(trec_dir, f"itc_{task[1:]}_V2T.txt"),
                    run_name=f"itc_{task[1:]}_V2T",
                )

                # ---- ITM TREC runs (only T2V always; V2T if computed) ----
                # if task in trec_itm_t2v:
                itm_t2v = trec_itm_t2v[task[1:]]
                save_trec_run(
                    score_matrix=itm_t2v,
                    query_ids=ids_txt,
                    doc_ids=ids,
                    out_path=os.path.join(trec_dir, f"itm_{task[1:]}_T2V.txt"),
                    run_name=f"itm_{task[1:]}_T2V",
                )

                # if model.config.ret_bidirection_evaluation and task in trec_itm_v2t:
                itm_v2t = trec_itm_v2t[task[1:]].T
                save_trec_run(
                    score_matrix=itm_v2t,
                    query_ids=ids,
                    doc_ids=ids_txt,
                    out_path=os.path.join(trec_dir, f"itm_{task[1:]}_V2T.txt"),
                    run_name=f"itm_{task[1:]}_V2T",
                )
            wandb.log(val_log)
    else:
        feat_a = torch.cat(feat_a, dim=0)
        feat_a = ddp_allgather_cpu(feat_a)

        feat_v = torch.cat(feat_v, dim=0)
        feat_v = ddp_allgather_cpu(feat_v)

        # area = area_computation(feat_t, feat_v, feat_a)
        if final_config.model_cfg.model_type == "triangle":
            area = area_computation_cpu_chunked(feat_t, feat_v, feat_a)
        else:
            alpha = F.softplus(model.area_alpha_raw).detach().cpu().item()
            if dist.get_rank() == 0 and global_step % 100 == 0:
                LOGGER.info(f"Alpha value evaluation: {alpha}")
            area = area_computation_cpu_chunked_cos(feat_t, feat_v, feat_a, alpha=alpha)

        min_values_area = torch.min(area, 1).values
        mean_values_area = torch.mean(min_values_area)
        val_log[f"area_value"] = {"value": mean_values_area.item()}

        log = compute_metric_ret_area(area, ids, ids_txt, direction='forward')
        log = {k.replace('forward', 'area_T2D'): v for k, v in log.items()}

        val_log[f'ret_area_forward'] = log

        # TODO doubt direction should be backward
        log = compute_metric_ret_area(area.T, ids, ids_txt, direction='forward')
        log = {k.replace('backward', 'area_D2T'): v for k, v in log.items()}

        val_log[f'ret_area_backward'] = log

        for task in subtasks:
            store_dict[f'condition_feats_{task[1:]}'] = torch.cat(store_dict[f'condition_feats_{task[1:]}'], dim=0)
            itm_rerank_num = model.config.itm_rerank_num

            score_matrix = refine_score_matrix(store_dict[f'condition_feats_{task[1:]}'], input_ids, attention_mask,
                                               -area, model, itm_rerank_num, direction='forward')
            itm_area_T2V_mat = score_matrix  # for TREC
            log = compute_metric_ret(score_matrix, ids, ids_txt, direction='forward')
            log = {k.replace('forward', 'area_ITM_T2D'): v for k, v in log.items()}

            score_matrix = refine_score_matrix(store_dict[f'condition_feats_{task[1:]}'], input_ids, attention_mask,
                                               -area, model, itm_rerank_num, direction='backward')
            itm_area_V2T_mat = score_matrix  # for TREC
            log2 = compute_metric_ret(score_matrix, ids, ids_txt, direction='backward')
            log2 = {k.replace('backward', 'area_ITM_D2T'): v for k, v in log2.items()}
            log.update(log2)

            val_log[f'ret_itm_area'] = log

        score_cosine_tv = torch.matmul(feat_t, feat_v.permute(1, 0))
        cosine_tv = compute_metric_ret(score_cosine_tv, ids, ids_txt, direction='forward')
        val_log[f'cosine_tv'] = cosine_tv

        score_cosine_vt = torch.matmul(feat_v, feat_t.permute(1, 0))
        cosine_vt = compute_metric_ret(score_cosine_vt, ids, ids_txt, direction='forward')
        val_log[f'cosine_vt'] = cosine_vt

        score_cosine_ta = torch.matmul(feat_t, feat_a.permute(1, 0))
        cosine_ta = compute_metric_ret(score_cosine_ta, ids, ids_txt, direction='forward')
        val_log[f'cosine_ta'] = cosine_ta

        score_cosine_at = torch.matmul(feat_a, feat_t.permute(1, 0))
        cosine_at = compute_metric_ret(score_cosine_at, ids, ids_txt, direction='forward')
        val_log[f'cosine_at'] = cosine_at

        # compute itc_score
        for task in subtasks:
            if task == "tvas" or task == "tva":
                continue
            if task == 'tv':
                score_matrix_t_cond = torch.matmul(feat_t, feat_v.permute(1, 0))
            elif task == 'ta':
                score_matrix_t_cond = torch.matmul(feat_t, feat_a.permute(1, 0))
            store_dict[f'score_matrix_t_cond_{task[1:]}'] = score_matrix_t_cond
            log = compute_metric_ret(score_matrix_t_cond, ids, ids_txt, direction='forward')
            log = {k.replace('forward', 'video'): v for k, v in log.items()}
            if str_to_bool_mapper[model.config.ret_bidirection_evaluation]:
                log2 = compute_metric_ret(score_matrix_t_cond, ids, ids_txt, direction='backward')
                log2 = {k.replace('backward', 'txt'): v for k, v in log2.items()}
                log.update(log2)

            val_log[f'ret_itc_{task[1:]}'] = log

        # compute itm_score
        for task in subtasks:
            if task == "tvas" or task == "tva":
                continue
            if task != "tvas" and task != "tva":
                store_dict[f'condition_feats_{task[1:]}'] = torch.cat(store_dict[f'condition_feats_{task[1:]}'], dim=0)
            itm_rerank_num = model.config.itm_rerank_num
            score_matrix = refine_score_matrix(store_dict[f'condition_feats_{task[1:]}'], input_ids, attention_mask,
                                               store_dict[f'score_matrix_t_cond_{task[1:]}'], model, itm_rerank_num,
                                               direction='forward')
            log = compute_metric_ret(score_matrix, ids, ids_txt, direction='forward')
            log = {k.replace('forward', 'video'): v for k, v in log.items()}

            if str_to_bool_mapper[model.config.ret_bidirection_evaluation]:
                score_matrix = refine_score_matrix(store_dict[f'condition_feats_{task[1:]}'], input_ids, attention_mask,
                                                   store_dict[f'score_matrix_t_cond_{task[1:]}'], model, itm_rerank_num,
                                                   direction='backward')
                log2 = compute_metric_ret(score_matrix, ids, ids_txt, direction='backward')
                log2 = {k.replace('backward', 'txt'): v for k, v in log2.items()}
                log.update(log2)

            val_log[f'ret_itm_{task[1:]}'] = log

        if dist.get_rank() == 0:
            # ---------------------------------------------------
            # QRELS + TREC RUNS
            # ---------------------------------------------------
            trec_dir = os.path.join(final_config.run_cfg.output_dir, "trec_runs", dset_name)
            os.makedirs(trec_dir, exist_ok=True)

            # 1) QRELS: separate files for T2V and V2T
            qrels_t2v_path = os.path.join(str(trec_dir), "qrels_t2v.txt")
            qrels_v2t_path = os.path.join(str(trec_dir), "qrels_v2t.txt")
            save_qrels_t2v(ids, ids_txt, qrels_t2v_path)
            save_qrels_v2t(ids, ids_txt, qrels_v2t_path)

            # 2) Cosine TV / VT
            save_trec_run(
                score_matrix=score_cosine_tv,
                query_ids=ids_txt,  # T2V: text queries
                doc_ids=ids,  # video docs
                out_path=os.path.join(str(trec_dir), "cosine_T2V.txt"),
                run_name="cosine_T2V",
            )

            save_trec_run(
                score_matrix=score_cosine_vt,
                query_ids=ids,  # V2T: video queries
                doc_ids=ids_txt,  # text docs
                out_path=os.path.join(str(trec_dir), "cosine_V2T.txt"),
                run_name="cosine_V2T",
            )

            # 3) Triangle -area TV / VT (area is distance; use -area as score)
            area_T2V_scores = -area  # (num_txt, num_vid)
            area_V2T_scores = (-area).T  # (num_vid, num_txt)

            save_trec_run(
                score_matrix=area_T2V_scores,
                query_ids=ids_txt,
                doc_ids=ids,
                out_path=os.path.join(str(trec_dir), "area_T2V.txt"),
                run_name="area_T2V",
            )

            save_trec_run(
                score_matrix=area_V2T_scores,
                query_ids=ids,
                doc_ids=ids_txt,
                out_path=os.path.join(str(trec_dir), "area_V2T.txt"),
                run_name="area_V2T",
            )

            # 4) ITM-on-area TV / VT (reuse itm_area_*_mat)
            save_trec_run(
                score_matrix=itm_area_T2V_mat,
                query_ids=ids_txt,
                doc_ids=ids,
                out_path=os.path.join(str(trec_dir), "ITMarea_T2V.txt"),
                run_name="ITMarea_T2V",
            )

            save_trec_run(
                score_matrix=itm_area_V2T_mat,
                query_ids=ids,
                doc_ids=ids_txt,
                out_path=os.path.join(str(trec_dir), "ITMarea_V2T.txt"),
                run_name="ITMarea_V2T",
            )
            wandb.log(val_log)

    return val_log


def refine_score_matrix(condition_feats, input_ids, attention_mask, score_matrix_t_cond, model, itm_rerank_num,
                        direction='forward'):
    top_k = itm_rerank_num
    device = next(model.parameters()).device

    score_matrix_t_cond = score_matrix_t_cond.detach().cpu()
    condition_feats = condition_feats.detach().cpu().float()
    input_ids = input_ids.detach().cpu().long()
    attention_mask = attention_mask.detach().cpu().long()

    if direction == 'forward':
        idxs = score_matrix_t_cond.topk(top_k, dim=1).indices.cpu()
    else:
        idxs = score_matrix_t_cond.topk(top_k, dim=0).indices.cpu()
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    nums = score_matrix_t_cond.shape[0] // world_size + 1

    cur_length = condition_feats.shape[0]
    length_ls = all_gather_list_cpu(cur_length)  # this returns python list; ok
    start = 0
    start_ls, end_ls = [], []
    for l in range(len(length_ls)):
        start_ls.append(start)
        end_ls.append(start + length_ls[l])
        start += length_ls[l]

    cur_cond = condition_feats
    col_start, col_end = start_ls[rank], end_ls[rank]

    cur_score_matrix_new = score_matrix_t_cond[:, col_start:col_end].clone() * 0.0

    if rank == 0:
        pbar = tqdm(total=cur_length)
    else:
        pbar = NoOp()

    small_batch = 25
    if direction == 'forward':
        for local_j in range(cur_length):
            j = col_start + local_j
            # Find which texts i selected this video j in their topk
            # mask_i: (Ntxt,)
            mask_i = (idxs == j).any(dim=1)
            if not mask_i.any():
                pbar.update(1)
                continue

            cur_input_ids = input_ids[mask_i]  # CPU (m, L)
            cur_attention_mask = attention_mask[mask_i]  # CPU (m, L)

            # condition feats for this video j, expanded to m
            cond = cur_cond[local_j].unsqueeze(0).expand(cur_input_ids.shape[0], -1, -1)  # CPU (m, ...)

            cur_scores_chunks = []
            total_len = cond.shape[0]
            times = (total_len + small_batch - 1) // small_batch

            for k in range(times):
                slice_input_ids = cur_input_ids[k * small_batch:(k + 1) * small_batch].to(device, dtype=torch.long,
                                                                                          non_blocking=True)
                slice_attention_mask = cur_attention_mask[k * small_batch:(k + 1) * small_batch].to(device,
                                                                                                    dtype=torch.long,
                                                                                                    non_blocking=True)
                slice_condition_feats = cond[k * small_batch:(k + 1) * small_batch].to(device, non_blocking=True)

                slice_scores = model.compute_slice_scores(slice_condition_feats, slice_input_ids, slice_attention_mask)
                cur_scores_chunks.append(slice_scores.detach().cpu())  # back to CPU

            cur_scores = torch.cat(cur_scores_chunks, dim=0)  # CPU (m,)
            cur_score_matrix_new[mask_i, local_j] = cur_scores
            pbar.update(1)
    else:
        for local_j in range(cur_length):
            j = col_start + local_j
            text_topk = idxs[:, j]  # CPU (top_k,)
            # gather unique + keep within range
            text_topk = torch.unique(text_topk)

            cur_input_ids = input_ids[text_topk]  # CPU (m, L)
            cur_attention_mask = attention_mask[text_topk]  # CPU (m, L)

            cond = cur_cond[local_j].unsqueeze(0).expand(cur_input_ids.shape[0], -1, -1)  # CPU (m, ...)

            cur_scores_chunks = []
            total_len = cond.shape[0]
            times = (total_len + small_batch - 1) // small_batch

            for k in range(times):
                slice_input_ids = cur_input_ids[k * small_batch:(k + 1) * small_batch].to(device, dtype=torch.long,
                                                                                          non_blocking=True)
                slice_attention_mask = cur_attention_mask[k * small_batch:(k + 1) * small_batch].to(device,
                                                                                                    dtype=torch.long,
                                                                                                    non_blocking=True)
                slice_condition_feats = cond[k * small_batch:(k + 1) * small_batch].to(device, non_blocking=True)

                slice_scores = model.compute_slice_scores(slice_condition_feats, slice_input_ids, slice_attention_mask)
                cur_scores_chunks.append(slice_scores.detach().cpu())

            cur_scores = torch.cat(cur_scores_chunks, dim=0)  # CPU (m,)
            cur_score_matrix_new[text_topk, local_j] = cur_scores
            pbar.update(1)

    pbar.close()

    parts = all_gather_list_cpu(cur_score_matrix_new)  # list of CPU tensors [Ntxt, local_cols]
    score_matrix_full = torch.cat(parts, dim=1)  # CPU (Ntxt, Nvid)
    return score_matrix_full


def compute_metric_ret(score_matrix, ids, ids_txt, direction='forward'):
    assert score_matrix.shape == (len(ids_txt), len(ids))

    if direction == 'forward':  # text-to-vision retrieval
        indice_matrix = score_matrix.sort(dim=-1, descending=True)[1].tolist()
        rank = []
        for i in range(len(ids_txt)):
            gt_indice = ids.index(ids_txt[i])
            rank.append(indice_matrix[i].index(gt_indice))

        rank = torch.tensor(rank).to(score_matrix)

        vr_r1 = (rank < 1).sum().item() / len(ids_txt)
        vr_r5 = (rank < 5).sum().item() / len(ids_txt)
        vr_r10 = (rank < 10).sum().item() / len(ids_txt)
        v_medianR = torch.median(rank).item() + 1
        v_meanR = torch.mean(rank).item() + 1

        eval_log = {'forward_r1': round(vr_r1 * 100, 1),
                    'forward_recall': f'{round(vr_r1 * 100, 1)}/{round(vr_r5 * 100, 1)}/{round(vr_r10 * 100, 1)}',
                    'forward_ravg': round((vr_r1 + vr_r5 + vr_r10) / 3 * 100, 1)
                    }

    else:  # vision-to-text retrieval
        indice_matrix = score_matrix.sort(dim=0, descending=True)[1].permute(1, 0).tolist()
        rank = []
        for i in range(len(ids)):
            gt_indices = []
            for idx, id in enumerate(ids_txt):
                if id == ids[i]:
                    gt_indices.append(idx)

            rank.append(min([indice_matrix[i].index(idx) for idx in gt_indices]))

        rank = torch.tensor(rank).to(score_matrix)

        tr_r1 = (rank < 1).sum().item() / len(ids)
        tr_r5 = (rank < 5).sum().item() / len(ids)
        tr_r10 = (rank < 10).sum().item() / len(ids)
        t_medianR = torch.median(rank).item() + 1
        t_meanR = torch.mean(rank).item() + 1

        eval_log = {
            'backward_r1': round(tr_r1 * 100, 1),
            'backward_recall': f'{round(tr_r1 * 100, 1)}/{round(tr_r5 * 100, 1)}/{round(tr_r10 * 100, 1)}',
            'backward_ravg': round((tr_r1 + tr_r5 + tr_r10) / 3 * 100, 1)
        }
    return eval_log


def compute_metric_ret_area(score_matrix, ids, ids_txt, direction='forward'):
    assert score_matrix.shape == (len(ids_txt), len(ids))

    if direction == 'forward':  # text-to-vision retrieval
        indice_matrix = score_matrix.sort(dim=-1, descending=False)[1].tolist()
        rank = []
        for i in range(len(ids_txt)):
            gt_indice = ids.index(ids_txt[i])
            rank.append(indice_matrix[i].index(gt_indice))

        rank = torch.tensor(rank).to(score_matrix)

        vr_r1 = (rank < 1).sum().item() / len(ids_txt)
        vr_r5 = (rank < 5).sum().item() / len(ids_txt)
        vr_r10 = (rank < 10).sum().item() / len(ids_txt)
        v_medianR = torch.median(rank).item() + 1
        v_meanR = torch.mean(rank).item() + 1

        eval_log = {'forward_r1': round(vr_r1 * 100, 1),
                    'forward_recall': f'{round(vr_r1 * 100, 1)}/{round(vr_r5 * 100, 1)}/{round(vr_r10 * 100, 1)}',
                    'forward_ravg': round((vr_r1 + vr_r5 + vr_r10) / 3 * 100, 1)
                    }

    else:  # vision-to-text retrieval
        indice_matrix = score_matrix.sort(dim=0, descending=False)[1].permute(1, 0).tolist()
        rank = []
        for i in range(len(ids)):
            gt_indices = []
            for idx, id in enumerate(ids_txt):
                if id == ids[i]:
                    gt_indices.append(idx)

            rank.append(min([indice_matrix[i].index(idx) for idx in gt_indices]))

        rank = torch.tensor(rank).to(score_matrix)

        tr_r1 = (rank < 1).sum().item() / len(ids)
        tr_r5 = (rank < 5).sum().item() / len(ids)
        tr_r10 = (rank < 10).sum().item() / len(ids)
        t_medianR = torch.median(rank).item() + 1
        t_meanR = torch.mean(rank).item() + 1

        eval_log = {
            'backward_r1': round(tr_r1 * 100, 1),
            'backward_recall': f'{round(tr_r1 * 100, 1)}/{round(tr_r5 * 100, 1)}/{round(tr_r10 * 100, 1)}',
            'backward_ravg': round((tr_r1 + tr_r5 + tr_r10) / 3 * 100, 1)
        }
    return eval_log
