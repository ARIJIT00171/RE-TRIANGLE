import torch


class NoOp(object):
    """ useful for distributed training No-Ops """
    def __getattr__(self, name):
        return self.noop

    def noop(self, *args, **kwargs):
        return


def split(frame_name_lists, sample_num):
    if len(frame_name_lists) < sample_num:
        frame_name_lists += [frame_name_lists[-1]]*(sample_num - len(frame_name_lists))
    k, m = divmod(len(frame_name_lists), sample_num)
    return [frame_name_lists[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in list(range(sample_num))]


def compute_max_vision_sample_num_for_position_embeddings(data_cfg):
    data_cfg_train = data_cfg.train
    vision_sample_num_ls_train = []
    for d_cfg in data_cfg_train:
        vision_sample_num = d_cfg.get('vision_sample_num', 1)
        vision_sample_num_ls_train.append(vision_sample_num * data_cfg.concatenated_nums)

    data_cfg_val = data_cfg.val
    vision_sample_num_ls_val = []
    for d_cfg in data_cfg_val:
        vision_sample_num = d_cfg.get('vision_sample_num', 1)
        vision_sample_num_ls_val.append(vision_sample_num)

    max_vision_sample_num = max(vision_sample_num_ls_train) if vision_sample_num_ls_train else max(
        vision_sample_num_ls_val)

    assert max_vision_sample_num > 0
    return max_vision_sample_num


def compute_max_audio_sample_num_for_position_embeddings(data_cfg):
    data_cfg_train = data_cfg.train
    audio_sample_num_ls_train = []
    for d_cfg in data_cfg_train:
        audio_sample_num = d_cfg.get('audio_sample_num', 1)
        audio_sample_num_ls_train.append(audio_sample_num * data_cfg.concatenated_nums)

    data_cfg_val = data_cfg.val
    audio_sample_num_ls_val = []
    for d_cfg in data_cfg_val:
        audio_sample_num = d_cfg.get('audio_sample_num', 1)
        audio_sample_num_ls_val.append(audio_sample_num)

    max_audio_sample_num = max(audio_sample_num_ls_train) if audio_sample_num_ls_train else max(audio_sample_num_ls_val)

    assert max_audio_sample_num > 0
    return max_audio_sample_num


def area_computation(language, video, audio):
    language_expanded = language.unsqueeze(1)  # Shape: (n, 1, dim)

    # Compute the differences for all pairs (i-th language embedding with all j-th video/audio embeddings)
    u = language_expanded - video.unsqueeze(0)  # Shape: (n, n, dim)
    v = language_expanded - audio.unsqueeze(0)  # Shape: (n, n, dim)

    # Compute the norms for u and v
    u_norm = torch.sum(u ** 2, dim=2)  # Shape: (n, n)
    v_norm = torch.sum(v ** 2, dim=2)  # Shape: (n, n)

    # Compute the dot products for all pairs
    uv_dot = torch.sum(u * v, dim=2)  # Shape: (n, n)

    # Calculate the area for all pairs. I remove sqrt calculation
    area = ((u_norm * v_norm) - (uv_dot ** 2)) / 2  # torch.sqrt((u_norm * v_norm) - (uv_dot ** 2)) / 2  # Shape: (n, n)
    return area


def area_computation_chunked(language, video, audio, block=128):
    n, d = language.shape

    language2 = (language * language).sum(dim=1)      # (n,)
    video2 = (video * video).sum(dim=1)      # (n,)
    audio2 = (audio * audio).sum(dim=1)      # (n,)
    video_audio_diag = (video * audio).sum(dim=1)   # (n,)

    video_t = video.T
    audio_t = audio.T

    out = []
    for i in range(0, n, block):
        language_i = language[i:i+block]            # (b,d)
        language_i2 = language2[i:i+block]          # (b,)

        language_video = language_i @ video_t                 # (b,n)
        language_audio = language_i @ audio_t                 # (b,n)

        u_norm = language_i2[:, None] + video2[None, :] - 2 * language_video
        v_norm = language_i2[:, None] + audio2[None, :] - 2 * language_audio
        uv_dot = language_i2[:, None] - language_audio - language_video + video_audio_diag[None, :]

        area = (u_norm * v_norm - uv_dot.pow(2)) / 2  # (b,n)

        out.append(area)

        del language_i, language_i2, language_video, language_audio, u_norm, v_norm, uv_dot, area
        torch.cuda.empty_cache()  # optional; helps with fragmentation

    return torch.cat(out, dim=0)  # (n,n) on CPU


@torch.inference_mode()
def area_computation_cpu_chunked(language, video, audio, block=64):
    assert not language.is_cuda, "area_computation_cpu_chunked expects CPU tensor"
    assert not video.is_cuda, "area_computation_cpu_chunked expects CPU tensor"
    assert not audio.is_cuda, "area_computation_cpu_chunked expects CPU tensor"

    n, d = language.shape
    out = []
    for i in range(0, n, block):
        language_i = language[i:i+block]                 # (b,d)
        u = language_i[:, None, :] - video[None, :, :]   # (b,n,d)
        v = language_i[:, None, :] - audio[None, :, :]   # (b,n,d)

        u_norm = (u * u).sum(dim=2)             # (b,n)
        v_norm = (v * v).sum(dim=2)             # (b,n)
        uv_dot = (u * v).sum(dim=2)             # (b,n)

        area = (u_norm * v_norm - uv_dot.pow(2)) / 2
        out.append(area)  # stays CPU

        del u, v, u_norm, v_norm, uv_dot, area
    return torch.cat(out, dim=0)                # (n,n)


def area_computation_chunked_cos(language, video, audio, alpha=0.0, block=128, eps=1e-8):
    b, d = language.shape
    n = video.shape[0]
    assert video.shape == audio.shape and video.shape[1] == d

    # Precompute candidate-only terms
    video2 = (video * video).sum(dim=1)  # [N]
    audio2 = (audio * audio).sum(dim=1)  # [N]
    video_audio = (video * audio).sum(dim=1)  # [N]

    # Precompute language norms
    language2 = (language * language).sum(dim=1)  # [B]

    video_t = video.T  # [D, N]
    audio_t = audio.T  # [D, N]

    out = []
    for i in range(0, b, block):
        language_i = language[i:i + block]  # [b, D]
        language_i2 = language2[i:i + block]  # [b]

        language_video = language_i @ video_t  # [b, N]  (language·video)
        language_audio = language_i @ audio_t  # [b, N]  (language·audio)

        # u = language - video, v = language - audio
        u_norm = language_i2[:, None] + video2[None, :] - 2.0 * language_video  # ||u||^2  [b,N]
        v_norm = language_i2[:, None] + audio2[None, :] - 2.0 * language_audio  # ||v||^2  [b,N]
        uv_dot = language_i2[:, None] - language_audio - language_video + video_audio[None, :]  # u·v      [b,N]

        area = u_norm * v_norm - uv_dot * uv_dot

        if alpha != 0.0:
            denominator = torch.sqrt(u_norm + eps) * torch.sqrt(v_norm + eps)
            cos_uv = uv_dot / (denominator + eps)  # [b,N]
            area = area - alpha * cos_uv  # subtract cosine regularization

        out.append(area)

    return torch.cat(out, dim=0)  # [B,N]


@torch.inference_mode()
def area_computation_cpu_chunked_cos(language, video, audio, alpha=0.0, block=64, eps=1e-8):
    assert not language.is_cuda and not video.is_cuda and not audio.is_cuda

    b, d = language.shape
    n = video.shape[0]
    assert video.shape == audio.shape and video.shape[1] == d

    video2 = (video * video).sum(dim=1)  # [N]
    audio2 = (audio * audio).sum(dim=1)  # [N]
    video_audio = (video * audio).sum(dim=1)  # [N]
    language2 = (language * language).sum(dim=1)  # [B]

    video_t = video.T  # [D, N]
    audio_t = audio.T  # [D, N]

    out = []
    for i in range(0, b, block):
        language_i = language[i:i + block]  # [b, D]
        language_i2 = language2[i:i + block]  # [b]

        language_video = language_i @ video_t  # [b, N]
        language_audio = language_i @ audio_t  # [b, N]

        u_norm = language_i2[:, None] + video2[None, :] - 2.0 * language_video
        v_norm = language_i2[:, None] + audio2[None, :] - 2.0 * language_audio
        uv_dot = language_i2[:, None] - language_audio - language_video + video_audio[None, :]

        area = u_norm * v_norm - uv_dot * uv_dot

        if alpha != 0.0:
            denominator = torch.sqrt(u_norm + eps) * torch.sqrt(v_norm + eps)
            cos_uv = uv_dot / (denominator + eps)
            area = area - alpha * cos_uv

        out.append(area)

    return torch.cat(out, dim=0)  # [B,N] on CPU
