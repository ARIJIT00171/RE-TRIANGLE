import os
import json
import random
import torch
from toolz.sandbox import unzip
from torch.utils.data import Dataset

from dataset.vision_mapper import VisionMapper
from dataset.audio_mapper import AudioMapper
from utils.logger import LOGGER


def check_files_start_with(directory, start_string):
    files = os.listdir(directory)

    for file in files:
        if file.startswith(start_string):
            return file
    return None


class AnnoIndexedDataset(Dataset):
    def __init__(self, d_cfg, final_config, device):
        self.vision_mapper = VisionMapper(d_cfg, final_config) if 'vision' in d_cfg else None
        self.audio_mapper = AudioMapper(d_cfg, final_config) if 'audio' in d_cfg else None
        self.annos = json.load(open(d_cfg['txt']))
        self.annos_new = []
        self.name = d_cfg['name']

        # Select only id that have audio and video:
        for key in self.annos:
            if self.name == "youcook_ret":
                path = os.path.join(d_cfg['audio'], f"{key['video_id']}.mp3")
                if os.path.exists(path):
                    self.annos_new.append(key)
            if self.name == "didemo_ret":
                path = os.path.join(d_cfg['audio'], f"{key['video_id']}.mp3")
                if os.path.exists(path):
                    self.annos_new.append(key)
            if self.name == "vatex_ret":
                path = os.path.join(d_cfg['audio'], f"{key['video_id']}.mp3")
                print(path, "AAA")
                if os.path.exists(path):
                    self.annos_new.append(key)
            if self.name == "msrvtt_ret":
                path = os.path.join(d_cfg['audio'], f"{key['video_id']}.mp3")
                if os.path.exists(path):
                    self.annos_new.append(key)
            if self.name == "activitynet_ret":
                path = os.path.join(d_cfg['audio'], f"{key['video_id']}.mp3")
                if os.path.exists(path):
                    if "desc" in key:
                        self.annos_new.append(key)
            if self.name == "audiocaps_ret":
                key['video_id'] = key["video_id"].split(".")[0]
                path = os.path.join(d_cfg['audio'], f"{key['video_id']}.mp3")
                if os.path.exists(path):
                    path = os.path.join(d_cfg['vision'], f"{key['video_id']}.mp4")
                    if os.path.exists(path):
                        self.annos_new.append(key)
            if self.name == "finetune_area":
                path = os.path.join(d_cfg['audio'], f"{key['clip_id']}.mp3")
                if os.path.exists(path):
                    path = os.path.join(d_cfg['vision'], f"{key['clip_id']}.mp4")
                    if os.path.exists(path):
                        self.annos_new.append(key)

        self.annos = self.annos_new
        LOGGER.info(f"Final anno count: {len(self.annos)} labels")
        self.idx = list(range(len(self.annos)))
        self.dataset_name = d_cfg['name']
        self.training = d_cfg.training

        self.worker_init_fn = None
        self.use_sampler = True
        self.collate_fn = annoindexedcollate
        self.device = device

        self.annfile = getattr(d_cfg, 'annfile', None)
        self.make_submission = getattr(d_cfg, 'make_submission', False)
        self.multi_evaluation = getattr(d_cfg, 'multi_evaluation', False)

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, i):
        anno = self.annos[i]
        id_ = ''
        for key in ['clip_id', 'video_id', 'image_id', 'image', 'id']:
            if key in anno:
                id_ = anno.get(key)
                break

        raw_captions = None
        raw_subtitles = None
        vision_pixels = None
        depth_pixels = None
        audio_spectrotriangles = None
        vision_cap = None
        audio_cap = None

        if 'desc' in anno:
            raw_captions = anno.get('desc')
        elif 'caption' in anno:
            raw_captions = anno.get('caption')
        elif 'vast_cap' in anno:
            raw_captions = anno.get('vast_cap')
        raw_captions = raw_captions[0] if isinstance(raw_captions, list) else raw_captions
        num_samples = len(raw_captions) if isinstance(raw_captions, list) else 1

        id_txt = [id_] * num_samples

        if 'subtitle' in anno:
            raw_subtitles = anno.get('subtitle')
        if 'vision_cap' in anno:
            if isinstance(anno.get('vision_cap'), list):  # vqav2
                vision_cap = random.choice(anno.get('vision_cap'))
            else:
                vision_cap = anno.get('vision_cap')
        if 'audio_cap' in anno:
            if isinstance(anno.get('audio_cap'), list):  # vqav2
                audio_cap = random.choice(anno.get('audio_cap'))
            else:
                audio_cap = anno.get('audio_cap')

        if self.vision_mapper:
            if self.vision_mapper.vision_format == 'video_feats':
                vision_feats = self.vision_mapper.read(id_)
            else:
                vision_pixels, depth_pixels = self.vision_mapper.read(id_)
                if vision_pixels is None:  # wrong img/video, resample when training and raise error when testing
                    if self.training:
                        resample_idx = random.choice(self.idx)
                        LOGGER.info(
                            f'current idx {id_} from {self.dataset_name} returns wrong image/video, use {resample_idx} '
                            f'instead.')
                        return self.__getitem__(resample_idx)
                    else:
                        resample_idx = random.choice(self.idx)
                        LOGGER.info(
                            f'current idx {id_} from {self.dataset_name} returns wrong '
                            f'image/video,!!!!!!!!!!!!!!!!!!!!!!!! use {resample_idx} instead.')
                        return self.__getitem__(resample_idx)
                        # raise ValueError

        if self.audio_mapper:
            audio_spectrotriangles = self.audio_mapper.read(id_)
            if audio_spectrotriangles is None:  # wrong audio, resample when training and raise error when testing
                if self.training:
                    resample_idx = random.choice(self.idx)
                    LOGGER.info(
                        f'current idx {id_} from {self.dataset_name} returns wrong audio, use {resample_idx} instead.')
                    return self.__getitem__(resample_idx)
                else:
                    raise ValueError
                    # print(raw_captions)
        return (id_, raw_captions, vision_pixels, id_txt, audio_spectrotriangles, raw_subtitles, vision_cap,
                audio_cap, depth_pixels)


def annoindexedcollate(inputs):
    batch = {}
    all_data = list(map(list, unzip(inputs)))
    keys = ['ids',
            'raw_captions',
            'vision_pixels',
            'ids_txt',
            'audio_spectrotriangles',
            'raw_subtitles',
            'vision_captions',
            'audio_captions',
            "depth_pixels"]

    for key, data in zip(keys, all_data):
        if data[0] is None:
            continue
        elif isinstance(data[0], torch.Tensor):
            shapes = [tuple(t.shape) for t in data]
            assert len(set(shapes)) == 1, f"{key} has varying shapes: {set(shapes)}"
            batch[key] = torch.stack(data, dim=0).float()
        else:
            batch[key] = data
    for k, v in batch.items():
        if torch.is_tensor(v) and v.is_cuda:
            raise RuntimeError(f"{k} is CUDA in collate (should be CPU).")
    return batch
