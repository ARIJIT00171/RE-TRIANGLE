import random
import torch
import h5py
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.transforms import *
from PIL import Image
import os
import glob
import cv2

from utils.logger import LOGGER
from utils.utils import split


def check_extension(id, folder):
    files_in_folder = glob.glob(os.path.join(folder, '*'))
    for file in files_in_folder:
        path = os.path.join(folder, id)
        if file.startswith(str(path)):
            if os.path.isfile(file):
                return file
    print(f"No File for Id: {id}")
    return ""


class VisionMapper(object):
    def __init__(self, d_cfg, final_config):

        self.vision = d_cfg.vision
        self.name = d_cfg.name
        self.training = d_cfg.training
        self.vision_format = d_cfg.vision_format

        # For feat extraction
        self.dense_extraction = getattr(d_cfg, 'dense_extraction', False)
        self.extract_fps = getattr(d_cfg, 'extract_fps', None)
        self.frame_fps = getattr(d_cfg, 'frame_fps', None)

        if self.vision_format.startswith('video'):
            self.sample_num = d_cfg.vision_sample_num

        self.resolution = final_config.model_cfg.vision_resolution

        if (final_config.model_cfg.vision_encoder_type.startswith('clip') or
                final_config.model_cfg.vision_encoder_type.startswith('evaclip')):
            self.mean = [0.48145466, 0.4578275, 0.40821073]
            self.std = [0.26862954, 0.26130258, 0.27577711]
            LOGGER.info(f'{self.name} Using clip mean and std.')
        else:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
            LOGGER.info(f'{self.name} Using imagenet mean and std.')

        self.vision_transforms = d_cfg.get('vision_transforms', 'none')
        if self.vision_transforms == 'none':
            if self.training:
                self.transforms = Compose([
                    Resize((self.resolution, self.resolution)),
                    Normalize(self.mean, self.std)])
            else:
                self.transforms = Compose([
                    Resize((self.resolution, self.resolution)),
                    Normalize(self.mean, self.std)])
        elif self.vision_transforms == 'crop_flip':
            if self.training:
                self.transforms = Compose([
                    RandomResizedCrop(self.resolution, [0.8, 1.0], [1.0, 1.0]),
                    RandomHorizontalFlip(),
                    Normalize(self.mean, self.std)])
            else:
                self.transforms = Compose([
                    Resize(self.resolution),
                    CenterCrop(self.resolution),
                    Normalize(self.mean, self.std)])

        else:
            raise NotImplementedError
        LOGGER.info(f'{self.name} transforms {self.vision_transforms}')

    def read(self, id_):
        if self.vision_format.startswith('video'):
            if self.vision_format == 'video_feats':
                if self.vision.endswith('hdf5'):
                    with h5py.File(self.vision, 'r') as f:
                        try:
                            feat = f[id_]['c3d_features'][:]
                        except Exception:
                            feat = f[id_][:]
                        feat = F.normalize(torch.from_numpy(feat), dim=1)
                else:
                    feat = np.load(os.path.join(self.vision, f'{id_}.npy'))
                    feat = F.normalize(torch.from_numpy(feat).float(), dim=1)

                num_pre_clips = self.num_pre_clips
                num_src_clips = feat.size(0)
                idxs = torch.arange(0, num_pre_clips + 1, 1.0) / num_pre_clips * num_src_clips
                idxs = idxs.round().long().clamp(max=num_src_clips - 1)

                meanfeats = []
                for i in range(num_pre_clips):
                    s, e = idxs[i], idxs[i + 1]
                    if s < e:
                        meanfeats.append(feat[s:e].mean(dim=0))
                    else:
                        meanfeats.append(feat[s])
                return torch.stack(meanfeats)

            else:
                vision_pixels = []
                sample_num = self.sample_num

                try:
                    if self.vision_format == 'video_rawvideo':
                        video_path = os.path.join(self.vision, str(id_))
                        # TODO verify ok for all data to subsitute check extension funciton
                        video_path = video_path + ".mp4"
                        video_path = check_extension(id_, self.vision)
                        cap = cv2.VideoCapture(video_path)

                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if total_frames < sample_num:
                            cap.release()
                            return None, None

                        # Get the FPS of the video
                        fps = cap.get(cv2.CAP_PROP_FPS)

                        # Determine the number of frames to sample
                        if self.dense_extraction:
                            sample_num = int(total_frames * extract_fps / fps)

                        # Create a list of frame IDs
                        frames_ids = list(range(total_frames))

                        # Split the frames for sampling
                        frames_splited = np.array_split(frames_ids, sample_num)
                        # Select the frames based on whether it's training or evaluation mode
                        if self.training:
                            sample_idx = [random.choice(i) for i in frames_splited]
                        else:
                            sample_idx = [i[(len(i) + 1) // 2 - 1] for i in frames_splited]

                        # Extract the selected frames
                        frames = []
                        for idx in sample_idx:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Set the video position to the selected frame
                            ret, frame = cap.read()  # Read the frame
                            if ret:
                                if frame.all():
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                                    frames.append(frame_rgb)

                        while len(frames) < sample_num and frames_ids != []:
                            idx = random.choice(frames_ids)
                            frames_ids.remove(idx)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)  # Set the video position to the selected frame
                            ret, frame = cap.read()  # Read the frame
                            if ret:
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                                frames.append(frame_rgb)

                        if len(frames) < sample_num:
                            cap.release()
                            return None, None
                        cap.release()  # Release the video capture object

                        # Convert the list of frames to a NumPy array
                        frames = np.array(frames)  # (8, 480, 848, 3) oppure (8, 1080, 1920, 3) etc.
                        # import pdb; pdb.set_trace()
                        # depth_pixels = extract_depth(frames, save_path=video_path)

                        # DEPTH
                        # TODO put something to say depth or not depth
                        if "MSRVTT_TODO" in video_path or "vast_TODO" in video_path:
                            output_dir_video = video_path[:-4]  # remove mp4
                            output_dir_video = output_dir_video.replace("videos", "depth")
                            try:
                                depth = np.load(f"{output_dir_video}_depth.npy")
                            except:
                                print("depth not found", id_)
                                return None, None
                            depth_pixels = torch.from_numpy(depth.transpose(0, 3, 1, 2) / 255.0)
                            depth_pixels = self.transforms(depth_pixels)
                        else:
                            depth_pixels = None
                        # Normalize and transpose the frames (N x H x W x C -> N x C x H x W)
                        vision_pixels = torch.from_numpy(frames.transpose(0, 3, 1, 2) / 255.0)
                        # Apply the necessary transforms
                        vision_pixels = self.transforms(vision_pixels)

                        return vision_pixels, depth_pixels
                    elif self.vision_format == 'video_frame':
                        frame_path = os.path.join(self.vision, str(id_))
                        frames = os.listdir(frame_path)
                        frames.sort()  # ['img_0001.jpg','img_0002.jpg',...]
                        sample_num = self.sample_num
                        if self.dense_extraction:  # for feature extraction
                            sample_num = int(len(frames) * self.extract_fps / self.frame_fps)
                        frames_splited = split(frames, sample_num)
                        if self.training:
                            sample_idx = [random.choice(i) for i in frames_splited]
                        else:
                            sample_idx = [i[(len(i) + 1) // 2 - 1] for i in frames_splited]
                        for i in range(sample_num):
                            frame = Image.open(os.path.join(frame_path, sample_idx[i]))
                            frame = np.array(frame).transpose(2, 0, 1) / 255.0
                            vision_pixels.append(frame)

                        vision_pixels = torch.from_numpy(np.stack(vision_pixels, axis=0))  # nX3xHxW
                        vision_pixels = self.transforms(vision_pixels)
                        return vision_pixels

                except Exception as e:
                    print(e)
                    print(id_)
                    return None, None

        elif self.vision_format.startswith('image'):
            try:
                if self.vision_format == 'image_rawimage':
                    img_path = os.path.join(self.vision, id_)
                    if not os.path.exists(img_path):
                        img_path += '.jpg'
                    if not os.path.exists(img_path):
                        img_path = img_path.replace('.jpg', '.JPEG')
                    if not os.path.exists(img_path):
                        print('not have im', id_)
                        assert self.name == 'llava_v1_5_mix665k'
                        return torch.zeros(1, 3, self.resolution, self.resolution)
                else:
                    img_path = ''
                img = Image.open(img_path).convert('RGB')
                img = np.array(img).transpose(2, 0, 1) / 255.0
                img = torch.from_numpy(img)
                img = self.transforms(img)
                img = img.unsqueeze(0)
                return img

            except Exception as e:
                print(e)
                return None

        else:
            raise NotImplementedError()
