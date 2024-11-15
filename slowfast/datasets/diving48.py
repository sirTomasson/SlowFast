import json
import av
import os

import numpy as np
import torch
from pytorchvideo.transforms import ShortSideScale, Div255, UniformTemporalSubsample
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.samplers import UniformClipSampler
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import CenterCropVideo
from torchvision.transforms.v2 import Lambda, ToTensor


def pad_video(video, size):
    padding_size = size - len(video)
    return np.concatenate((video, np.zeros((padding_size, *video.shape[1:]))))

def collate_fn(batch):
    max_frames = max(video.shape[0] for video in batch)
    return [pad_video(video, max_frames) for video in batch]

class Diving48Dataset(Dataset):

    def __init__(self, videos_path, annotations_path, transform_fn=None, target_fps=None):
        super().__init__()
        self.videos_path = videos_path
        self.annotations_path = annotations_path
        self.target_fps = target_fps
        self.transform_fn = transform_fn
        self._init_dataset()

    def _init_dataset(self):
        with open(self.annotations_path, 'rb') as f:
            self.data = json.loads(f.read())


    def _read_frames(self, video_id):
        video_path = os.path.join(self.videos_path, f'{video_id}.mp4')
        container = av.open(video_path)

        video_stream = container.streams.video[0]

        fps = video_stream.average_rate

        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format='rgb24'))

        frames = np.stack(frames)
        if self.transform_fn:
            frames = self.transform_fn(frames)

        return frames

    def __getitem__(self, index):
        video_id = self.data[index]['vid_name']
        label = self.data[index]['label']
        return self._read_frames(video_id), label


    def __len__(self):
        return self.num_videos

    @property
    def num_videos(self):
        return len(self.data)


import unittest

VIDEOS_PATH = './data/Diving48'
ANNOTATIONS_PATH = './data/Diving48/Diving48_V2_test.json'

class Diving48DatasetTest(unittest.TestCase):

    def test_init(self):
        diving48 = Diving48Dataset(VIDEOS_PATH, ANNOTATIONS_PATH)
        self.assertEqual(diving48.videos_path, VIDEOS_PATH)
        self.assertEqual(diving48.annotations_path, ANNOTATIONS_PATH)
        self.assertEqual(diving48.target_fps, None)

    def test_get_item(self):
        diving48 = Diving48Dataset(VIDEOS_PATH, ANNOTATIONS_PATH)
        diving48_iter = iter(diving48)
        (video, label) = next(diving48_iter)
        self.assertIsNotNone(label)
        self.assertGreater(len(video), 0)


    def test_transforms(self):
        transform = Compose([
            Lambda(lambda x: torch.tensor(x).permute(3, 0, 1, 2)),
            Div255(), # Div255 assumes C x T x W x H
            ShortSideScale(size=256),
            CenterCropVideo(crop_size=(224, 224))
        ])
        diving48 = Diving48Dataset(VIDEOS_PATH, ANNOTATIONS_PATH, transform_fn=transform)
        x, _ = next(iter(diving48))
        # assert dims equal to 3 x T x 224 x 224
        self.assertEqual(x.shape[0], 3)
        self.assertEqual(x.shape[2:], (224, 224))

    def test_dataloader(self):
        transform = Compose([
            Lambda(lambda x: torch.tensor(x).permute(3, 0, 1, 2)),
            Div255(), # Div255 assumes C x T x W x H
            ShortSideScale(size=256),
            CenterCropVideo(crop_size=(224, 224)),
            UniformTemporalSubsample(128) # ensures each sample has the same number of frames, will under sample if T < 128, and over sample if T > 128
        ])
        diving48 = Diving48Dataset(VIDEOS_PATH, ANNOTATIONS_PATH, transform_fn=transform)
        diving48_dataloader = DataLoader(diving48, batch_size=4, shuffle=True)
        diving48_iter = iter(diving48_dataloader)
        x, y = next(diving48_iter)
        self.assertEqual(y.size()[0], 4)
        # assert shape B x C x T x W x H
        self.assertEqual(x.size(), (4, 3, 128, 224, 224))

    def test_pad_video(self):
        x = np.random.rand(10, 24, 24, 3)
        x_padded = pad_video(x, 12)
        self.assertEqual(x_padded.shape, (12, 24, 24, 3))
        zero = np.sum(x_padded[10:], axis=(0, 1, 2, 3))
        self.assertEqual(zero, 0)
        not_zero = np.sum(x_padded[:10], axis=(0, 1, 2, 3))
        self.assertNotEqual(not_zero, 0)

    def test_collate_fn(self):
        xb = [
            np.random.rand(12, 24, 24, 3),
            np.random.rand(64, 24, 24, 3),
            np.random.rand(8, 24, 24, 3),
            np.random.rand(128, 24, 24, 3)
        ]
        xb_padded = collate_fn(xb)
        xb_padded = np.stack(xb_padded)
        self.assertEqual(xb_padded.shape, (4, 128, 24, 24, 3))


if __name__ == '__main__':
    unittest.main()