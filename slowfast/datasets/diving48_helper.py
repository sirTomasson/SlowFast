import torch
import slowfast.utils.logging as logging

from pytorchvideo.transforms import Div255, ShortSideScale
from torchvision.transforms._transforms_video import CenterCropVideo

from torchvision.transforms.v2 import Compose, Lambda, UniformTemporalSubsample

from diving48 import Diving48Dataset
from .build import DATASET_REGISTRY


logger = logging.get_logger(__name__)

@DATASET_REGISTRY.register()
class Diving48(Diving48Dataset):

    def __init__(self, cfg, _):
        print('1. hI m0m')
        logger.info('2. hI m0m')
        transforms = Compose([
            Lambda(lambda x: torch.tensor(x).permute(3, 0, 1, 2)),
            Div255(), # Div255 assumes C x T x W x H
            ShortSideScale(size=256),
            CenterCropVideo(crop_size=(224, 224)),
            UniformTemporalSubsample(4) # ensures each sample has the same number of frames, will under sample if T < 128, and over sample if T > 128
        ])
        super().__init__(cfg.DATA.VIDEOS_PATH, cfg.DATA.ANNOTATIONS_PATH, transform_fn=transforms)
        self.data = self.data[:10] # shrink dataset for testing purposes
