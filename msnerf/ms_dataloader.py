from nerfstudio.data.utils.dataloaders import CacheDataloader, EvalDataloader, FixedIndicesEvalDataloader, RandIndicesEvalDataloader
import concurrent.futures
import multiprocessing
import random
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sized, Tuple, Union

import torch
from rich.progress import track
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.nerfstudio_collate import nerfstudio_collate
from nerfstudio.utils.misc import get_dict_to_torch
from nerfstudio.utils.rich_utils import CONSOLE


class MSEvalDataloader(EvalDataloader):
    def __init__(
            self,
            input_dataset: InputDataset,
            num_ms: int,
            device: Union[torch.device, str] = "cpu",
            **kwargs,
    ):
        self.num_ms = num_ms
        super().__init__(input_dataset, device=device, **kwargs)


    def get_camera(self, image_idx: int = 0) -> Tuple[Cameras, Dict]:
        camera = self.cameras[image_idx : image_idx + 1]
        batch = self.input_dataset[image_idx]
        batch = get_dict_to_torch(batch, device=self.device, exclude=["image"])
        assert isinstance(batch, dict)
        return camera, batch