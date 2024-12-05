from __future__ import annotations
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from msnerf.ms_dataset import MSDataset, MSSRDataset
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager, ParallelDataManagerConfig
from nerfstudio.data.utils.dataloaders import CacheDataloader, FixedIndicesEvalDataloader, RandIndicesEvalDataloader
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.utils.rich_utils import CONSOLE

@dataclass
class MSParallelDataManagerConfig(ParallelDataManagerConfig):
    _target: Type = field(default_factory=lambda: MSParallelDataManager)


class MSParallelDataManager(ParallelDataManager[MSDataset]):
    pass


@dataclass
class MSSRParallelDataManagerConfig(ParallelDataManagerConfig):
    _target: Type = field(default_factory=lambda: MSSRParallelDataManager)


class MSSRParallelDataManager(ParallelDataManager[MSSRDataset]):
    # def setup_eval(self):
    #     """Sets up the data loader for evaluation."""
    #     assert self.eval_dataset is not None
    #     CONSOLE.print("Setting up evaluation dataset...")
    #     self.eval_image_dataloader = MSCacheDataloader(
    #         self.eval_dataset,
    #         num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
    #         num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
    #         device=self.device,
    #         num_workers=self.world_size * 4,
    #         pin_memory=True,
    #         collate_fn=self.config.collate_fn,
    #         exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
    #     )
    #     self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
    #     self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset,
    #                                                       self.config.eval_num_rays_per_batch)  # type: ignore
    #     self.eval_ray_generator = RayGenerator(self.eval_dataset.cameras.to(self.device))
    #     # for loading full images
    #     self.fixed_indices_eval_dataloader = MSFixedIndicesEvalDataloader(
    #         input_dataset=self.eval_dataset,
    #         device=self.device,
    #         num_workers=self.world_size * 4,
    #     )
    #     self.eval_dataloader = MSRandIndicesEvalDataloader(
    #         input_dataset=self.eval_dataset,
    #         device=self.device,
    #         num_workers=self.world_size * 4,
    #     )
    pass