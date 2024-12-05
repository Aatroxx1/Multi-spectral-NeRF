from __future__ import annotations

from typing import Literal, Dict

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from jaxtyping import Float, UInt8
from torch import Tensor

from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_image_mask_tensor_from_path


class MSDataset(InputDataset):
    """Dataset that returns images."""

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["multi-spectral"]

    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
            image_type: the type of images returned
        """
        if image_type == "float32":
            image = self.get_image_float32(image_idx)
        elif image_type == "uint8":
            image = self.get_image_uint8(image_idx)
        else:
            raise NotImplementedError(f"image_type (={image_type}) getter was not implemented, use uint8 or float32")

        data = {"image_idx": image_idx, "image": image}
        if self._dataparser_outputs.mask_filenames is not None:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert (
                    data["mask"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
        if self.mask_color:
            data["image"] = torch.where(
                data["mask"] == 1.0, data["image"], torch.ones_like(data["image"]) * torch.tensor(self.mask_color)
            )
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        stem = image_filename.stem
        suffix = image_filename.suffix
        images = []
        for i in range(self.metadata["channel_num"]):
            par = image_filename.parent
            par = par / ("part" + str(i + 1))
            # pil_image = Image.open(par / (stem + "_part" + str(i + 1) + suffix))
            pil_image = Image.open(par / (stem + suffix))
            images.append(np.array(pil_image, dtype="uint8"))
        image = np.stack(images, axis=-1)  # shape is (h, w) or (h, w, 3 or 4)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] == self.metadata["channel_num"], "multi-spectral channel number is wrong"
        return image

    def get_image_float32(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32") / 255.0)
        return image

    def get_image_uint8(self, image_idx: int) -> UInt8[Tensor, "image_height image_width num_channels"]:
        image = torch.from_numpy(self.get_numpy_image(image_idx))
        return image


class MSSRDataset(InputDataset):
    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["multi-spectral"]

    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        if image_type == "float32":
            image = self.get_image_float32(image_idx)
        elif image_type == "uint8":
            image = self.get_image_uint8(image_idx)
        else:
            raise NotImplementedError(f"image_type (={image_type}) getter was not implemented, use uint8 or float32")

        data = {"image_idx": image_idx, "image": image}
        if self._dataparser_outputs.mask_filenames is not None:
            mask_filepath = self._dataparser_outputs.mask_filenames[image_idx]
            data["mask"] = get_image_mask_tensor_from_path(filepath=mask_filepath, scale_factor=self.scale_factor)
            assert (
                    data["mask"].shape[:2] == data["image"].shape[:2]
            ), f"Mask and image have different shapes. Got {data['mask'].shape[:2]} and {data['image'].shape[:2]}"
        if self.mask_color:
            data["image"] = torch.where(
                data["mask"] == 1.0, data["image"], torch.ones_like(data["image"]) * torch.tensor(self.mask_color)
            )
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        pil_image = Image.open(image_filename)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        image = image[:, :, None]
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] == 1
        return image

    def get_image_float32(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32") / 255.0)
        return image

    def get_image_uint8(self, image_idx: int) -> UInt8[Tensor, "image_height image_width num_channels"]:
        image = torch.from_numpy(self.get_numpy_image(image_idx))
        return image
