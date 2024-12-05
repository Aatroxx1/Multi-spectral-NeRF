from typing import Literal, Optional, Tuple, Union

import nerfacc
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn

BackgroundColor = Union[Literal["random", "last_sample", "black", "white"], Float[Tensor, "3"], Float[Tensor, "*bs 3"]]
BACKGROUND_COLOR_OVERRIDE: Optional[Float[Tensor, "3"]] = None


class MSRenderer(nn.Module):
    def __init__(self, num_ms: int, background_color: BackgroundColor = "random", semantic: bool = False) -> None:
        super().__init__()
        self.num_ms = num_ms
        self.background_color: BackgroundColor = background_color
        self.semantic = semantic

    def combine_ms(
            self,
            ms: Float[Tensor, "*bs num_samples num_ms"],
            weights: Float[Tensor, "*bs num_samples 1"],
            background_color: BackgroundColor = "random",
            ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
            num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs num_ms"]:
        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            if background_color == "last_sample":
                raise NotImplementedError("Background color 'last_sample' not implemented for packed samples.")
            comp_ms = nerfacc.accumulate_along_rays(
                weights[..., 0], values=ms, ray_indices=ray_indices, n_rays=num_rays
            )
            accumulated_weight = nerfacc.accumulate_along_rays(
                weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
            )
        else:
            comp_ms = torch.sum(weights * ms, dim=-2)
            accumulated_weight = torch.sum(weights, dim=-2)
        if BACKGROUND_COLOR_OVERRIDE is not None:
            background_color = BACKGROUND_COLOR_OVERRIDE
        if background_color == "random":
            return comp_ms
        elif background_color == "last_sample":
            # Note, this is only supported for non-packed samples.
            background_color = ms[..., -1, :]
        background_color = self.get_background_color(background_color, shape=comp_ms.shape, device=comp_ms.device)

        assert isinstance(background_color, torch.Tensor)
        comp_ms = comp_ms + background_color * (1.0 - accumulated_weight)
        return comp_ms

    def get_background_color(
            self, background_color: BackgroundColor, shape: Tuple[int, ...], device: torch.device
    ) -> Union[Float[Tensor, "3"], Float[Tensor, "*bs 3"]]:
        assert background_color not in {"last_sample", "random"}
        # assert shape[-1] == 3, "Background color must be RGB."
        if BACKGROUND_COLOR_OVERRIDE is not None:
            background_color = BACKGROUND_COLOR_OVERRIDE
        if isinstance(background_color, str) and background_color == "ms_black":
            background_color = torch.tensor([0.0, ] * self.num_ms, device=device)
        assert isinstance(background_color, Tensor)

        return background_color.expand(shape).to(device)

    def blend_background(
            self,
            image: Tensor,
            background_color: Optional[BackgroundColor] = None,
    ) -> Float[Tensor, "*bs num_ms"]:
        if not self.semantic:
            return image
        ms, opacity = image[..., :-1], image[..., -1:]
        assert self.background_color == "random"
        if background_color is None:
            background_color = self.background_color
            if background_color in {"last_sample", "random"}:
                background_color = "ms_black"
        background_color = self.get_background_color(background_color, shape=ms.shape, device=ms.device)
        assert isinstance(background_color, torch.Tensor)
        return ms * opacity + background_color.to(ms.device) * (1 - opacity)

    def blend_background_for_loss_computation(
            self,
            pred_image: Tensor,
            pred_accumulation: Tensor,
            gt_image: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        background_color = self.background_color
        if background_color == "last_sample":
            background_color = "black"  # No background blending for GT
        elif background_color == "random":
            background_color = torch.rand_like(pred_image)
            pred_image = pred_image + background_color * (1.0 - pred_accumulation)
        gt_image = self.blend_background(gt_image, background_color=background_color)
        return pred_image, gt_image

    def forward(
            self,
            ms: Float[Tensor, "*bs num_samples num_ms"],
            weights: Float[Tensor, "*bs num_samples 1"],
            ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
            num_rays: Optional[int] = None,
            background_color: Optional[BackgroundColor] = None,
    ) -> Float[Tensor, "*bs num_ms"]:
        if background_color is None:
            background_color = self.background_color

        if not self.training:
            ms = torch.nan_to_num(ms)
        ms = self.combine_ms(
            ms, weights, background_color=background_color, ray_indices=ray_indices, num_rays=num_rays
        )
        if not self.training:
            torch.clamp_(ms, min=0.0, max=1.0)
        return ms


class MSSRRenderer(nn.Module):
    def __init__(self, num_ms: int, background_color: BackgroundColor = "random", semantic=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_ms = num_ms
        self.background_color: BackgroundColor = background_color
        self.semantic = semantic

    def combine_ms(
            self,
            ms: Float[Tensor, "*bs num_samples num_ms"],
            weights: Float[Tensor, "*bs num_samples 1"],
            background_color: BackgroundColor = "random",
            ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
            num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs num_ms"]:
        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            if background_color == "last_sample":
                raise NotImplementedError("Background color 'last_sample' not implemented for packed samples.")
            comp_ms = nerfacc.accumulate_along_rays(
                weights[..., 0], values=ms, ray_indices=ray_indices, n_rays=num_rays
            )
            accumulated_weight = nerfacc.accumulate_along_rays(
                weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
            )
        else:
            comp_ms = torch.sum(weights * ms, dim=-2)
            accumulated_weight = torch.sum(weights, dim=-2)
        if BACKGROUND_COLOR_OVERRIDE is not None:
            background_color = BACKGROUND_COLOR_OVERRIDE
        if background_color == "random":
            return comp_ms
        elif background_color == "last_sample":
            # Note, this is only supported for non-packed samples.
            background_color = ms[..., -1, :]
        background_color = self.get_background_color(background_color, shape=comp_ms.shape, device=comp_ms.device)

        assert isinstance(background_color, torch.Tensor)
        comp_ms = comp_ms + background_color * (1.0 - accumulated_weight)
        return comp_ms

    def get_background_color(
            self, background_color: BackgroundColor, shape: Tuple[int, ...], device: torch.device
    ) -> Union[Float[Tensor, "3"], Float[Tensor, "*bs 3"]]:
        assert background_color not in {"last_sample", "random"}
        # assert shape[-1] == 3, "Background color must be RGB."
        if BACKGROUND_COLOR_OVERRIDE is not None:
            background_color = BACKGROUND_COLOR_OVERRIDE
        if isinstance(background_color, str) and background_color == "ms_black":
            background_color = torch.tensor([0.0, ], device=device)
        assert isinstance(background_color, Tensor)

        return background_color.expand(shape).to(device)

    def blend_background(
            self,
            image: Tensor,
            background_color: Optional[BackgroundColor] = None,
    ) -> Float[Tensor, "*bs num_ms"]:
        if not self.semantic:
            return image
        opacity = (image > 0).to(image)
        assert self.background_color == "random"
        if background_color is None:
            background_color = self.background_color
            if background_color in {"last_sample", "random"}:
                background_color = "ms_black"
        background_color = self.get_background_color(background_color, shape=image.shape, device=image.device)
        assert isinstance(background_color, torch.Tensor)
        return image * opacity + background_color.to(image.device) * (1 - opacity)

    def blend_background_for_loss_computation(
            self,
            pred_image: Tensor,
            pred_accumulation: Tensor,
            ms_index: Tensor,
            gt_image: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        pred_image = self.extract_band(pred_image, ms_index)
        background_color = self.background_color
        if background_color == "last_sample":
            background_color = "black"  # No background blending for GT
        elif background_color == "random":
            background_color = torch.rand_like(pred_image)
            pred_image = pred_image + background_color * (1.0 - pred_accumulation)
        gt_image = self.blend_background(gt_image, background_color=background_color)
        return pred_image, gt_image

    def extract_band(self, pred_image: Tensor, ms_index: Tensor) -> Tensor:
        assert pred_image.shape[:-1] == ms_index.shape[:-1]
        return torch.gather(pred_image, -1, ms_index)

    def forward(
            self,
            ms: Float[Tensor, "*bs num_samples num_ms"],
            weights: Float[Tensor, "*bs num_samples 1"],
            ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
            num_rays: Optional[int] = None,
            background_color: Optional[BackgroundColor] = None,
    ) -> Float[Tensor, "*bs num_ms"]:
        if background_color is None:
            background_color = self.background_color

        if not self.training:
            ms = torch.nan_to_num(ms)
        ms = self.combine_ms(
            ms, weights, background_color=background_color, ray_indices=ray_indices, num_rays=num_rays
        )
        if not self.training:
            torch.clamp_(ms, min=0.0, max=1.0)
        return ms


class SemanticRenderer(nn.Module):

    @classmethod
    def forward(
            cls,
            semantics: Float[Tensor, "*bs num_samples num_classes"],
            weights: Float[Tensor, "*bs num_samples 1"],
            ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
            num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs num_classes"]:
        """Calculate semantics along the ray."""
        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            return nerfacc.accumulate_along_rays(
                weights[..., 0], values=semantics, ray_indices=ray_indices, n_rays=num_rays
            )
        else:
            return torch.sum(weights * semantics, dim=-2)
