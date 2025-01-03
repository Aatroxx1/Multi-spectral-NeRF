from dataclasses import field, dataclass
from typing import Dict, List, Literal, Tuple, Type

import numpy as np
import torch
from torch.nn import MSELoss
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio, SpectralAngleMapper

from msnerf.ms_field import MSNerfField
from msnerf.ms_renderer import MSRenderer, MSSRRenderer
from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.rays import RaySamples, RayBundle
from nerfstudio.engine.callbacks import TrainingCallbackAttributes, TrainingCallback, TrainingCallbackLocation
from nerfstudio.field_components.field_heads import (
    FieldHeadNames, )
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.model_components.losses import distortion_loss, interlevel_loss, orientation_loss, pred_normal_loss
from nerfstudio.model_components.ray_samplers import UniformSampler, ProposalNetworkSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps


@dataclass
class MSNerfModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: MSNerfModel)
    num_multispectral: int = 25
    appearance_embed_dim: int = 32
    num_levels: int = 16
    base_res: int = 16
    max_res: int = 2048
    log2_hashmap_size: int = 19
    features_per_level: int = 2
    num_layers: int = 2
    hidden_dim: int = 64
    geo_feat_dim: int = 15
    num_layers_ms: int = 3
    hidden_dim_ms: int = 64
    implementation: Literal["tcnn", "torch"] = "tcnn"
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    use_proposal_weight_anneal: bool = True
    num_proposal_iterations: int = 2
    use_same_proposal_network: bool = False
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    average_init_density: float = 1.0

    proposal_update_every: int = 5
    proposal_warmup: int = 5000
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    num_nerf_samples_per_ray: int = 48
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    use_single_jitter: bool = True
    near_plane: float = 0.05
    far_plane: float = 1000.0
    predict_normals: bool = False
    background_color: Literal["random", "last_sample", "black", "white", "ms_white", "ms_black"] = "random"
    proposal_weights_anneal_max_num_iters: int = 1000
    proposal_weights_anneal_slope: float = 10.0
    disable_scene_contraction: bool = False

    interlevel_loss_mult: float = 1.0
    distortion_loss_mult: float = 0.002
    orientation_loss_mult: float = 0.0001
    pred_normal_loss_mult: float = 0.001

    senmantic: bool = False


class MSNerfModel(Model):
    config: MSNerfModelConfig

    def populate_modules(self):
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.field = MSNerfField(
            aabb=self.scene_box.aabb,
            num_images=self.num_train_data,
            num_multispectral=self.config.num_multispectral,
            num_levels=self.config.num_levels,
            base_res=self.config.base_res,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            features_per_level=self.config.features_per_level,
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            geo_feat_dim=self.config.geo_feat_dim,
            hidden_dim_ms=self.config.hidden_dim_ms,
            num_layers_ms=self.config.num_layers_ms,
            implementation=self.config.implementation,
            use_pred_normals=self.config.predict_normals,
            spatial_distortion=scene_contraction,
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        self.proposal_networks = torch.nn.ModuleList()
        for i in range(num_prop_nets):
            prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
        self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        self.renderer_ms = MSRenderer(background_color=self.config.background_color,
                                      num_ms=self.config.num_multispectral,
                                      semantic=self.config.senmantic)
        self.renderer_accumulation = AccumulationRenderer()

        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()

        self.normals_shader = NormalsShader()

        self.ms_loss = MSELoss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.sam = SpectralAngleMapper()
        # self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        return param_groups

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        # anneal the weights of the proposal network before doing PDF sampling
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                self.step = step
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        ms = self.renderer_ms(ms=field_outputs[FieldHeadNames.MS], weights=weights)

        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "ms": ms,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)

        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_ms = batch["image"].to(self.device)
        gt_ms = self.renderer_ms.blend_background(gt_ms)  # RGB or RGBA image
        predicted_ms = outputs["ms"]
        metrics_dict["psnr"] = self.psnr(predicted_ms, gt_ms)

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        # todo seg
        pred_ms, gt_ms = self.renderer_ms.blend_background_for_loss_computation(
            pred_image=outputs["ms"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        loss_dict["ms_loss"] = self.ms_loss(gt_ms, pred_ms)
        # loss_dict["sam_loss"] = sam_loss(pred_ms, gt_ms) * self.config.sam_loss_mult
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_ms = batch["image"].to(self.device)
        predicted_ms = outputs["ms"]  # Blended with background (black if random background)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        expected_depth = colormaps.apply_depth_colormap(
            outputs["expected_depth"],
            accumulation=outputs["accumulation"],
        )
        if self.config.senmantic and gt_ms.shape[-1] != predicted_ms.shape[-1]:
            gt_ms = gt_ms[..., :-1] * gt_ms[..., -1:]
        combined_ms = torch.cat([gt_ms, predicted_ms], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)
        combined_expected_depth = torch.cat([expected_depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_ms = torch.moveaxis(gt_ms, -1, 0)[None, ...]
        predicted_ms = torch.moveaxis(predicted_ms, -1, 0)[None, ...]

        psnr = self.psnr(gt_ms, predicted_ms)
        ssim = self.ssim(gt_ms, predicted_ms)
        # lpips = self.lpips(gt_ms, predicted_ms)
        mse = self.ms_loss(gt_ms, predicted_ms)
        sam = self.sam(gt_ms.permute((0, 3, 1, 2)), predicted_ms.permute((0, 3, 1, 2)))


        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "mse": float(mse), "sam": float(sam)}  # type: ignore
        # metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_ms, "accumulation": combined_acc, "depth": combined_depth,
                       "expected_depth": combined_expected_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

    def get_rgba_image(self, outputs: Dict[str, torch.Tensor], output_name: str = "ms") -> torch.Tensor:
        ms = outputs[output_name]
        return torch.cat((ms, torch.ones_like(ms[..., :1])), dim=-1)


@dataclass
class MSSuperResolutionModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: MSSuperResolutionModel)
    num_multispectral: int = 25
    num_levels: int = 16
    base_res: int = 16
    max_res: int = 2048
    log2_hashmap_size: int = 19
    features_per_level: int = 2
    num_layers: int = 2
    hidden_dim: int = 64
    geo_feat_dim: int = 15
    num_layers_ms: int = 3
    hidden_dim_ms: int = 64
    implementation: Literal["tcnn", "torch"] = "tcnn"
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="SO3xR3"))
    use_proposal_weight_anneal: bool = True
    num_proposal_iterations: int = 2
    use_same_proposal_network: bool = False
    proposal_net_args_list: List[Dict] = field(
        default_factory=lambda: [
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 128, "use_linear": False},
            {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256, "use_linear": False},
        ]
    )
    average_init_density: float = 1.0

    proposal_update_every: int = 5
    proposal_warmup: int = 5000
    proposal_initial_sampler: Literal["piecewise", "uniform"] = "piecewise"
    num_nerf_samples_per_ray: int = 48
    num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    use_single_jitter: bool = True
    near_plane: float = 0.05
    far_plane: float = 1000.0
    predict_normals: bool = False
    background_color: Literal["random", "last_sample", "black", "white", "ms_white", "ms_black"] = "random"
    proposal_weights_anneal_max_num_iters: int = 1000
    proposal_weights_anneal_slope: float = 10.0
    disable_scene_contraction: bool = False

    interlevel_loss_mult: float = 1.0
    distortion_loss_mult: float = 0.002
    orientation_loss_mult: float = 0.0001
    pred_normal_loss_mult: float = 0.001

    senmantic: bool = False


class MSSuperResolutionModel(Model):
    config: MSSuperResolutionModelConfig

    def populate_modules(self):
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        self.field = MSNerfField(
            aabb=self.scene_box.aabb,
            num_images=self.num_train_data,
            num_multispectral=self.config.num_multispectral,
            num_levels=self.config.num_levels,
            base_res=self.config.base_res,
            max_res=self.config.max_res,
            log2_hashmap_size=self.config.log2_hashmap_size,
            features_per_level=self.config.features_per_level,
            num_layers=self.config.num_layers,
            hidden_dim=self.config.hidden_dim,
            geo_feat_dim=self.config.geo_feat_dim,
            hidden_dim_ms=self.config.hidden_dim_ms,
            num_layers_ms=self.config.num_layers_ms,
            implementation=self.config.implementation,
            use_pred_normals=self.config.predict_normals,
            spatial_distortion=scene_contraction,
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        self.proposal_networks = torch.nn.ModuleList()
        for i in range(num_prop_nets):
            prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                average_init_density=self.config.average_init_density,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
        self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        self.renderer_ms = MSSRRenderer(background_color=self.config.background_color,
                                        num_ms=self.config.num_multispectral,
                                        semantic=self.config.senmantic)
        self.renderer_accumulation = AccumulationRenderer()

        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()

        self.normals_shader = NormalsShader()

        self.ms_loss = MSELoss()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        # self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        return param_groups

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        # anneal the weights of the proposal network before doing PDF sampling
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                self.step = step
                train_frac = np.clip(step / N, 0, 1)
                self.step = step

                def bias(x, b):
                    return b * x / ((b - 1) * x + 1)

                anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        if self.training:
            self.camera_optimizer.apply_to_raybundle(ray_bundle)
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        ms = self.renderer_ms(ms=field_outputs[FieldHeadNames.MS], weights=weights)

        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "ms": ms,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        if ray_bundle.metadata.get("ms_index", None) is not None:
            outputs["ms_index"] = ray_bundle.metadata["ms_index"]

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)

        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])
        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        gt_ms = batch["image"].to(self.device)
        gt_ms = self.renderer_ms.blend_background(gt_ms)  # RGB or RGBA image
        predicted_ms = outputs["ms"]
        metrics_dict["psnr"] = self.psnr(predicted_ms, gt_ms)

        if self.training:
            metrics_dict["distortion"] = distortion_loss(outputs["weights_list"], outputs["ray_samples_list"])

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        image = batch["image"].to(self.device)
        pred_ms, gt_ms = self.renderer_ms.blend_background_for_loss_computation(
            pred_image=outputs["ms"],
            ms_index=outputs["ms_index"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )

        loss_dict["ms_loss"] = self.ms_loss(gt_ms, pred_ms)
        # loss_dict["sam_loss"] = sam_loss(pred_ms, gt_ms) * self.config.sam_loss_mult
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs["rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
        return loss_dict

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_ms = batch["image"].to(self.device)
        predicted_ms = outputs["ms"]  # Blended with background (black if random background)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        expected_depth = colormaps.apply_depth_colormap(
            outputs["expected_depth"],
            accumulation=outputs["accumulation"],
        )
        predicted_ms = self.renderer_ms.extract_band(predicted_ms, outputs["ms_index"])
        combined_ms = torch.cat([gt_ms, predicted_ms], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)
        combined_expected_depth = torch.cat([expected_depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_ms = torch.moveaxis(gt_ms, -1, 0)[None, ...]
        predicted_ms = torch.moveaxis(predicted_ms, -1, 0)[None, ...]

        psnr = self.psnr(gt_ms, predicted_ms)
        ssim = self.ssim(gt_ms, predicted_ms)
        # lpips = self.lpips(gt_ms, predicted_ms)
        mse = self.ms_loss(gt_ms, predicted_ms)
        # sam = self.sam(gt_ms.permute((0, 3, 1, 2)), predicted_ms.permute((0, 3, 1, 2)))

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "mse": float(mse),
                        # "sam": float(sam)
                        }  # type: ignore
        # metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_ms, "accumulation": combined_acc, "depth": combined_depth,
                       "expected_depth": combined_expected_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

    def get_rgba_image(self, outputs: Dict[str, torch.Tensor], output_name: str = "ms") -> torch.Tensor:
        ms = outputs[output_name]
        return torch.cat((ms, torch.ones_like(ms[..., :1])), dim=-1)
