from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from nerfstudio.cameras.rays import Frustums, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.sdf_field import SDFField  # noqa
from nerfstudio.utils.eval_utils import eval_setup

if __name__ == '__main__':
    path = r"D:\files\PHD\myNeRF\nerfstudio\outputs\12_0\mssr\2024-11-28_152436"
    ms_num = 25

    load_config = os.path.join(path, "config.yml")
    filename = os.path.join(path, "pc_sample.ply")
    _, pipeline, _, _ = eval_setup(Path(load_config))
    field = pipeline.model.field
    field.eval()

    batch_size = 1024 * 64 * 64 * 2
    sample_num = 256  # 每个维度的采样数

    e = 1.0e-4
    # scale = 2
    # x = torch.linspace(-scale + e, scale - e, sample_num)
    # y = torch.linspace(-scale + e, scale - e, sample_num)
    # z = torch.linspace(-scale + e, scale - e, sample_num)

    x = torch.linspace(-0.526 + e, 0.344 - e, sample_num)
    y = torch.linspace(-0.263 + e, 0.255 - e, sample_num)
    z = torch.linspace(-1 + e, -0.670 - e, sample_num)

    xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')

    # 将坐标拼接成一个 (N, 3) 的张量，N 是采样点的总数
    positions = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    pn = 0
    all_pn = sample_num ** 3
    all_data = []

    # 对位置张量进行分批处理
    for i in range(0, positions.shape[0], batch_size):
        with torch.no_grad():
            batch_positions = positions[i:i + batch_size].to("cuda")
            directions = torch.rand(batch_size, 3).to("cuda")
            directions = torch.softmax(directions, dim=-1)
            frustums = Frustums(origins=batch_positions, directions=directions,
                                starts=torch.zeros((batch_size, 1)).to("cuda"),
                                ends=torch.zeros((batch_size, 1)).to("cuda"),
                                pixel_area=torch.zeros((batch_size, 1)).to("cuda"))
            raysample = RaySamples(frustums)
            d, ed = field.get_density(raysample)
            d = d.nan_to_num()
            o = field.get_outputs(raysample, ed)[FieldHeadNames.MS]

            se = (10 < d).all(dim=-1)
            d = d[se].to("cpu").numpy()
            o = o[se].to("cpu").numpy()
            batch_positions = batch_positions[se].to("cpu").numpy()

            # d = d.to("cpu").numpy()
            # o = o.to("cpu").numpy()
            # batch_positions = batch_positions.to("cpu").numpy()

            batch_data = np.hstack([
                batch_positions,  # (N, 3)
                d,  # (N, 1)
                o  # (N, ms_dim)
            ])
            all_data.append(batch_data)
            pn += batch_positions.shape[0]
            print(
                f"Processed batch {i // batch_size + 1}: valid points({batch_positions.shape[0]}) sample points({i + batch_size}/{positions.shape[0]})")
    print(f"all points:  {pn}")

    with open(filename, 'wb') as f:
        f.write(f"ply\n".encode())
        f.write(f"format binary_little_endian 1.0\n".encode())
        f.write(f"element vertex {pn}\n".encode())
        f.write(f"property float x\n".encode())
        f.write(f"property float y\n".encode())
        f.write(f"property float z\n".encode())
        f.write(f"property float d\n".encode())
        for i in range(ms_num):
            f.write(f"property float ms_{i + 1}\n".encode())
        f.write(f"end_header\n".encode())
        all_data = np.vstack(all_data).astype(np.float32)
        f.write(all_data.tobytes())
