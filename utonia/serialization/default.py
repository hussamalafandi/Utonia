"""
Serialization Encoding
Pointcept detached version

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from .z_order import xyz2key as z_order_encode_
from .z_order import key2xyz as z_order_decode_
from .hilbert import encode as hilbert_encode_
from .hilbert import decode as hilbert_decode_


@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z"):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}
    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code
    return code


@torch.inference_mode()
def encode_orders(grid_coord, batch, depth, orders):
    """Batched serialization codes for several orders: ``(len(orders), N)``.

    Same codes as ``torch.stack([encode(grid_coord, batch, depth, o) for o in
    orders])``, with two changes in cost and one in meaning:

    - the Hilbert orders are encoded in one call over the stacked (plain and
      transposed) coordinates: Hilbert encoding is a Python loop of tiny kernels
      over bits x dims, so this halves its launches;
    - ``depth`` may be a per-sample list. Each sample is then Hilbert-encoded in
      its own cube, which makes its order independent of the other samples in
      the batch (z-order codes do not depend on the depth). The batch index
      still occupies the top bits above the largest depth, so codes sort by
      sample first.
    """
    orders = [orders] if isinstance(orders, str) else list(orders)
    assert set(orders) <= {"z", "z-trans", "hilbert", "hilbert-trans"}, orders
    per_sample = isinstance(depth, (list, tuple))
    depth_max = max(depth) if per_sample else depth
    trans = grid_coord[:, [1, 0, 2]]
    codes = {}
    hilbert_orders = [o for o in orders if o.startswith("hilbert")]
    if hilbert_orders:
        coords = torch.cat([grid_coord if o == "hilbert" else trans for o in hilbert_orders])
        if not per_sample or len(set(depth)) == 1:
            code = hilbert_encode(coords, depth=depth_max)
        else:
            point_depth = torch.as_tensor(depth, device=grid_coord.device)[batch.long()]
            point_depth = point_depth.repeat(len(hilbert_orders))
            code = torch.empty(coords.shape[0], dtype=torch.long, device=grid_coord.device)
            for d in sorted(set(depth)):
                sel = point_depth == d
                code[sel] = hilbert_encode(coords[sel], depth=d)
        codes.update(zip(hilbert_orders, code.split(grid_coord.shape[0])))
    if "z" in orders:
        codes["z"] = z_order_encode(grid_coord, depth=depth_max)
    if "z-trans" in orders:
        codes["z-trans"] = z_order_encode(trans, depth=depth_max)
    code = torch.stack([codes[o] for o in orders])
    if batch is not None:
        code = batch.long() << depth_max * 3 | code
    return code


@torch.inference_mode()
def decode(code, depth=16, order="z"):
    assert order in {"z", "hilbert"}
    batch = code >> depth * 3
    code = code & ((1 << depth * 3) - 1)
    if order == "z":
        grid_coord = z_order_decode(code, depth=depth)
    elif order == "hilbert":
        grid_coord = hilbert_decode(code, depth=depth)
    else:
        raise NotImplementedError
    return grid_coord, batch


def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # we block the support to batch, maintain batched code in Point class
    code = z_order_encode_(x, y, z, b=None, depth=depth)
    return code


def z_order_decode(code: torch.Tensor, depth):
    x, y, z = z_order_decode_(code, depth=depth)
    grid_coord = torch.stack([x, y, z], dim=-1)  # (N,  3)
    return grid_coord


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)


def hilbert_decode(code: torch.Tensor, depth: int = 16):
    return hilbert_decode_(code, num_dims=3, num_bits=depth)
