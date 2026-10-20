"""
Point Transformer - V3 Mode3 - Utonia
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


import os
import math
from packaging import version
from huggingface_hub import hf_hub_download, PyTorchModelHubMixin
from addict import Dict
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import spconv.pytorch as spconv
import torch_scatter
from timm.layers import DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None

from .structure import Point
from .module import PointSequential, PointModule
from .utils import offset2bincount

MODELS = [
    "utonia",
    "utonia_linear_prob_head_sc",
]


class Point3DRoPE(nn.Module):
    def __init__(self, head_dim, base=10000):
        super().__init__()
        assert (
            head_dim % 3 == 0
        ), f"Head dimension must be divisible by 3 for 3D RoPE, {head_dim}"

        self.head_dim = head_dim
        self.chunk_dim = head_dim // 3
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.chunk_dim, 2).float() / self.chunk_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def get_cos_sin(self, xyz):
        x = xyz[:, 0:1]
        y = xyz[:, 1:2]
        z = xyz[:, 2:3]
        freqs = self.inv_freq.unsqueeze(0)

        emb_x = x * freqs
        emb_y = y * freqs
        emb_z = z * freqs

        emb_x = torch.cat((emb_x, emb_x), dim=-1)
        emb_y = torch.cat((emb_y, emb_y), dim=-1)
        emb_z = torch.cat((emb_z, emb_z), dim=-1)
        emb_3d = torch.cat((emb_x, emb_y, emb_z), dim=-1)
        return emb_3d.cos().unsqueeze(1), emb_3d.sin().unsqueeze(1)

    def rotate_half(self, x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, xyz):
        cos, sin = self.get_cos_sin(xyz)
        qs = torch.split(q, self.chunk_dim, dim=-1)
        ks = torch.split(k, self.chunk_dim, dim=-1)

        coss = torch.split(cos, self.chunk_dim, dim=-1)
        sins = torch.split(sin, self.chunk_dim, dim=-1)

        q_outs = []
        k_outs = []

        for i in range(3):
            q_part = (qs[i] * coss[i]) + (self.rotate_half(qs[i]) * sins[i])
            k_part = (ks[i] * coss[i]) + (self.rotate_half(ks[i]) * sins[i])

            q_outs.append(q_part)
            k_outs.append(k_part)

        q_rot = torch.cat(q_outs, dim=-1)
        k_rot = torch.cat(k_outs, dim=-1)

        return q_rot, k_rot


# Compiled eval path of a Block (``PointTransformerV3(compile_blocks=True)``). The eager
# block spends most of its GPU time in small memory-bound kernels: RoPE recomputed per
# block and head in fp32, the q/k/v stack and cast for flash attention, LayerNorm and the
# residual adds. These functions hold that work between the two kernels torch.compile
# cannot trace (the sparse conv and flash attention) so inductor fuses it. Same math as
# the eager path; RoPE cos/sin is computed once per stage (it depends only on the
# coordinates) and gathered, which is exact.
def _layer_norm(x, m):
    return nn.functional.layer_norm(x, m.normalized_shape, m.weight, m.bias, m.eps)


def _layer_scale(x, seq):
    m = seq[0]
    return x * m.gamma if isinstance(m, LayerScale) else x


def _rope_cos_sin(coord, inv_freq):
    freqs = inv_freq.unsqueeze(0)
    emb = torch.cat(
        [torch.cat((coord[:, j : j + 1] * freqs,) * 2, dim=-1) for j in range(3)], dim=-1
    )
    return emb.cos(), emb.sin()


def _rope_rotate(x, cos, sin, chunk):
    out = []
    for i in range(3):
        xi = x[..., i * chunk : (i + 1) * chunk]
        h = chunk // 2
        rot = torch.cat((-xi[..., h:], xi[..., :h]), dim=-1)
        sl = slice(i * chunk, (i + 1) * chunk)
        out.append(xi * cos[..., sl] + rot * sin[..., sl])
    return torch.cat(out, dim=-1)


def _cpe_tail(shortcut, conv_feat, linear, norm):
    return shortcut + _layer_norm(
        nn.functional.linear(conv_feat, linear.weight, linear.bias), norm
    )


def _attn_qkv(feat, norm1, qkv, cos, sin, num_heads, chunk):
    """norm1 + qkv projection + RoPE on q/k, packed (N, 3, H, C/H) bf16 for flash attention."""
    h = nn.functional.linear(_layer_norm(feat, norm1), qkv.weight, qkv.bias)
    n, c = h.shape[0], h.shape[-1] // 3
    x = h.view(n, 3, num_heads, c // num_heads)
    roped = _rope_rotate(x, cos.view(n, 1, 1, -1), sin.view(n, 1, 1, -1), chunk)
    is_qk = torch.arange(3, device=x.device).view(1, 3, 1, 1) < 2
    return torch.where(is_qk, roped, x.float()).to(torch.bfloat16)


def _attn_out_mlp(attn_feat, shortcut, proj, ls1, norm2, fc1, fc2, ls2):
    x = shortcut + _layer_scale(
        nn.functional.linear(attn_feat, proj.weight, proj.bias), ls1
    )
    y = nn.functional.linear(_layer_norm(x, norm2), fc1.weight, fc1.bias)
    y = nn.functional.linear(nn.functional.gelu(y), fc2.weight, fc2.bias)
    return x + _layer_scale(y, ls2)


_COMPILED = {}


def _compiled(fn):
    if fn not in _COMPILED:
        _COMPILED[fn] = torch.compile(fn, dynamic=True)
    return _COMPILED[fn]


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


@torch.no_grad()
def padding_and_inverse(offset, patch_size):
    """Patch padding of a batched point cloud: ``(pad, unpad, cu_seqlens)``.

    Each sample with more than ``patch_size`` points is padded to a multiple of
    ``patch_size`` by repeating its last full patch's tail; ``pad`` maps padded
    positions to points, ``unpad`` maps points to padded positions and
    ``cu_seqlens`` holds the patch boundaries (int32, for flash attention).

    Vectorised form of the original per-sample loop, which indexed with GPU
    scalars and synchronised the host several times per sample per stage. Here
    the only sync is one read of the per-sample counts, needed for the output
    shapes; the result is identical (integer index maths).
    """
    K = patch_size
    device = offset.device
    counts = offset2bincount(offset)
    padded = torch.where(counts > K, (counts + K - 1) // K * K, counts)
    n_chunks = (padded + K - 1) // K
    n_real, n_pad, n_cu = (int(v) for v in torch.stack(
        [counts.sum(), padded.sum(), n_chunks.sum()]
    ).tolist())
    start = torch.cumsum(counts, 0) - counts
    start_pad = torch.cumsum(padded, 0) - padded
    shift = start_pad - start
    sample_ids = torch.arange(len(counts), device=device)

    sample = torch.repeat_interleave(sample_ids, counts, output_size=n_real)
    unpad = torch.arange(n_real, device=device) + shift[sample]

    sample_pad = torch.repeat_interleave(sample_ids, padded, output_size=n_pad)
    pos = torch.arange(n_pad, device=device)
    # Tail positions (past the sample's real points) repeat the entries one patch
    # earlier: the original loop's ``pad[tail] = pad[tail - K]``.
    tail = (pos - start_pad[sample_pad]) >= counts[sample_pad]
    pad = torch.where(tail, pos - K, pos) - shift[sample_pad]

    chunk_sample = torch.repeat_interleave(sample_ids, n_chunks, output_size=n_cu)
    chunk_rank = torch.arange(n_cu, device=device) - (torch.cumsum(n_chunks, 0) - n_chunks)[
        chunk_sample
    ]
    cu_seqlens = torch.cat(
        [start_pad[chunk_sample] + K * chunk_rank, padded.sum().reshape(1)]
    ).to(torch.int32)
    return pad, unpad, cu_seqlens


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        rope_base=10,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None
        self.rope = Point3DRoPE(head_dim=channels // num_heads, base=rope_base)
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            pad, unpad, cu_seqlens = padding_and_inverse(point.offset, self.patch_size)
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = cu_seqlens
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point):
        if not self.enable_flash:
            self.patch_size = min(
                offset2bincount(point.offset).min().tolist(), self.patch_size_max
            )

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        rope_coord = point.coord[order].clone()
        if not self.enable_flash:
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = (
                qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            )
            # rope expects 3D (total_N, H, C//H); reshape (N', H, K, C//H) accordingly
            N_prime = q.shape[0]
            q_3d = q.permute(0, 2, 1, 3).reshape(-1, H, C // H)
            k_3d = k.permute(0, 2, 1, 3).reshape(-1, H, C // H)
            q_3d, k_3d = self.rope(q_3d, k_3d, rope_coord)
            q = q_3d.reshape(N_prime, K, H, C // H).permute(0, 2, 1, 3)
            k = k_3d.reshape(N_prime, K, H, C // H).permute(0, 2, 1, 3)
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            qkv = qkv.reshape(-1, 3, H, C // H)
            q, k, v = qkv.unbind(dim=1)
            q, k = self.rope(q, k, rope_coord)
            qkv_roped = torch.stack([q, k, v], dim=1)
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv_roped.to(torch.bfloat16),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        rope_base=10,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.compile_blocks = False  # set by PointTransformerV3(compile_blocks=...)

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.ls1 = PointSequential(
            LayerScale(channels, init_values=layer_scale)
            if layer_scale is not None
            else nn.Identity()
        )
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            rope_base=rope_base,
            shift_coords=shift_coords,
            jitter_coords=jitter_coords,
            rescale_coords=rescale_coords,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.ls2 = PointSequential(
            LayerScale(channels, init_values=layer_scale)
            if layer_scale is not None
            else nn.Identity()
        )
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        if (
            self.compile_blocks
            and not self.training
            and self.pre_norm
            and self.attn.enable_flash
            and isinstance(self.mlp[0].act, nn.GELU)
        ):
            return self._forward_compiled(point)
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.ls1(self.attn(point)))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.ls2(self.mlp(point)))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point

    def _forward_compiled(self, point: Point):
        """Eval forward with the elementwise work compiled (see ``_attn_qkv``)."""
        attn, rope = self.attn, self.attn.rope
        shortcut = point.feat
        point.sparse_conv_feat = self.cpe[0](point.sparse_conv_feat)
        feat = _compiled(_cpe_tail)(
            shortcut, point.sparse_conv_feat.features, self.cpe[1], self.cpe[2]
        )
        cached = point.get("rope_cos_sin")
        if cached is None or cached[0] is not point.coord or cached[1] != rope.chunk_dim:
            cached = (point.coord, rope.chunk_dim) + tuple(
                _compiled(_rope_cos_sin)(point.coord, rope.inv_freq)
            )
            point["rope_cos_sin"] = cached
        pad, unpad, cu_seqlens = attn.get_padding_and_inverse(point)
        order = point.serialized_order[attn.order_index][pad]
        inverse = unpad[point.serialized_inverse[attn.order_index]]
        packed = _compiled(_attn_qkv)(
            feat, self.norm1[0], attn.qkv, cached[2], cached[3], attn.num_heads, rope.chunk_dim
        )
        # The serialization gather stays eager: inductor's version of this gather is slow.
        qkv = packed.view(packed.shape[0], -1).index_select(0, order)
        attn_feat = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv.view(-1, *packed.shape[1:]),
            cu_seqlens,
            max_seqlen=attn.patch_size,
            dropout_p=0.0,
            softmax_scale=attn.scale,
        ).reshape(-1, attn.channels)
        if torch.is_autocast_enabled("cuda"):
            attn_feat = attn_feat.to(torch.get_autocast_dtype("cuda"))
        else:
            attn_feat = attn_feat.to(attn.qkv.weight.dtype)
        out = _compiled(_attn_out_mlp)(
            attn_feat.index_select(0, inverse),
            feat,
            attn.proj,
            self.ls1,
            self.norm2[0],
            self.mlp[0].fc1,
            self.mlp[0].fc2,
            self.ls2,
        )
        point.feat = out
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(out)
        return point


class GridPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
        per_sample_depth=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.per_sample_depth = per_sample_depth
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        if "grid_coord" in point.keys():
            grid_coord = point.grid_coord
        elif {"coord", "grid_size"}.issubset(point.keys()):
            grid_coord = torch.div(
                point.coord - point.coord.min(0)[0],
                point.grid_size,
                rounding_mode="trunc",
            ).int()
        else:
            raise AssertionError(
                "[gird_coord] or [coord, grid_size] should be include in the Point"
            )
        grid_coord = torch.div(grid_coord, self.stride, rounding_mode="trunc").long()
        # One int64 key per point, (batch, x, y, z) in 16-bit fields: the same
        # sorted order as torch.unique(dim=0) over the rows (batch, x), y, z, which
        # is far slower. Grid coords are < 2^16 (serialization asserts depth <= 16).
        key = (
            point.batch.long() << 48
            | grid_coord[:, 0] << 32
            | grid_coord[:, 1] << 16
            | grid_coord[:, 2]
        )
        key, cluster, counts = torch.unique(
            key, sorted=True, return_inverse=True, return_counts=True
        )
        mask16 = (1 << 16) - 1
        grid_coord = torch.stack(
            [(key >> 32) & mask16, (key >> 16) & mask16, key & mask16], dim=1
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=grid_coord,
            batch=point.batch[head_indices],
        )
        if "origin_coord" in point.keys():
            point_dict["origin_coord"] = torch_scatter.segment_csr(
                point.origin_coord[indices], idx_ptr, reduce="mean"
            )
        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context
        if "name" in point.keys():
            point_dict["name"] = point.name
        if "split" in point.keys():
            point_dict["split"] = point.split
        if "color" in point.keys():
            point_dict["color"] = torch_scatter.segment_csr(
                point.color[indices], idx_ptr, reduce="mean"
            )
        if "grid_size" in point.keys():
            point_dict["grid_size"] = point.grid_size * self.stride

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
            point_dict["idx_ptr"] = idx_ptr
        order = point.order
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.serialization(
            order=order,
            shuffle_orders=self.shuffle_orders,
            per_sample_depth=self.per_sample_depth,
        )
        point.sparsify()
        return point


class GridUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pooling_inverse
        feat = point.feat

        parent = self.proj_skip(parent)
        parent.feat = parent.feat + self.proj(point).feat[inverse]
        parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)

        if self.traceable:
            point.feat = feat
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
        mask_token=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        self.stem = PointSequential(linear=nn.Linear(in_channels, embed_channels))
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

        if mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, embed_channels))
        else:
            self.mask_token = None

    def forward(self, point: Point):
        point = self.stem(point)
        if "mask" in point.keys():
            point.feat = torch.where(
                point.mask.unsqueeze(-1),
                self.mask_token.to(point.feat.dtype),
                point.feat,
            )
        return point


class PointTransformerV3(PointModule, PyTorchModelHubMixin):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(3, 3, 3, 3),
        dec_channels=(96, 96, 192, 384),
        dec_num_head=(6, 6, 12, 32),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        layer_scale=None,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=False,
        rope_base=10,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
        per_sample_depth=False,
        compile_blocks=False,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        # Serialize each sample in its own cube (depth from its own extent) instead
        # of one cube sized by the largest sample in the batch: the Hilbert orders
        # depend on the depth, so with the batch-wide depth a sample's patches (and
        # its features) change with the other samples it is batched with.
        self.per_sample_depth = per_sample_depth
        self.order = [order] if isinstance(order, str) else order
        self.enc_mode = enc_mode
        self.shuffle_orders = shuffle_orders
        self.freeze_encoder = freeze_encoder

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.enc_mode or self.num_stages == len(dec_depths) + 1
        assert self.enc_mode or self.num_stages == len(dec_channels) + 1
        assert self.enc_mode or self.num_stages == len(dec_num_head) + 1
        assert self.enc_mode or self.num_stages == len(dec_patch_size) + 1

        # normalization layer
        ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=ln_layer,
            act_layer=act_layer,
            mask_token=mask_token,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    GridPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        per_sample_depth=per_sample_depth,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        layer_scale=layer_scale,
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        rope_base=rope_base,
                        shift_coords=shift_coords,
                        jitter_coords=jitter_coords,
                        rescale_coords=rescale_coords,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.enc_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    GridUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        traceable=traceable,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            layer_scale=layer_scale,
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            rope_base=rope_base,
                            shift_coords=shift_coords,
                            jitter_coords=jitter_coords,
                            rescale_coords=rescale_coords,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")
        # Eval-mode blocks run their elementwise work through torch.compile (Block
        # ._forward_compiled); training keeps the eager path (drop path, dropout).
        for m in self.modules():
            if isinstance(m, Block):
                m.compile_blocks = compile_blocks
        if self.freeze_encoder:
            for p in self.embedding.parameters():
                p.requires_grad = False
            for p in self.enc.parameters():
                p.requires_grad = False
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, spconv.SubMConv3d):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, data_dict):
        point = Point(data_dict)
        point = self.embedding(point)

        point.serialization(
            order=self.order,
            shuffle_orders=self.shuffle_orders,
            per_sample_depth=self.per_sample_depth,
        )
        point.sparsify()

        point = self.enc(point)
        if not self.enc_mode:
            point = self.dec(point)
        return point


def load(
    name: str = "utonia",
    repo_id="Pointcept/utonia",
    download_root: str = None,
    custom_config: dict = None,
    ckpt_only: bool = False,
):
    if name in MODELS:
        print(f"Loading checkpoint from HuggingFace: {name} ...")
        ckpt_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{name}.pth",
            repo_type="model",
            revision="main",
            local_dir=download_root or os.path.expanduser("~/.cache/utonia/ckpt"),
        )
    elif os.path.isfile(name):
        print(f"Loading checkpoint in local path: {name} ...")
        ckpt_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {MODELS}")

    if version.parse(torch.__version__) >= version.parse("2.4"):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    if custom_config is not None:
        for key, value in custom_config.items():
            ckpt["config"][key] = value

    if ckpt_only:
        return ckpt

    model = PointTransformerV3(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_parameters / 1e6:.2f}M")
    return model
