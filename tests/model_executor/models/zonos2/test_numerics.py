# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Small CPU regressions for expert packing and unnormalized biased routing."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.zonos2.configuration_zonos2 import Zonos2Config
from vllm_omni.model_executor.models.zonos2.zonos2_talker import Zonos2MoE, Zonos2RMSNorm, Zonos2RotaryEmbedding

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FixedRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("balancing_biases", torch.tensor([0.0, 0.5, 1.0]))

    def forward(self, x, prev_state):
        return torch.tensor([[0.6, 0.3, 0.1]]).expand(x.shape[0], -1), x


def _moe(topk):
    cfg = Zonos2Config(dim=2, n_heads=1, head_dim=2, moe_router_dim=2, moe_n_experts=3, special_topk_layers={3: topk})
    module = Zonos2MoE(cfg, layer_id=3)
    module.router = _FixedRouter()
    with torch.no_grad():
        module.experts.w13.zero_()
        module.experts.w2.zero_()
        # Every expert sees gate=1, up=2 on input [1,1]. All other channels
        # are zero. Only experts 1 and 2 are selected by biased ranking.
        module.experts.w13[:, 0, 0] = 1
        module.experts.w13[:, 1, 1] = 2
        module.experts.w2[0, :, 0] = torch.tensor([100.0, 100.0])
        module.experts.w2[1, :, 0] = torch.tensor([3.0, -4.0])
        module.experts.w2[2, :, 0] = torch.tensor([5.0, 6.0])
    return module


def test_expert_packing_keeps_canonical_parameters_and_nonpersistent_cache():
    moe = _moe(1)
    with torch.no_grad():
        for row, value in enumerate((11.0, 22.0, 33.0, 44.0)):
            moe.experts.w13[:, row].fill_(value)
    canonical = moe.experts.w13.detach().clone()
    names = set(dict(moe.named_parameters()))
    moe.prepare_expert_weights()
    packed = moe._packed_w13
    assert packed.is_contiguous()
    assert torch.equal(moe.experts.w13, canonical)
    assert torch.equal(packed[:, 0], torch.full_like(packed[:, 0], 11))
    assert torch.equal(packed[:, 1], torch.full_like(packed[:, 1], 33))
    assert torch.equal(packed[:, 3072], torch.full_like(packed[:, 3072], 22))
    assert torch.equal(packed[:, 3073], torch.full_like(packed[:, 3073], 44))
    assert set(dict(moe.named_parameters())) == names
    assert "_packed_w13" not in moe.state_dict()
    with torch.no_grad():
        moe.experts.w13[:, 0].fill_(55)
    moe.prepare_expert_weights()
    assert torch.equal(moe._packed_w13[:, 0], torch.full_like(packed[:, 0], 55))


@pytest.mark.parametrize("topk", [1, 2])
def test_biased_selection_keeps_prebias_probabilities_without_renormalization(topk):
    moe = _moe(topk)
    inputs = torch.ones((1, 2))
    output, state = moe(inputs, None)
    # Legacy scores choose expert 2 first (1.1), expert 1 second (0.8).
    # Its real probability is still 0.1, not 1.1 or renormalized to 1.0.
    weighted_down = torch.tensor([0.5, 0.6])
    if topk == 2:
        weighted_down += torch.tensor([0.9, -1.2])
    expected = (2 / (1 + math.exp(-1))) * weighted_down
    torch.testing.assert_close(output[0], expected)
    assert torch.equal(state, inputs)


def test_interleaved_rope_keeps_fp32_cache_across_module_dtype_changes():
    cfg = Zonos2Config(dim=16, head_dim=8, n_heads=2, n_kv_heads=1, max_seqlen=1024)
    rope = Zonos2RotaryEmbedding(cfg)
    before = rope.cos_sin_cache.clone()
    rope.to(dtype=torch.bfloat16)
    assert rope.cos_sin_cache.dtype == torch.float32
    torch.testing.assert_close(rope.cos_sin_cache, before, rtol=0, atol=0)
    assert "cos_sin_cache" not in rope.state_dict()

    positions = torch.tensor([3, 257, 1023])
    q = torch.tensor([[1, 2, -3, 4, 5, -6, 7, 8] * 2] * 3, dtype=torch.bfloat16)
    k = q[:, :8].clone()
    q_before, k_before = q.clone(), k.clone()
    expected = torch.empty_like(q)
    # Independent scalar oracle for adjacent (interleaved) pairs. It never
    # rounds sin/cos to BF16 before rotating the activation.
    for row, position in enumerate(positions.tolist()):
        for head in range(2):
            for pair in range(4):
                angle = position / (10000 ** (2 * pair / 8))
                a, b = float(q[row, head * 8 + 2 * pair]), float(q[row, head * 8 + 2 * pair + 1])
                expected[row, head * 8 + 2 * pair] = a * math.cos(angle) - b * math.sin(angle)
                expected[row, head * 8 + 2 * pair + 1] = b * math.cos(angle) + a * math.sin(angle)
    q_rot, k_rot = rope.forward_native(positions, q, k)
    assert k_rot is not None
    assert q_rot.dtype == torch.bfloat16 and k_rot.dtype == torch.bfloat16
    torch.testing.assert_close(q_rot, expected, rtol=0, atol=0)
    torch.testing.assert_close(k_rot, expected[:, :8], rtol=0, atol=0)
    torch.testing.assert_close(q, q_before, rtol=0, atol=0)
    torch.testing.assert_close(k, k_before, rtol=0, atol=0)


def test_fused_residual_norm_uses_unrounded_sum_and_float_weight_multiply():
    norm = Zonos2RMSNorm(2, eps=1e-5, dtype=torch.bfloat16)
    with torch.no_grad():
        norm.weight.copy_(torch.tensor([1.625, 1.125], dtype=torch.bfloat16))
    x = torch.ones((1, 2), dtype=torch.bfloat16)
    residual = torch.tensor([[0.005859375, 0.01171875]], dtype=torch.bfloat16)
    a, b = 1.005859375, 1.01171875
    scale = math.sqrt((a * a + b * b) / 2 + 1e-5)
    expected = torch.tensor([[a / scale * 1.625, b / scale * 1.125]], dtype=torch.bfloat16)
    actual, saved_residual = norm.forward_native(x, residual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(saved_residual, torch.tensor([[a, b]], dtype=torch.bfloat16), rtol=0, atol=0)
    # The norm consumes the original FP32 sum, independently of the rounded
    # residual tensor saved for the next block.
    rounded = saved_residual.float()
    wrong = (rounded * torch.rsqrt(rounded.square().mean(-1, keepdim=True) + 1e-5) * norm.weight.float()).to(
        torch.bfloat16
    )
    assert not torch.equal(actual, wrong)
