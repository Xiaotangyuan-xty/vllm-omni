# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ZONOS2 Stage-0 talker: 28-layer GQA + sonic-EDA-MoE backbone emitting
9-codebook DAC tokens per AR step.

Registered under ``Zonos2ForConditionalGeneration`` (canonical HF arch) and
``Zonos2TalkerForConditionalGeneration`` (explicit alias).

M1 skeleton scope:
  * Module tree mirrors the official checkpoint layout 1:1 (507 tensors,
    see tools/convert_zonos2_to_safetensors.py manifest) so dummy loading
    exercises the full parameter surface.
  * Attention runs on vLLM's PagedAttention (KV cache) with interleaved RoPE;
    QK RMSNorm / per-head temperature / headwise sigmoid gate are structurally
    present and get numerically validated in M2a.
  * The sonic EDA router and expert math are plain-torch references; M2b swaps
    in FusedMoE without changing module names.
  * The model-owned sampler emits [B, 9] codes per step with shear (delay)
    padding and eoa countdown; per-request sampling parameters land in M3.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.layers.attention import Attention
from vllm.config import VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.utils import make_layers

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.zonos2.configuration_zonos2 import Zonos2Config


class _WeightModule(nn.Module):
    """Bare holder exposing a single ``weight`` Parameter with a custom shape."""

    def __init__(self, *shape: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(*shape))


class Zonos2Attention(nn.Module):
    """GQA attention with fused wkv, QK RMSNorm, per-head temperature and
    headwise sigmoid gate, on top of vLLM PagedAttention."""

    def __init__(self, config: Zonos2Config, prefix: str):
        super().__init__()
        self.n_heads = config.n_heads  # 16
        self.n_kv_heads = config.n_kv_heads  # 4
        self.head_dim = config.head_dim  # 128
        self.norm_eps = config.norm_eps

        self.wq = nn.Linear(config.dim, config.dim, bias=False)
        self.wkv = _WeightModule(2, config.n_kv_heads * config.head_dim, config.dim)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.gater = nn.Linear(config.dim, config.n_heads, bias=False)
        # per-head learnable temperature, checkpoint key ``attention.temp``
        self.temp = nn.Parameter(torch.ones(1, config.n_heads, 1))

        self.rotary = get_rope(
            config.head_dim,
            max_position=config.max_seqlen,
            is_neox_style=False,  # interleaved RoPE
            rope_parameters={"rope_theta": config.rope_theta},
        )
        self.attn = Attention(
            num_heads=self.n_heads,
            head_size=self.head_dim,
            scale=self.head_dim**-0.5,  # default 1/sqrt(head_dim); temp handles per-head scaling
            num_kv_heads=self.n_kv_heads,
            prefix=f"{prefix}.attn",
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        t = x.shape[0]
        q = self.wq(x)  # [T, 2048]
        kv = F.linear(x, self.wkv.weight.reshape(2 * self.n_kv_heads * self.head_dim, -1))
        k, v = kv.chunk(2, dim=-1)  # [T, 512] each

        q = q.view(t, self.n_heads, self.head_dim)
        k = k.view(t, self.n_kv_heads, self.head_dim)
        v = v.view(t, self.n_kv_heads, self.head_dim)

        # QK RMSNorm (weight-free; official ckpt carries no qk-norm params).
        # Official uses eps=1e-6 here (attention-only), not the model-wide 1e-5.
        q = F.rms_norm(q, (self.head_dim,), eps=1e-6)
        k = F.rms_norm(k, (self.head_dim,), eps=1e-6)

        # Per-head temperature: q scaled by abs(temp) (absolute value per the
        # official implementation; verified in M2a).
        q = q * self.temp.abs()

        q, k = self.rotary(positions, q, k)
        attn_out = self.attn(q, k, v)  # [T, 16*128]

        # Headwise sigmoid gate (Qwen gated-attention style).
        gate = torch.sigmoid(self.gater(x))  # [T, 16]
        attn_out = attn_out.view(t, self.n_heads, self.head_dim) * gate.unsqueeze(-1)
        return self.wo(attn_out.view(t, -1))


class Zonos2DenseFFN(nn.Module):
    """Dense SwiGLU FFN for layers 0-2 and 27 (checkpoint: w_in/w_out)."""

    def __init__(self, config: Zonos2Config):
        super().__init__()
        self.w_in = _WeightModule(2, 3072, config.dim)
        self.w_out = _WeightModule(config.dim, 3072)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Official w_in layout: first half = up (h), second half = gate;
        # y = h * silu(gate) = up * silu(gate).
        h_gate = F.linear(x, self.w_in.weight.reshape(2 * 3072, -1))
        h, gate = h_gate.chunk(2, dim=-1)
        return F.linear(h * F.silu(gate), self.w_out.weight)


class Zonos2SonicRouter(nn.Module):
    """Sonic EDA router: down_proj -> EDA state mix -> RMSNorm -> 3-layer GeLU
    MLP, with aux-loss-free balancing bias on the top-k selection."""

    def __init__(self, config: Zonos2Config, has_prev_state: bool):
        super().__init__()
        rd = config.moe_router_dim
        self.down_proj = nn.Linear(config.dim, rd, bias=True)
        # RMSNorm over router_dim (checkpoint key ``rmsnorm_eda.weight``).
        self.rmsnorm_eda = _WeightModule(rd)
        self.router_mlp = nn.ModuleList(
            [
                nn.Linear(rd, rd, bias=True),
                nn.GELU(),
                nn.Linear(rd, rd, bias=True),
                nn.GELU(),
                nn.Linear(rd, config.moe_n_experts, bias=False),
            ]
        )
        self.balancing_biases = nn.Parameter(torch.zeros(config.moe_n_experts))
        if has_prev_state:
            # Layers 4..26 mix in the previous MoE layer's router state.
            self.router_states_scale = nn.Parameter(torch.zeros(rd))

    def forward(
        self, x: torch.Tensor, prev_router_state: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (expert_prob, next_router_state).

        Official order: down_proj -> EDA blend (scale * prev_state) -> keep the
        pre-norm state for the next MoE layer -> rmsnorm_eda -> router_mlp ->
        softmax (float32). Top-k selection happens in the caller on
        ``expert_prob + balancing_biases`` (legacy strategy); routing weights
        are the pre-bias probabilities.
        """
        h = self.down_proj(x)  # [T, 128]
        if prev_router_state is not None and hasattr(self, "router_states_scale"):
            h = h + self.router_states_scale * prev_router_state
        state = h  # pre-norm; carried to the next MoE layer's EDA blend
        h = F.rms_norm(h, (h.shape[-1],), weight=self.rmsnorm_eda.weight, eps=1e-5)
        for layer in self.router_mlp:
            h = layer(h)
        expert_prob = F.softmax(h.float(), dim=-1)  # [T, 16]
        return expert_prob, state


class Zonos2MoE(nn.Module):
    """MoE FFN: 16 experts (fused w13/w2), top-k per layer from config."""

    def __init__(self, config: Zonos2Config, layer_id: int):
        super().__init__()
        self.topk = config.router_topk(layer_id)
        self.n_experts = config.moe_n_experts
        self.experts = nn.Module()
        self.experts.w13 = nn.Parameter(torch.zeros(self.n_experts, 2 * 3072, config.dim))
        self.experts.w2 = nn.Parameter(torch.zeros(self.n_experts, config.dim, 3072))
        self.router = Zonos2SonicRouter(config, has_prev_state=layer_id > config.moe_start_from_layer)

    def forward(self, x: torch.Tensor, prev_router_state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        expert_prob, state = self.router(x, prev_router_state)
        # Legacy aux-loss-free balancing: top-k on prob + bias, routing weights
        # are the pre-bias probabilities (no renormalization).
        scores = expert_prob + self.router.balancing_biases.float()
        topk_idx = torch.topk(scores, self.topk, dim=-1).indices  # [T, k]
        weights = expert_prob.gather(-1, topk_idx)  # [T, k]

        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            mask = (topk_idx == e).any(dim=-1)
            if not bool(mask.any()):
                continue
            xe = x[mask]
            # Official w13 is row-INTERLEAVED: even rows = gate, odd rows = up.
            w13 = self.experts.w13[e]
            gate = F.linear(xe, w13[0::2])
            up = F.linear(xe, w13[1::2])
            ye = F.linear(F.silu(gate) * up, self.experts.w2[e])
            # routing weight for expert e at its selected position (0 elsewhere)
            w = (weights[mask] * (topk_idx[mask] == e).float()).sum(dim=-1, keepdim=True)
            out[mask] = out[mask] + ye * w.to(ye.dtype)
        return out, state


class Zonos2DecoderLayer(nn.Module):
    def __init__(self, config: Zonos2Config, layer_id: int, prefix: str):
        super().__init__()
        self.layer_id = layer_id
        self.attention_norm = _WeightModule(config.dim)
        self.attention = Zonos2Attention(config, prefix=f"{prefix}.attention")
        self.ffn_norm = _WeightModule(config.dim)
        self.feed_forward: nn.Module
        if config.is_moe_layer(layer_id):
            self.feed_forward = Zonos2MoE(config, layer_id)
        else:
            self.feed_forward = Zonos2DenseFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        prev_router_state: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        eps = 1e-5
        h = x + self.attention(
            F.rms_norm(x, (x.shape[-1],), weight=self.attention_norm.weight, eps=eps), positions
        )
        ff_in = F.rms_norm(h, (h.shape[-1],), weight=self.ffn_norm.weight, eps=eps)
        if isinstance(self.feed_forward, Zonos2MoE):
            ff_out, state = self.feed_forward(ff_in, prev_router_state)
            return h + ff_out, state
        return h + self.feed_forward(ff_in), prev_router_state


class Zonos2MultiEmbedder(nn.Module):
    """10-column embedding tables (9 audio codebooks + 1 text), summed.

    Audio tables: [codebook_size+2=1026, dim], padding_idx=audio_pad_id (1025).
    Text table:   [text_vocab+1=520, dim], padding_idx=text_vocab (519).
    Padding indices zero out pad positions, matching the official MultiEmbedding.
    """

    def __init__(self, config: Zonos2Config):
        super().__init__()
        audio_vocab = config.codebook_vocab_size  # 1026
        text_table_vocab = config.text_vocab + 1  # 520
        self.embedders = nn.ModuleList(
            [
                nn.Embedding(audio_vocab, config.dim, padding_idx=config.audio_pad_id)
                for _ in range(config.n_codebooks)
            ]
            + [nn.Embedding(text_table_vocab, config.dim, padding_idx=config.text_vocab)]
        )
        self.frame_width = config.frame_width

    def forward(self, frame_ids: torch.Tensor) -> torch.Tensor:
        # frame_ids: [T, 10] per-position column ids
        out = None
        for col, table in enumerate(self.embedders):
            e = table(frame_ids[:, col])
            out = e if out is None else out + e
        return out


class Zonos2TalkerForConditionalGeneration(nn.Module):
    """Stage-0 AR backbone for ZONOS2 (see module docstring for M1 scope)."""

    # Runner hooks
    have_multimodal_outputs: bool = True
    prefer_model_sampler: bool = True
    has_postprocess: bool = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        if isinstance(hf_config, Zonos2Config):
            self.config = hf_config
        else:
            self.config = Zonos2Config(**hf_config.to_dict())
        cfg = self.config

        self.multi_embedder = Zonos2MultiEmbedder(cfg)
        _, _, self.layers = make_layers(
            cfg.n_layers,
            lambda prefix: Zonos2DecoderLayer(cfg, layer_id=int(prefix.split(".")[-1]), prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.out_norm = _WeightModule(cfg.dim)
        self.multi_output = nn.Linear(cfg.dim, cfg.n_codebooks * cfg.codebook_vocab_size, bias=False)

        # Speaker chain: Qwen3 voice embedding (2048) -> LDA (1024) -> hidden.
        self.speaker_lda_projection = nn.Linear(cfg.speaker_embedding_dim, cfg.speaker_lda_dim, bias=True)
        self.speaker_projection = nn.Linear(cfg.speaker_lda_dim, cfg.dim, bias=True)

        # LM lifecycle channel reuses the codebook-0 logits slice.
        self.logits_processor = LogitsProcessor(cfg.codebook_vocab_size)

        self._last_audio_codes: torch.Tensor | None = None
        self._postprocess_cursor: int = 0

    # ------------------------------------------------------------------ embed
    def _embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """M1 prompt path: treat each position as a text-column token; audio
        columns contribute their pad-row embeddings. Full 10-column frame
        layout (delay/shear, 17-frame silence tail, speaker slot) is P2-01.
        The summed embedding passes through the weight-free ``emb_norm``
        RMSNorm, matching the official forward."""
        cfg = self.config
        text_ids = input_ids.clamp(min=0, max=cfg.text_vocab)  # text table has 520 rows
        emb = self.multi_embedder.embedders[cfg.n_codebooks](text_ids)
        pad = torch.zeros_like(text_ids) + cfg.audio_pad_id
        for col in range(cfg.n_codebooks):
            emb = emb + self.multi_embedder.embedders[col](pad)
        # official: x = emb_norm(x) with elementwise_affine=False
        emb = F.rms_norm(emb, (cfg.dim,), eps=cfg.norm_eps)
        return emb

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return self._embed_input_ids(input_ids)

    # ----------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            hidden = self._embed_input_ids(input_ids)
        else:
            hidden = inputs_embeds
        prev_router_state: torch.Tensor | None = None
        for layer in self.layers:
            hidden, prev_router_state = layer(hidden, positions, prev_router_state)
        return F.rms_norm(hidden, (hidden.shape[-1],), weight=self.out_norm.weight, eps=self.config.norm_eps)

    # ------------------------------------------------------------ logits/sample
    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: Any = None) -> torch.Tensor:
        fused = self.multi_output(hidden_states)  # [N, 9*1026]
        fused = fused.view(-1, self.config.n_codebooks, self.config.codebook_vocab_size)
        # tanh softcap tau=15 on all codebook logits
        cap = self.config.loss_softcap
        fused = cap * torch.tanh(fused / cap)
        self._last_fused_logits = fused
        # LM lifecycle channel: codebook-0 logits.
        lm_logits = fused[:, 0, :].float()
        return lm_logits

    def sample(self, logits: torch.Tensor, sampling_metadata: Any) -> None:
        """Model-owned 9-codebook sampler (M1: argmax; per-request params in M3).

        Stores ``self._last_audio_codes`` [num_requests, 9] and returns None so
        the runner falls back to the default sampler for the LM lifecycle token.
        """
        fused = getattr(self, "_last_fused_logits", None)
        if fused is None:
            self._last_audio_codes = None
            return None
        codes = fused.argmax(dim=-1)  # [N, 9]
        self._last_audio_codes = codes.to(torch.long)
        self._postprocess_cursor = 0
        return None

    def postprocess(self, hidden_states_slice: torch.Tensor, multimodal_outputs: Any = None, **req_infos: Any) -> dict:
        """Publish this step's codes row for one request (incremental, not cumulative)."""
        codes_full = getattr(self, "_last_audio_codes", None)
        if codes_full is None:
            return {}
        cursor = int(getattr(self, "_postprocess_cursor", 0))
        if cursor >= int(codes_full.shape[0]):
            self._postprocess_cursor = 0
            return {}
        slice_codes = codes_full[cursor : cursor + 1]
        self._postprocess_cursor = cursor + 1
        return {"codes": {"audio": slice_codes.to(torch.int32)}}

    def make_omni_output(self, model_outputs: Any, **kwargs: Any) -> OmniOutput:
        """Wrap decoder outputs into the OmniOutput contract.

        The runner threads per-request ``codes.audio`` (published by
        ``postprocess``) into ``model_intermediate_buffer`` in batch order;
        assemble them into the multimodal payload Stage 1 consumes.
        """
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs

        info_dicts = kwargs.get("model_intermediate_buffer")
        if info_dicts is None:
            info_dicts = kwargs.get("runtime_additional_information")
        if info_dicts is None:
            info_dicts = []

        audio_codes_list: list[torch.Tensor] = []
        any_nonempty = False
        for info in info_dicts:
            ac: torch.Tensor | None = None
            if isinstance(info, dict):
                codes_field = info.get("codes")
                if isinstance(codes_field, dict):
                    ac = codes_field.get("audio")
                else:
                    ac = info.get("audio_codes")
            if isinstance(ac, torch.Tensor) and ac.numel() > 0:
                audio_codes_list.append(ac)
                any_nonempty = True
            else:
                # keep list length == batch size so per-request indexing never
                # falls back to element[0] for higher slots
                audio_codes_list.append(torch.empty(0, dtype=torch.long))

        if any_nonempty:
            return OmniOutput(
                text_hidden_states=hidden,
                multimodal_outputs={"codes": {"audio": audio_codes_list}},
            )
        return OmniOutput(text_hidden_states=hidden, multimodal_outputs={})

    # ------------------------------------------------------------------- load
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load the converted safetensors (keys mirror this module tree 1:1).

        L0 completeness contract: every tensor in the converted manifest must be
        consumed (nothing skipped), and every model parameter must be initialized
        (nothing missing). Stats are logged for the L0 check; shape mismatches
        raise immediately.
        """
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        skipped: list[str] = []
        for name, w in weights:
            target = params.get(name)
            if target is None:
                skipped.append(name)
                continue
            if tuple(target.shape) != tuple(w.shape):
                raise ValueError(
                    f"shape mismatch for {name}: checkpoint {tuple(w.shape)} vs model {tuple(target.shape)}"
                )
            target.data.copy_(w.to(target.dtype))
            loaded.add(name)
        missing = sorted(set(params) - loaded)
        print(
            f"[Zonos2] load_weights: consumed={len(loaded)} skipped={len(skipped)} missing={len(missing)}",
            flush=True,
        )
        if skipped:
            print(f"[Zonos2] skipped (unexpected) keys: {skipped[:20]}", flush=True)
        if missing:
            print(f"[Zonos2] missing (uninitialized) params: {missing[:20]}", flush=True)
        return loaded
