# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ZONOS2 Stage-1: DAC 44.1kHz code2wav decoder.

M1 skeleton: the full vLLM-runtime contract is implemented (engine hooks,
per-request split, OmniOutput assembly, left-context trim), but the DAC
network itself is a placeholder that emits a correctly-shaped zero waveform.
The real DAC decoder (descript-audio-codec 44.1kHz, hop_length=512,
min_decode_chunk=16 / overlap=4 streaming semantics) lands in the M4
milestone; ``load_weights`` will then consume the codec checkpoint.

Registered as ``Zonos2Code2WavForConditionalGeneration`` (Stage-1 arch
override in the pipeline config).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from vllm.config import VllmConfig

from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.model_executor.models.zonos2.configuration_zonos2 import Zonos2Config


class Zonos2Code2WavForConditionalGeneration(nn.Module):
    """Stage-1 codec decoder for ZONOS2 (9 codebooks, 44.1kHz output)."""

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        hf_config = vllm_config.model_config.hf_config
        if isinstance(hf_config, Zonos2Config):
            self.config = hf_config
        else:
            self.config = Zonos2Config(**hf_config.to_dict())
        self._model_path: str | None = vllm_config.model_config.model

        self.sample_rate: int = 44100
        self.num_codebooks: int = int(self.config.n_codebooks)  # 9
        self.num_real_codes: int = int(self.config.codebook_size)  # 1024
        # Official ZONOS2 streaming DAC: hop_length=512 (86.13 fps at 44.1kHz).
        self.hop_length: int = 512

        # Engine-runner hooks (Stage 1 has no token sampling).
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        # Real DAC modules are registered here in M4; dummy mode only needs
        # the contract-compliant placeholder path below.
        self._codec_loaded = False

    # ------------------------------------------------------------- engine hooks
    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        """Stage 1 ignores embeddings; vLLM's runner still needs a stable shape."""
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    def load_weights(self, weights: Any) -> set[str]:
        """Engine path: codec weights do not come from the Stage-0 safetensors
        iterator, so consume and ignore it. The real codec load (descript DAC
        44.1kHz) arrives with the M4 decoder implementation."""
        try:
            for _ in weights:
                pass
        except TypeError:
            pass
        return {name for name, _ in self.named_parameters()}

    # ------------------------------------------------------------------ decode
    @torch.no_grad()
    def _decode_request(self, codes_qt: torch.Tensor) -> torch.Tensor:
        """Decode one request's ``[num_codebooks, T]`` codes to 1-D float32 PCM.

        M1 placeholder: returns zeros of the correct length (T * hop_length).
        """
        num_frames = int(codes_qt.shape[-1])
        return torch.zeros(num_frames * self.hop_length, dtype=torch.float32)

    # ---------------------------------------------------- vLLM runtime forward
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        """Engine runtime forward: flat codebook-major codes -> OmniOutput audio."""
        sr_tensor = torch.tensor(self.sample_rate, dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        ids = input_ids.reshape(-1).to(dtype=torch.long)
        seq_token_counts = kwargs.get("seq_token_counts")
        if seq_token_counts is None:
            counts = [int(ids.numel())]
        else:
            counts = [int(c) for c in seq_token_counts]

        # Split the flat payload back into per-request code tensors.
        wavs: list[torch.Tensor] = []
        offset = 0
        for count in counts:
            chunk = ids[offset : offset + count]
            offset += count
            if chunk.numel() == 0:
                wavs.append(empty)
                continue
            num_frames = chunk.numel() // self.num_codebooks
            if num_frames == 0:
                wavs.append(empty)
                continue
            # codebook-major flat -> [Q, T]
            codes_qt = chunk[: num_frames * self.num_codebooks].reshape(self.num_codebooks, num_frames)
            codes_qt = codes_qt.clamp_(min=0, max=self.num_real_codes - 1)
            wavs.append(self._decode_request(codes_qt))

        if not wavs:
            wavs = [empty]
        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={
                "model_outputs": wavs,
                "sr": [sr_tensor] * len(wavs),
            },
        )
