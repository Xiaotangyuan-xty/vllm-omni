# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage-input processor for ZONOS2: Talker -> DAC decoder.

M1 skeleton ships the sync adapter only (deploy yaml sets ``async_chunk:
false``): after Stage 0 finishes, all audio frames are collected, de-sheared
(delay pattern reverted), clipped to real DAC codes and flattened into a
codebook-major ``OmniTokensPrompt`` for Stage 1.

ZONOS2 layout: 9 codebooks x 1024 real codes per frame; eoa=1024 and pad=1025
are stream specials that the DAC decoder never sees.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

__all__ = ["talker2dac"]

logger = init_logger(__name__)

# ZONOS2 constants (params.json: n_codebooks=9, codebook_size=1024,
# eoa_id=1024, audio_pad_id=1025). Kept local so this module stays free of
# model-package imports during stage-input-processor discovery.
_NUM_CODEBOOKS = 9
_NUM_REAL_CODES = 1024  # codes in [0, 1023] are real DAC codes


def _revert_delay_pattern(audio_codes_qt: torch.Tensor) -> torch.Tensor:
    """Reverse the shear/delay layout: codebook ``j`` was shifted ``j`` frames.

    Input shape: ``[num_codebooks, seq_len + num_codebooks - 1]``.
    Output shape: ``[num_codebooks, seq_len]``.
    """
    if audio_codes_qt.ndim != 2:
        raise ValueError(f"_revert_delay_pattern expects [Q, T] input; got {tuple(audio_codes_qt.shape)}")
    q, t = audio_codes_qt.shape
    if t < q:
        # Not enough frames to revert delay pattern; return as-is.
        return audio_codes_qt
    seq_len = t - q + 1
    out_l = []
    for i in range(q):
        out_l.append(audio_codes_qt[i : i + 1, i : seq_len + i])
    return torch.cat(out_l, dim=0)


def talker2dac(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    """Sync: collect all talker codes, then pass to the DAC decoder at once."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    dac_inputs: list[OmniTokensPrompt] = []
    for talker_output in source_outputs:
        if not talker_output.finished:
            continue
        output = talker_output.outputs[0]
        mm = output.multimodal_output
        mm_codes = (mm or {}).get("codes", {})

        audio_codes = mm_codes.get("audio")
        if audio_codes is None or not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
            # Nothing to decode for this request; emit an empty payload so the
            # downstream stage can still close the response.
            dac_inputs.append(
                OmniTokensPrompt(
                    prompt_token_ids=[],
                    multi_modal_data=None,
                    mm_processor_kwargs=None,
                    additional_information=None,
                )
            )
            continue

        audio_codes = audio_codes.to(torch.long)
        if audio_codes.ndim == 1:
            if audio_codes.numel() % _NUM_CODEBOOKS != 0:
                raise ValueError(
                    f"flat audio_codes length {audio_codes.numel()} not divisible by num_codebooks={_NUM_CODEBOOKS}"
                )
            audio_codes = audio_codes.reshape(-1, _NUM_CODEBOOKS)
        if audio_codes.ndim != 2:
            raise ValueError(f"audio_codes must be 1D or 2D; got shape {tuple(audio_codes.shape)}")

        # Stage-0 frames arrive in shear/delay layout; revert, drop stream
        # specials (eoa/pad) and trim the edge frames the DAC never consumes.
        codes_qt = audio_codes.transpose(0, 1).contiguous().cpu()
        codes_qt = _revert_delay_pattern(codes_qt)
        codes_qt = codes_qt.clamp_(min=0, max=_NUM_REAL_CODES - 1)
        if codes_qt.shape[-1] >= 3:
            codes_qt = codes_qt[:, 1:-1]
        # DAC decoder expects codebook-major flat: [Q * num_frames].
        codec_codes = codes_qt.reshape(-1).tolist()

        dac_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=None,
            )
        )
    return dac_inputs
