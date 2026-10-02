# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ZONOS2 registration and dummy-weight speech serving skeleton.

The P2-01 processor constructs canonical normalized text frames. Serving still
requires an explicitly dummy-loaded two-stage pipeline until the request-local
sampler and real DAC milestones land; its returned waveform remains silent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from vllm.utils.async_utils import make_async

from vllm_omni.entrypoints.openai.tts_adapters import register_tts_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest, apply_max_new_tokens

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

_SKELETON_FIELDS = frozenset(
    {"input", "model", "voice", "response_format", "speed", "stream", "stream_format", "max_new_tokens", "sample_rate"}
)


@register_tts_adapter
class Zonos2Adapter(ARTTSAdapter):
    name = "zonos2"
    stage_keys = frozenset({"zonos2"})
    model_archs = frozenset({"Zonos2ForConditionalGeneration", "Zonos2TalkerForConditionalGeneration"})
    supported_output_sample_rates = frozenset({44100})
    max_new_tokens_max = 1024

    def __init__(self, ctx):
        super().__init__(ctx)
        self._processor = None

    def _get_processor(self):
        from vllm_omni.model_executor.models.zonos2.zonos2_processor import Zonos2Processor

        if self._processor is None:
            self._processor = Zonos2Processor(self.ctx.engine_client.model_config.hf_config)
        return self._processor

    def _is_dummy_pipeline(self) -> bool:
        stages = getattr(self.ctx.engine_client, "stage_configs", ()) or ()
        if len(stages) != 2:
            return False
        for stage in stages:
            load_config = getattr(stage, "load_config", None)
            if load_config is not None:
                load_format = getattr(load_config, "load_format", None)
            else:
                # Legacy resolved deployment: CLI overrides are already folded
                # into this effective engine_args mapping.
                load_format = getattr(getattr(stage, "engine_args", None), "load_format", None)
            if load_format != "dummy":
                return False
        return True

    def _load_supported_speakers(self) -> set[str]:
        return {"default"}

    def validate(self, request: OpenAICreateSpeechRequest) -> str | None:
        if not self._is_dummy_pipeline():
            return (
                "ZONOS2 speech serving currently supports dummy-weight skeleton validation only; "
                "both stages must use load_format=dummy"
            )
        if not request.input or not request.input.strip():
            return "Input text cannot be empty"
        if request.voice is not None and request.voice.lower() != "default":
            return "ZONOS2 dummy validation only supports voice='default'"
        if request.speed not in (None, 1.0):
            return "ZONOS2 dummy validation does not support speed adjustments"
        if request.is_streaming():
            return "ZONOS2 dummy validation only supports non-streaming requests"
        if request.response_format not in ("wav", "pcm"):
            return "ZONOS2 dummy validation only supports response_format='wav' or 'pcm'"
        for field in sorted(request.model_fields_set - _SKELETON_FIELDS):
            if getattr(request, field) is not None:
                return f"ZONOS2 dummy validation does not support '{field}'"
        if request.sample_rate not in (None, 44100):
            return "ZONOS2 dummy validation uses sample_rate=44100"
        if request.max_new_tokens is not None and request.max_new_tokens > self.max_new_tokens_max:
            return f"max_new_tokens cannot exceed {self.max_new_tokens_max} in dummy validation"
        return None

    def apply_sampling_overrides(
        self,
        sampling_params_list: list,
        request: OpenAICreateSpeechRequest,
        prompt: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> list:
        return apply_max_new_tokens(sampling_params_list, request)

    async def build(
        self,
        request: OpenAICreateSpeechRequest,
        sampling_params_list: list,
        has_inline_ref_audio: bool,
    ) -> PreparedRequest:
        error = self.validate(request)
        if error is not None:
            raise ValueError(error)
        build_prompt = make_async(
            self._get_processor().build_prompt, executor=getattr(self.ctx.server, "_tts_executor", None)
        )
        return PreparedRequest(prompt=await build_prompt(request.input), model_type=self.name)
