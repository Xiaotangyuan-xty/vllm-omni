# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""P1 CPU contracts: registered dummy-only ZONOS2 speech skeleton."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from vllm import SamplingParams

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters import all_tts_stage_keys, detect_tts_model_type, resolve_adapter
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.zonos2 import Zonos2Adapter

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _adapter(formats=("dummy", "dummy"), *, legacy=False):
    stages = []
    for stage_id, (stage_name, fmt) in enumerate(zip(("zonos2", "dac_decoder"), formats)):
        args = SimpleNamespace(model_stage=stage_name, model_arch=None, worker_type="ar", load_format=fmt)
        if legacy:
            stage = SimpleNamespace(engine_args=args)
        else:
            from vllm_omni.config.stage_config import StagePipelineConfig

            stage = SimpleNamespace(
                stage_pipeline_config=StagePipelineConfig(stage_id=stage_id, model_stage=stage_name),
                model_config=SimpleNamespace(model_arch=None),
                load_config=SimpleNamespace(load_format=fmt),
                worker_type="ar",
            )
        stages.append(stage)
    engine = SimpleNamespace(stage_configs=stages)
    return Zonos2Adapter(SpeechServingContext(server=SimpleNamespace(), engine_client=engine))


def test_zonos2_registration_and_talker_detection():
    assert resolve_adapter("zonos2") is Zonos2Adapter
    assert "zonos2" in all_tts_stage_keys()
    for arch in (None, "Zonos2ForConditionalGeneration", "Zonos2TalkerForConditionalGeneration"):
        assert detect_tts_model_type("zonos2", arch) == "zonos2"
    assert detect_tts_model_type(None, "Zonos2ForConditionalGeneration") == "zonos2"
    assert detect_tts_model_type("dac_decoder", "Zonos2Code2WavForConditionalGeneration") is None


@pytest.mark.parametrize("legacy", [False, True])
def test_serving_discovers_the_talker_and_accepts_dummy_request(legacy):
    adapter = _adapter(legacy=legacy)
    server = OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)
    server.engine_client = adapter.ctx.engine_client
    server._tts_stage = server._find_tts_stage()
    assert server._tts_stage is server.engine_client.stage_configs[0]
    assert server._detect_tts_model_type() == "zonos2"
    request = OpenAICreateSpeechRequest(input="Hello.", voice="default", max_new_tokens=16)
    assert adapter.validate(request) is None
    assert adapter.load_capabilities().supported_speakers == {"default"}


@pytest.mark.parametrize("formats", [("auto", "auto"), ("dummy", "auto"), ("auto", "dummy"), ()])
def test_skeleton_rejects_non_dummy_or_incomplete_deployment(formats):
    adapter = _adapter(formats)
    request = OpenAICreateSpeechRequest(input="Hello.")
    assert "dummy" in adapter.validate(request)
    with pytest.raises(ValueError, match="dummy"):
        asyncio.run(adapter.build(request, [], False))


@pytest.mark.parametrize("text", ["", " ", "\n\t"])
def test_rejects_empty_input(text):
    assert "empty" in _adapter().validate(OpenAICreateSpeechRequest(input=text)).lower()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"voice": "vivian"},
        {"speed": 1.2},
        {"language": "English"},
        {"ref_audio": "data:audio/wav;base64,AAAA"},
        {"speaker_embedding": [0.0]},
        {"instructions": "happy"},
        {"seed": 7},
        {"extra_params": {"temperature": 0.5}},
        {"stream": True},
        {"stream_format": "audio"},
        {"response_format": "flac"},
        {"sample_rate": 24000},
        {"max_new_tokens": 1025},
        {"word_timestamps": True},
    ],
)
def test_skeleton_rejects_features_outside_p1(kwargs):
    assert _adapter().validate(OpenAICreateSpeechRequest(input="Hello.", **kwargs)) is not None


def test_build_uses_the_p2_processor_without_a_hf_tokenizer():
    adapter = _adapter()
    calls = []

    def build_prompt(text):
        calls.append(text)
        return {"prompt_token_ids": [519, 2, 3], "additional_information": {"zonos2_frames": "sentinel"}}

    adapter._processor = SimpleNamespace(build_prompt=build_prompt)
    request = OpenAICreateSpeechRequest(input="中文 dummy.", max_new_tokens=16)
    prepared = asyncio.run(adapter.build(request, [], False))
    assert calls == ["中文 dummy."]
    assert prepared.model_type == "zonos2"
    assert prepared.prompt["additional_information"]["zonos2_frames"] == "sentinel"
    assert prepared.tts_params == {}
    assert request.input == "中文 dummy."


def test_max_tokens_override_does_not_mutate_defaults_or_codec_params():
    adapter = _adapter()
    params = [SamplingParams(max_tokens=64), SamplingParams(max_tokens=128)]
    request = OpenAICreateSpeechRequest(input="Hello.", max_new_tokens=16)
    result = adapter.apply_sampling_overrides(params, request)
    assert result[0].max_tokens == 16
    assert result[1].max_tokens == 128
    assert params[0].max_tokens == 64
