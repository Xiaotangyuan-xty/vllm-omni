# Reproduction — `f43b420323bcfcbe89c3a49fa9a0e3303063a33c`

Run from the repository root. Use Python 3.12.13, official vLLM 0.34.0 (`62d4bc3c7`), Torch 2.14.0+cu130, FlashInfer 0.7.3 and Transformers 5.22.0. The focused/GPU environment is separate from the expanded repository-UT environment. The latter adds the declared torchmetrics, Ray, telemetry instrumentation, mistune, hyperpyyaml and openpyxl dependencies without replacing the runtime.

Provision the pinned converted ZONOS2 safetensors, local FP32 DAC checkpoint, speaker snapshot and owned TN cache as documented in `tests/e2e/zonos2/README.md`. Set `ZONOS2_TEST_MODEL_PATH`, `VLLM_ZONOS2_DAC_PATH`, `VLLM_ZONOS2_SPEAKER_PATH`, `VLLM_ZONOS2_TN_CACHE_DIR` and an owned `ZONOS2_TEST_OUTPUT_DIR`. Inference uses `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. Router replay is off for E2E.

## Native handoff and scheduling

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/model_executor/models/zonos2/test_sync_handoff.py -m 'core_model and cpu' --run-level core_model
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/diffusion/test_sampler.py tests/model_executor/models/zonos2/test_sampler.py -m 'core_model and cpu' --run-level core_model
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/worker_v2/test_omni_data_plane.py tests/core/sched/test_omni_scheduling_coordinator.py tests/core/sched/test_omni_scheduler_ready_inbox.py tests/distributed/omni_connectors/test_chunk_transfer_adapter.py -m 'core_model and cpu' --run-level core_model
```

The first two new regressions fail at `8215923f7`. The native callback test retains raw frames, checks terminal metadata and traverses the real WAITING_FOR_INPUT/WAITING transition. The import regression must collect both sampler modules in the same process.

## Final GPU acceptance

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_ZONOS2_ROUTER_REPLAY=0 python -m pytest -q tests/model_executor/models/zonos2 tests/e2e/zonos2 -m 'cuda and cards_1 and (core_model or advanced_model or full_model)' --run-level full_model
```

This selects two component tests and four E2E cases: B1 chunked offline, B1 synchronous offline, B4 isolation, B1 HTTP WAV/raw/SSE. Sync sets both `async_chunk` and `codec_streaming` false and requires exactly one DAC call with independently reconstructed PCM. All GPU jobs bind the single authorized A40 GPU0; other processes are preserved. Logs/controller hashes bind evidence to the tested source.

## Repository UT attempts

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -sv tests/ -m 'core_model and cpu'
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -sv tests/ -m 'core_model and cpu' --continue-on-collection-errors -n 4 --dist=loadfile
```

Run both head and merged main `f175351ce` in the same expanded UT environment. The first command stops on three collection errors. Two CUDA IPC modules have import-time device guards under hidden CUDA; the CosyVoice3 GPU expansion resolves an uncached snapshot during import despite the CPU selection. The second command preserves those errors and executes successfully collected CPU tests. It must not be described as a green complete CI pass. Its exact results, additional failures and base/head comparison are recorded in the evidence packet.

The NPU pipeline's CPU-only command is:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -sv tests/platforms/npu/test_qwen3_omni_model_state.py tests/platforms/npu/test_npu_generation_v2.py tests/platforms/npu/test_code2wav_graph_wrapper.py -m 'core_model and cpu' --run-level core_model
```

The code2wav wrapper executes on CPU. The other two modules use `pytest.importorskip('vllm_ascend')` and skip in this CUDA environment. Actual ROCm/Ascend hardware jobs were not executed. The pipeline inventory preserves every selected job's source YAML hash, commands, hardware and shard declarations; it is an inventory, not execution evidence.

The standard branch-diff `pre_commit` run includes mypy and Buildkite schema. No gates were bypassed. The fixed-C1 output comparison retains all raw codes and NumPy/WAV artifacts on the test server; the packet includes exact shapes and equality results without shipping model weights.
