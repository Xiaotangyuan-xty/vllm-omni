# ZONOS2 synchronous MRv2 handoff — validation evidence

PR: https://github.com/vllm-project/vllm-omni/pull/8436

Head: `f43b420323bcfcbe89c3a49fa9a0e3303063a33c`. The runtime fix is `37104a48f`; the follow-up adds a test package marker. Previous PR head: `8215923f7`. Merged main comparison base: `f175351ce30b0e4e9a1dca04bfc3a731a746e4c6`.

## Revision binding

CI retry head: `d84ded4eac86ca009b1cea9d93f5a7be2f9132f5`. This is an empty, DCO-signed commit on `3964b5fa6f99a9ed6d07af43d97581fe804092b8`; it retries the manually cancelled Read the Docs build 36012220. Both have the exact same source tree as tested `f43b420323bcfcbe89c3a49fa9a0e3303063a33c`: `1f07f6c6989d5e2c9f7b560bb0a7b61a31d6f660`. Tests were executed at f43b42032; they were not rerun for the empty retry commit.

## Fix and reproduction

Two new regressions fail at the previous head: no full-payload producer is registered for sync, and the old inline converter expects codec data on the orchestrator boundary. Stage 0 now registers `talker2dac_full_payload`; stage 1 admits `talker2dac_token_only`. The native connector sends the complete raw codec sequence and terminal boundary, then the existing coordinator releases WAITING_FOR_INPUT. No readiness guard is removed. The obsolete inline converter is deleted.

Real-weight offline acceptance now covers chunked and sync transport. Sync uses `async_chunk: false` and `codec_streaming: false`, must deliver one DAC call and independently reconstruct the delivered waveform. Full collection also found the ZONOS2/diffusion `test_sampler.py` basename collision; adding the ZONOS2 test package marker fixes it. Both sampler modules execute together (68 passed).

## Same-head results

| Run | Result |
| --- | --- |
| Focused model/adapter/CI/inventory/protocol/assets and native V2 CPU | 783 passed, 12 deselected |
| Same selection without DAC/audiotools | 780 passed, 3 expected skips, 12 deselected |
| Native transport/coordinator/legacy adapter CPU | 279 passed |
| NPU pipeline CPU selection | 23 passed, 2 modules skipped without `vllm_ascend` |
| GPU components + real-weight acceptance | 6 passed, 232 deselected: 2 components + 4 E2E |
| Branch-diff pre-commit | Passed including mypy and Buildkite schema |

The groups overlap; their counts are not summed. The four E2E cases are B1 offline chunked, B1 offline sync, B4 isolation and B1 HTTP WAV/raw/SSE. B4 checks routing/reconstruction rather than speech quality. The default E2E router replay setting is off.

On the fixed C1 input/profile, both modes produce the V1 baseline's exact 357×9 codes. Default chunked output retains all 177,664 identical FP32 waveform samples (maximum absolute error 0). Sync has 177,664 samples and independent DAC reconstruction error 0. Sync/OLA waveforms are not equated across different chunk schedules. Raw tensors/audio remain in the operator's retained run directories; this packet carries the comparison records.

## Whole-repository CPU comparison

The exact unsharded command `python -m pytest -sv tests/ -m 'core_model and cpu'` was attempted with CUDA hidden on both head and base. Initial dependency gaps were filled in a separate UT environment. Both then stop on the same three collection errors: the two CUDA IPC weight-sharing modules require visible CUDA at import, and the CosyVoice3 expansion imports an uncached model snapshot under offline mode.

A diagnostic continuation executes the successfully collected CPU selection using four isolated pytest workers, one CPU thread per worker:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -sv tests/ -m 'core_model and cpu' --continue-on-collection-errors -n 4 --dist=loadfile
```

| Revision | Passed | Failed | Errors | Skipped | Xfailed |
| --- | ---: | ---: | ---: | ---: | ---: |
| PR head `f43b42032` | 42,191 | 423 | 199 | 348 | 1 |
| Main base `f175351ce` | 41,796 | 423 | 199 | 348 | 1 |

The failing node-ID sets are identical. There are no head-only error node IDs. Head has 395 more passing tests. Full logs, commands, source SHA, environment versions and node-ID comparisons are included in the archive. These local runs are not green complete CI results. Shared failures include unavailable accelerator/CPU kernels, optional model assets and test-environment dependencies; they were not edited as part of this ZONOS2 repair.

Deselection follows `core_model and cpu`; GPU/e2e/other levels are excluded by that expression. Skips include module-level optional dependencies and unsupported hardware/features. The three codec skips were separately verified in a deliberately codec-free environment; the two NPU module skips come from `pytest.importorskip('vllm_ascend')`. The canonical `-sv` whole-suite output does not print a complete per-case `-rs` skip-reason report. All failures and errors remain visible; no failing test is ignored or converted to a skip.

## Runtime, hardware and remaining coverage

Python 3.12.13, official vLLM 0.34.0 (`62d4bc3c7`), Torch 2.14.0+cu130, FlashInfer 0.7.3, Transformers 5.22.0. Expanded UT packages are recorded separately. The ignored local Omni version file still prints old version metadata; controller source hashes and import paths bind the actual source. The V1 baseline remains on its original vLLM 0.33.0 environment.

CUDA validation uses one physical A40 GPU0 alongside an existing process. Its existing PID remains present and memory returns to its initial 4,361 MiB after the owned workers exit. These are functional tests, not shared-GPU latency measurements.

`ci_unit_matrix.json` inventories pinned CUDA/AMD/NPU Unit/Simple and non-E2E core-model jobs, including commands, hardware, shard declarations and YAML hashes. It is not backend execution evidence. ROCm and Ascend hardware jobs, matching-head hosted Buildkite, and the broader RFC quality/performance matrix remain unverified here. Maintainers retain CI admission and merge decisions.

See `reproduce.md` for commands. Model weights, virtual environments, HF caches and reference audio are excluded. Operator paths, private addresses and token-like strings are sanitized; test results and assertions are retained.
