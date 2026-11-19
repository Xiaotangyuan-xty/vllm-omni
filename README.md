# ZONOS2 MRv2 validation evidence

Evidence for [vllm-omni PR #8436](https://github.com/vllm-project/vllm-omni/pull/8436). This branch contains validation artifacts only; it is not a code-change branch or a merge request. GitHub comment attachment uploads stalled, so the sanitized packet is hosted in the author fork.

## Revision binding

CI retry head: `d84ded4eac86ca009b1cea9d93f5a7be2f9132f5`. This is an empty, DCO-signed commit on `3964b5fa6f99a9ed6d07af43d97581fe804092b8`; it retries the manually cancelled Read the Docs build 36012220. Both have the exact same source tree as tested `f43b420323bcfcbe89c3a49fa9a0e3303063a33c`: `1f07f6c6989d5e2c9f7b560bb0a7b61a31d6f660`. Tests were executed at f43b42032; they were not rerun for the empty retry commit.

- [Results and coverage boundaries](EVIDENCE.md)
- [Commands and reproduction](reproduce.md)
- [Whole-repository CPU comparison](full_comparison.json)
- [Output-equivalence records](output_comparison.json)
- [Runtime](runtime_environment.json) and [expanded UT environment](ut_environment.json)
- [Pinned CUDA/AMD/NPU unit-job inventory](ci_unit_matrix.json) (inventory, not execution)
- [Download logs and records](zonos2-mrv2-validation.zip); verify against [SHA256SUMS](SHA256SUMS). The archive includes an individual-file SHA256 manifest.

The CPU continuation is non-green on both revisions; the failing/error node-ID sets match. Focused CPU, transport and CUDA functional results are listed separately. No hosted Buildkite, ROCm/Ascend hardware qualification or V2 performance claim is made. Raw codes/audio/model weights are excluded; the output-comparison records are included.

Operator paths, private IPs and token-like values are sanitized. AI assistance: Codex helped collect, compare and package these records.
