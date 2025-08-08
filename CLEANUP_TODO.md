## REMB Cleanup Checklist

Cross-file goal: remove legacy shims, dead code, overly verbose comments, and redundant logic. Keep concise, high-signal docs and strict fail-fast behavior.

### Completed

- [x] `engine/nnue_inference.cpp`: remove legacy fallback; strict CSV logits+density output
- [x] `evaluate.py`:
  - [x] NNUE: require CSV logits+density; drop single-float legacy parsing
  - [x] EtinyNet: strict RESULT_i parsing → [1, C] tensor
  - [x] remove dead `compiled_parity_check`
  - [x] fix return type annotation `Dict[str, any]` → `Dict[str, Any]`
- [x] `serialize.py`:
  - [x] enforce required metadata keys; no silent defaults
  - [x] require `visual_threshold`
- [x] `nnue.py`: remove debug prints in `_calculate_conv_params`
- [x] `tests/test_compiled_parity.py`: replace xfail with assert
- [x] `engine/include/nnue_engine.h`: trim low-signal comments
- [x] `engine/src/nnue_engine.cpp`: trim non-actionable `cerr` and verbose comments in load paths

### Pending (apply in order)

- [x] `train.py` (light trim)
  - [x] Remove redundant comments that restate code
  - [x] Keep concise error messages and early logs

- [x] `training_utils.py`
  - [x] Remove unused imports/branches and comments
  - [x] Keep explicit exceptions; minimal messages

- [x] `scripts/compare_engine_speed.py`
  - [x] Reduce banner and section comments; keep usage and key steps
  - [x] Consider reusing build command flags from container/train for consistency (no behavior change)

- [x] `tests/conftest.py`
  - [x] Drop unused stubs/warning filters (verify no test relies on them)
  - [x] Keep only filters that trigger in current suite

- [x] `engine/include/nnue_engine.h` and `engine/src/nnue_engine.cpp` (second pass)
  - [x] Implement safe fallback for `>64` channels feature extraction (linear scan)
  - [x] Ensure default constants don’t conflict with serialized sizes; notes remain near getters

- [x] `container_setup.sh`
  - [x] Removed nonessential tools (`tree`, `htop`) to keep footprint lean

- [x] `requirements_dev.txt`
  - [x] Remove dev deps no longer used after cleanup

- [x] Repo-wide
  - [x] Grep for TODO/legacy/xfail/Conv params; cleaned where applicable
  - [x] Ensure fast test suite stable; parity quick tests pass

### Test gate

- After each logical group of edits, run:
  - `cmake --build engine/build --target nnue_inference etinynet_inference -j4`
  - `pytest -q -k compiled_parity --timeout=20`
  - `pytest -q -k 'not slow' --timeout=20`


