# Next Codex Goals

Last updated: 2026-05-23

## Immediate Priority

1. Restore a stable `exp2` voice flow for UV-Vis sample loading and spectra scan.
   - Current pain point from the last live run:
     - User says: `现在是第一组，我已经放好了。`
     - System should record group `1` and mark `step_3_uv_vis_sample1-4_load_cuvette` complete.
     - Next user says: `开始扫描。`
     - System should reuse group `1` and call `uvvis_measure_spectra` directly.
   - It should not ask again: `这是第几组的样品扫描？`
   - It should not wait for Codex model generation when the intent is locally obvious.

2. Re-apply the targeted `intentHandler.py` UX fix on top of a clean source file.
   - `main/xiaozhi-server/core/handle/intentHandler.py` was restored to a clean, compilable git version after a bad encoding/writeback corrupted strings.
   - The corrupted working copy was backed up at:
     - `main/xiaozhi-server/core/handle/manual_backups/intentHandler_before_recover_20260523_172425.py`
   - Do not use that backup as source code directly. It contains many broken strings and syntax issues.

3. Restart `xiaozhi` after any `intentHandler.py` change and verify with `94:a9:90:28:eb:cc`.
   - Expected WebSocket:
     - `ws://192.168.1.100:8000/xiaozhi/v1/`
   - Expected HTTP:
     - `http://192.168.1.100:8003/`
   - Keep TUN/v2rayN compatible by binding `xiaozhi` to `192.168.1.100`, not `0.0.0.0`.

## Current Known State

- `intentHandler.py` compiles after recovery.
- `textUtils.py` contains a narrow ASR normalization fix for `参比位` being heard as `shen比位` or similar variants.
- `exp2` UV-Vis shared dark current and air baseline should write under:
  - `lab_runs/exp2_UV_Vis_analysis/data/uv_data_common`
- Do not let UV-Vis shared data write under:
  - `lab_runs/exp1_AgNPs_synthesis/data/uv_data_common`
- Previous fix direction:
  - `uvvis_prepare_dark_current` should receive/use explicit current-experiment shared output dir.
  - Switching `exp1 <-> exp2` should restart the UV-Vis MCP/wrapper/spectrometer chain.

## Specific Code Work To Do Next

### `intentHandler.py`

Re-implement only these scoped behaviors:

1. Use current group number when starting spectra.
   - In `_handle_uvvis_spectra_measurement(...)`, if current utterance lacks explicit group number, reuse `conn.experiment_current_group_number` when it is a valid integer.
   - Only ask `这是第几组的样品扫描？` when neither explicit nor current group is available.

2. Handle load-cuvette completion with group memory.
   - For `step_3_uv_vis_sample1-4_load_cuvette`, accept utterances like:
     - `现在是第一组，我已经放好了`
     - `第一组，做好了`
   - Behavior:
     - Sync group number to graph/session state.
     - Mark load-cuvette step complete.
     - Advance to `step_3_uv_vis_sample1-4_record_data`.
     - Reply concisely, e.g. `好，已经记成第1组。现在可以直接说开始扫描。`

3. Direct scan when utterance includes both ready and start.
   - For utterances like:
     - `现在是第一组，已经放好了，开始扫描`
   - Behavior:
     - Complete load step.
     - Reuse group `1`.
     - Call `uvvis_measure_spectra` directly.

4. Keep current `exp2` step IDs aligned with YAML.
   - Current YAML uses:
     - `step_3_uv_vis_sample1-4_load_cuvette`
     - `step_3_uv_vis_sample1-4_record_data`
   - Old code may still mention:
     - `step_3_uv_vis_sample1-5_load_cuvette`
     - `step_3_uv_vis_sample1-5_record_data`
   - Preserve backward compatibility if useful, but the live graph must not redirect to nonexistent old step IDs.

### `textUtils.py`

Already contains the `参比位` normalization change. Verify it still compiles after any future edits:

```powershell
python -m py_compile main/xiaozhi-server/core/utils/textUtils.py
```

## Validation Checklist

Run these before restarting service:

```powershell
python -m py_compile main/xiaozhi-server/core/handle/intentHandler.py
python -m py_compile main/xiaozhi-server/core/utils/textUtils.py
```

Then restart `xiaozhi` and test this exact voice sequence:

1. `继续刚才的实验`
   - Should speak current sample loading step, not shared dark/air prep.
2. `现在是第一组，我已经放好了`
   - Should not repeat the full load instruction.
   - Should mark load step complete and remember group `1`.
3. `开始扫描`
   - Should not ask for group again.
   - Should call `uvvis_measure_spectra`.
   - Data should land in:
     - `lab_runs/exp2_UV_Vis_analysis/data/uv_data_common/<device_id>/1/uvvis_measure_spectra`

## Data/Report Notes

- For device `94_a9_90_28_eb_cc`, previous manual data fixes were applied:
  - Group 1 and group 2 max absorbance spectra were generated.
  - Group 2 spectra were remapped from reversed placement `5 4 3 2 1`.
  - Group-level reports were generated under:
    - `lab_runs/exp2_UV_Vis_analysis/data/uv_data_common/94_a9_90_28_eb_cc/1`
    - `lab_runs/exp2_UV_Vis_analysis/data/uv_data_common/94_a9_90_28_eb_cc/2`
- Report YAML/PDF paths should be relative paths, not absolute paths.

## Cleanup Notes

- There are untracked decompile artifacts from a failed recovery attempt:
  - `intentHandler.cpython-311.py`
  - `intentHandler.cpython-313.py`
- These are not runtime files. Remove them after confirming they are not needed.

