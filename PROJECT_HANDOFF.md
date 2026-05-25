# Project Handoff

Last updated: 2026-05-23

## Project Context

This repo is the `xiaozhi-esp32-server` workspace used to run classroom lab assistant flows.

Main active experiments:

- `exp1_AgNPs_synthesis`
- `exp2_UV_Vis_analysis`

The current operational focus is `exp2_UV_Vis_analysis` with UV-Vis hardware connected on `COM7`.

The active LAN endpoint should remain:

- WebSocket: `ws://192.168.1.100:8000/xiaozhi/v1/`
- HTTP: `http://192.168.1.100:8003/`

The user needs TUN/v2rayN enabled. Avoid binding `xiaozhi` to `0.0.0.0:8003` because it can conflict with `sing-box`. Binding to `192.168.1.100:8003` is the intended setup.

## Important Current Status

### `intentHandler.py`

`main/xiaozhi-server/core/handle/intentHandler.py` was temporarily corrupted during iterative edits because Chinese strings were written back with bad encoding. It has been restored to a clean, compilable git version.

Validation result:

```powershell
python -m py_compile main/xiaozhi-server/core/handle/intentHandler.py
```

Current result: passes after recovery.

The corrupted copy was saved here:

```text
main/xiaozhi-server/core/handle/manual_backups/intentHandler_before_recover_20260523_172425.py
```

Do not restore from that backup. Use it only as historical reference.

### `textUtils.py`

`main/xiaozhi-server/core/utils/textUtils.py` has a targeted speech normalization fix:

- `shen比位`
- `深比位`
- `生比位`
- `申比位`
- `身比位`
- `伸比位`
- `深笔位`
- `参笔位`
- `参北位`

These normalize to:

- `参比位`

This was added because ASR sometimes read `参比位` as `shen比位`.

## `exp2` Flow Problems Seen

### 1. Repeated shared dark/air prep

Observed behavior:

- Shared dark current and air energy correction were already complete.
- Files existed in:
  - `lab_runs/exp2_UV_Vis_analysis/data/uv_data_common`
- But the assistant repeatedly said:
  - `已复用共享暗电流和空气能量校正，直接进入下一步。`
  - or incorrectly returned to shared dark/air guidance.

Fix direction already explored:

- If `step_3_uv_vis_shared_dark_air_prep` is complete and cached shared calibration exists, `继续下一步` should call graph advancement, not re-run or re-speak shared calibration.
- Legacy `pure_water_blank/shared_blank` path should be disabled unless the active YAML actually contains the pure-water blank step.

### 2. Wrong spoken template after graph advanced

Observed behavior:

- Graph moved to:
  - `step_3_uv_vis_sample1-4_load_cuvette`
- Assistant still spoke the previous shared dark/air instruction.

Fix direction already explored:

- `_compose_uvvis_step_reply(...)` should prefer exact `step_id` over keyword matching.
- Exact current IDs:
  - `step_3_uv_vis_sample1-4_load_cuvette`
  - `step_3_uv_vis_sample1-4_record_data`

### 3. Too many confirmations before spectra scan

Observed behavior from `94_a9_90_28_eb_cc.log`:

1. User: `现在是第一组，我已经放好了。`
2. Assistant repeated load instruction.
3. User: `做好了。`
4. Assistant advanced to record-data step but spoke long internal instruction.
5. User: `开始扫描。`
6. Assistant asked: `这是第几组的样品扫描？`
7. User: `第一组开始扫描。`

Desired behavior:

- User says `现在是第一组，我已经放好了。`
  - System records group `1`.
  - System marks load-cuvette step complete.
  - System advances to spectra record step.
- User then says `开始扫描。`
  - System reuses group `1`.
  - System starts `uvvis_measure_spectra` without asking group again.

This is the next code task.

## UV-Vis Process/Path Rules

For `exp2`, all shared UV-Vis files should write under:

```text
lab_runs/exp2_UV_Vis_analysis/data/uv_data_common
```

Do not write current `exp2` dark/air files under:

```text
lab_runs/exp1_AgNPs_synthesis/data/uv_data_common
```

When switching `exp1 <-> exp2`, restart the full UV-Vis chain:

1. `spectrometer_server.py`
2. `uvvis_http_wrapper.py`
3. `uvvis_mcp_server.py`
4. `xiaozhi` app

Expected UV-Vis ports:

- `8765`: spectrometer backend
- `8766`: UV-Vis MCP server

## Data Already Produced/Fixes Applied

Device:

```text
94_a9_90_28_eb_cc
```

Known data directories:

```text
lab_runs/exp2_UV_Vis_analysis/data/uv_data_common/94_a9_90_28_eb_cc/1/uvvis_measure_spectra
lab_runs/exp2_UV_Vis_analysis/data/uv_data_common/94_a9_90_28_eb_cc/2/uvvis_measure_spectra
lab_runs/exp2_UV_Vis_analysis/data/uv_data_common/94_a9_90_28_eb_cc/1/uvvis_measure_kinetic
lab_runs/exp2_UV_Vis_analysis/data/uv_data_common/94_a9_90_28_eb_cc/2/uvvis_measure_kinetic
```

Manual data/report actions already done earlier:

- Group 1 max absorbance spectra measured.
- Group 2 max absorbance spectra measured.
- Group 2 spectra were remapped because cuvettes were placed in reverse order `5 4 3 2 1`.
- Group-level YAML/PDF reports were generated for groups `1` and `2`.
- Report path output was changed so YAML/PDF paths should be relative.
- PDF乱码 in observation records was corrected for the known `94_a9_90_28_eb_cc` reports.

## Current Working Tree Notes

At the last check, these files had meaningful pending changes:

- `main/xiaozhi-server/core/utils/textUtils.py`
- Various UV-Vis startup/path/restart related files from earlier work:
  - `main/xiaozhi-server/app.py`
  - `main/xiaozhi-server/connection.py` related code may have been touched
  - `main/xiaozhi-server/scripts/ensure_uvvis_http_mcp.py`
  - `main/xiaozhi-server/core/providers/tools/server_mcp/uvvis_scan_rule.py`
  - `main/xiaozhi-server/mcp_server_settings.json`

Also seen untracked:

```text
intentHandler.cpython-311.py
intentHandler.cpython-313.py
main/xiaozhi-server/core/handle/manual_backups/
```

The two `intentHandler.cpython-*` files are failed decompilation artifacts and should not be used by runtime.

## Recommended Next Session Procedure

1. Confirm `intentHandler.py` still compiles.

```powershell
python -m py_compile main/xiaozhi-server/core/handle/intentHandler.py
```

2. Re-apply only the scoped UX fix for `exp2` sample loading and spectra start:

- Reuse `conn.experiment_current_group_number` in spectra measurement.
- Let `第一组 + 放好了` complete load step.
- Let `开始扫描` reuse the remembered group.
- Keep live step IDs on `sample1-4`, not obsolete `sample1-5`.

3. Compile:

```powershell
python -m py_compile main/xiaozhi-server/core/handle/intentHandler.py
python -m py_compile main/xiaozhi-server/core/utils/textUtils.py
```

4. Restart `xiaozhi`.

5. Test with device `94:a9:90:28:eb:cc`:

```text
继续刚才的实验
现在是第一组，我已经放好了
开始扫描
```

Expected:

- No repeated long load instruction after the second utterance.
- No repeated group question after `开始扫描`.
- `uvvis_measure_spectra` starts and writes into:

```text
lab_runs/exp2_UV_Vis_analysis/data/uv_data_common/94_a9_90_28_eb_cc/1/uvvis_measure_spectra
```

