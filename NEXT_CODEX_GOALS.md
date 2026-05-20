# Next Codex Goals

This file is an execution checklist for the next Codex session. It assumes the repo root is:

`C:\Users\11979\Documents\GitHub\xiaozhi-esp32-server`

## Immediate Operating Rules

1. Communicate in Chinese unless the user asks otherwise.
2. Before changing files, inspect current contents; do not rely only on memory.
3. Use `rg` for searches.
4. Use `apply_patch` for manual edits.
5. Do not revert unrelated user changes.
6. After changing YAML, always parse/validate it and check for garbled Chinese.
7. After changing Python, run focused tests or at least `py_compile` on touched files.
8. Keep experiment graph MCP as the source of truth for step progress.

## Priority 1: Preserve Clean Architecture

Goal: make sure xiaozhi is only transport/audio/device glue and Codex app-server owns all MCP tools.

Check these areas:

- `main\xiaozhi-server\core\connection.py`
- `main\xiaozhi-server\codex_app.py`
- `main\xiaozhi-server\device_trigger_mcp_server.py`
- `main\xiaozhi-server\core\providers\tools\server_mcp`
- Config fields in `main\xiaozhi-server\data\.config*.yaml`

Tasks:

- Remove or disable any remaining local experiment fast-path progression.
- Remove or disable local UV-Vis intent interception in xiaozhi-server.
- Confirm experimental graph MCP and UV-Vis MCP are available only through Codex app-server tool mounting.
- Confirm model cannot move to next step until graph MCP has actually advanced.

Expected behavior:

- User says "下一步" -> Codex calls graph MCP -> graph advances -> only then assistant speaks next step.
- User says "开始扫描" in Exp2 -> Codex/UV-Vis MCP path handles it, not local xiaozhi intent logic.

## Priority 2: Exp1 YAML Final Cleanup

Files:

- `C:\Users\11979\Documents\GitHub\codex_edu\lab_runs\exp1_AgNPs_synthesis\configs\experiments.yaml`
- `C:\Users\11979\Documents\GitHub\codex_edu\lab_runs\exp1_AgNPs_synthesis\configs\experiments.md`

Backup rule:

- Put backups under `...\configs\backups`.

Required procedure:

- Five samples, one magnetic stir bar.
- Work sample-by-sample: sample 1 complete, then sample 2, up to sample 5.
- Exp1 should not mount or call UV-Vis MCP.

Step edits to verify:

- Merge general setup with placing the stir bar into sample 1 if still separate.
- For each sample:
  - Sodium citrate step.
  - Silver nitrate step.
  - Hydrogen peroxide plus initial mixing/stirring step.
  - Pure water plus KBr plus mixing/stirring step.
  - NaBH4 plus timer plus stirring/observing color step.
  - Photo permission/confirmation as a single photo step.
  - Clean stir bar and place it into next beaker as one step.
- Cleanup/move step must not mention putting the beaker on magnetic stirrer.
- NaBH4 step must be spoken as one coherent instruction, not split into two partial oral outputs.

Validation:

- YAML parses as UTF-8.
- Chinese is not garbled.
- IDs are unique and ordered.
- Graph MCP initializes.
- Assistant does not ask to generate report at the end.

## Priority 3: Exp1 Photo Robustness

Files likely involved:

- `main\xiaozhi-server\device_trigger_mcp_server.py`
- `main\xiaozhi-server\core\api\device_mcp_handler.py`
- `main\xiaozhi-server\trigger_take_photo.py`
- `main\xiaozhi-server\core\connection.py`

Required behavior:

- Any direct photo request triggers exactly one `xiaozhi_take_photo`.
- No graph rollback just because user asks for photo.
- Retake request creates a new timestamped file and does not overwrite older photo.
- Same utterance must not produce duplicate photos.
- Photo names should include sample number and time.
- Add/keep per-device capture lock.
- Log photo timing for bottleneck diagnosis.
- Before actual capture, trigger a short beep, wait 500-800 ms, then capture.

Suggested tests:

- "帮我拍一张一号样品的照片" -> one photo.
- "重新拍2号样品照片" -> one new photo only.
- Two devices request photos close together -> different devices should not block each other unless MCP transport forces it.

## Priority 4: Exp2 UV-Vis Flow

Files to inspect:

- `C:\Users\11979\Documents\GitHub\codex_edu\lab_runs\exp2_UV_Vis_analysis\configs\experiments.yaml`
- `C:\Users\11979\Documents\GitHub\codex_edu\lab_runs\exp2_UV_Vis_analysis\configs\experiments.md`
- UV MCP repo: `C:\Users\11979\Documents\GitHub\xiaozhi_uv_edu`
- Codex app-server MCP config in `main\xiaozhi-server\data\.config_exp2.yaml`

Required flow:

- Dark current.
- Air energy calibration.
- No pure-water blank correction.
- Load samples/reference liquids.
- UV-Vis spectra for samples 1-5.
- Ask for group number before kinetics. Default to group 1 only if absent.
- Kinetics measurement.

UV scan rule:

- Scan all wavelengths for one sample first.
- Then switch to next sample.
- Stop after sample position 5 at final wavelength.

Kinetics rule:

- Position 2 = sample 2 reaction.
- Position 3 = sample 2 reference.
- Position 4 = sample 4 reaction.
- Position 5 = sample 4 reference.
- Output under active group folder:
  - `<device_id>\<group>\uvvis_measure_kinetic\2号样品`
  - `<device_id>\<group>\uvvis_measure_kinetic\4号样品`

Timeout:

- Kinetics can take 70-90 minutes.
- Avoid treating long-running HTTP tool calls as failed at 10 or 60 minutes.
- Consider async job start + polling instead of a single blocking HTTP call.

## Priority 5: Exp2 Reports

Goal: per-group real-time report generation.

Required:

- Each group folder should have its own `experimental_graph_records.yaml` and `experimental_graph_records.pdf`.
- Max absorption wavelength data and kinetics data from the same group must be in the same report.
- `plot_png` images must be embedded directly in the PDF and be readable.
- Kinetics plots should annotate the fitted line slope.

Check related repo:

- `C:\Users\11979\Documents\GitHub\ExperimentalAssistantServer`

Likely report code:

- `tools\yaml_report_to_pdf.py` in the graph/report repo.

Validation:

- Regenerate reports from existing group folders `1` and `2`.
- Compare plot size/readability against Exp1 PDF reference style.

## Priority 6: ASR / VAD Improvements

Goal: faster transcript display without cutting off slow speakers.

Implement or verify:

- VAD config explicitly present in `.config.yaml`, `.config_exp1.yaml`, and `.config_exp2.yaml`.
- `min_silence_duration_ms` around 700-800 ms for general use.
- `threshold` around 0.45.
- `threshold_low` around 0.18-0.20.
- Tail merge window 300-500 ms after endpoint.
- Step-adaptive endpointing:
  - Short confirmations: 600-800 ms.
  - Data reporting: 1200-1500 ms.

Also add chemistry ASR context/hotwords to all three config files:

- sodium citrate / 柠檬酸钠
- silver nitrate / 硝酸银
- hydrogen peroxide / 过氧化氢
- potassium bromide / 溴化钾
- sodium borohydride / 硼氢化钠
- Tyndall / 丁达尔
- nitrophenol / 硝基苯酚
- UV-Vis / 紫外可见

## Priority 7: TTS Normalization

Goal: avoid bad speech for hyphens, tables, and experiment notation.

Implement/verify text normalization before TTS:

- `1-5号样品` -> `1到5号样品`
- `4-硝基苯酚` -> `4位硝基苯酚`
- Avoid Markdown tables in spoken responses.
- Avoid horizontal rule `---` in spoken responses.
- Avoid bullet dashes if they get spoken as "负".

Behavioral prompt rule:

- If user asks a side question during a pending step, answer it first.
- Then ask: "我们现在能继续做实验了吗？"
- Keep current graph step pending in backend.
- Do not repeatedly re-speak the same unfinished step unless user asks.

## Priority 8: Pad + PC Webcam Migration

Goal: replace chip as user-facing terminal with a pad, and use a separate PC webcam.

Phase 1: inspect and reuse browser client:

- `main\xiaozhi-server\test\test_page.html`
- `main\xiaozhi-server\test\js\app.js`
- `main\xiaozhi-server\test\js\core\audio\recorder.js`
- `main\xiaozhi-server\test\js\core\audio\player.js`
- `main\xiaozhi-server\test\js\core\network\websocket.js`

Phase 2: create minimal pad client:

- Connection status.
- ASR transcript.
- Assistant/TTS text.
- Current operation subtitle.
- Start/stop or push-to-talk controls.
- Audio playback through pad, allowing bone-conduction headset.

Phase 3: add PC webcam backend:

- Config-selectable camera provider.
- Use Windows host webcam instead of device-side `self.camera.take_photo`.
- Keep same MCP tool name if possible so prompts do not change.
- Preserve photo naming, data path, dedupe, per-device lock, and beep-before-capture.

Important:

- Browser microphone over LAN may require HTTPS depending on pad OS/browser.
- Test protocol with existing browser client before refactoring server code.

## Common Commands

Start Codex app-server, from `main\xiaozhi-server`:

```powershell
python codex_app.py --host 127.0.0.1 --port 9001 --llm-name codex_app_server
```

Start xiaozhi-server, from `main\xiaozhi-server`:

```powershell
python app.py
```

Useful checks:

```powershell
git status --short
rg -n "experiment_fast_path_enabled|strict_graph_path|take_photo|voiceprint|vad|tool_timeout" main/xiaozhi-server
```

## Stop Conditions

Stop and ask the user before:

- Deleting preserved device data folders.
- Resetting git state.
- Rewriting large experiment YAML files without backup.
- Changing the live experiment config during a student run unless the user explicitly asks.

