# Project Handoff

Last updated: 2026-05-20

This repository is the local xiaozhi AI teaching assistant server used to drive two chemistry teaching experiments through Codex, MCP tools, ASR/TTS, device/photo control, experimental graph records, and UV-Vis instrumentation.

Primary repo:

`C:\Users\11979\Documents\GitHub\xiaozhi-esp32-server`

Related repos and data roots:

- Experiment YAML / teaching assets: `C:\Users\11979\Documents\GitHub\codex_edu`
- UV-Vis MCP: `C:\Users\11979\Documents\GitHub\xiaozhi_uv_edu`
- Experimental graph MCP: `C:\Users\11979\Documents\GitHub\ExperimentalAssistantServer`
- Voiceprint API: `C:\Users\11979\Documents\GitHub\voiceprint-api`

## Core Architecture Decision

The clean target architecture is:

- `xiaozhi-server` handles ASR/TTS, websocket clients, device HTTP, logs, and audio/display transport.
- Codex app-server owns all experiment reasoning and all MCP tool calls.
- Experimental graph MCP, UV-Vis MCP, and device/photo MCP tools should be mounted on the Codex app-server side.
- `xiaozhi-server` should not initialize or advance experiment graph or UV-Vis MCP locally.
- Local fast-path logic such as `experiment_fast_path_enabled`, `strict_graph_path`, or local UV-Vis intent interception should be treated as deprecated. If any remaining code path still advances the experiment locally, remove or disable it.

Reason: when xiaozhi and Codex each own MCP clients or graph state, sessions diverge. The assistant then answers from chat history while graph state does not advance, or a new graph process overwrites the previous one.

## Main Config Files

Runtime configs:

- `main\xiaozhi-server\data\.config.yaml`
- `main\xiaozhi-server\data\.config_exp1.yaml`
- `main\xiaozhi-server\data\.config_exp2.yaml`

Experiment 1 graph/prompt files:

- `C:\Users\11979\Documents\GitHub\codex_edu\lab_runs\exp1_AgNPs_synthesis\configs\experiments.yaml`
- `C:\Users\11979\Documents\GitHub\codex_edu\lab_runs\exp1_AgNPs_synthesis\configs\experiments.md`
- Backups for exp1 config edits should go under `...\configs\backups`, not directly under `configs`.

Experiment 2 graph/prompt files:

- `C:\Users\11979\Documents\GitHub\codex_edu\lab_runs\exp2_UV_Vis_analysis\configs\experiments.yaml`
- `C:\Users\11979\Documents\GitHub\codex_edu\lab_runs\exp2_UV_Vis_analysis\configs\experiments.md`

Known service document:

- `docs\codex_app_split_mode.md`

## Model / Runtime Notes

- The preferred Codex model during recent tests was `gpt-5.3-codex-spark` with medium reasoning.
- The user also tested/considered `gpt-5.3-codex` high/medium.
- `xiaozhi-server` uses the `xiaozhi` conda environment.
- UV-Vis MCP uses the `py314` environment.
- Some earlier imports needed Python 3.10 compatibility, for example replacing `tomllib` with `tomli` where needed.
- `approval_policy: never` does not mean shell is disabled. It means no interactive approval prompts. Local shell should still be usable if sandbox and permissions allow it.

## Codex App-Server / Session Behavior

The app-server resumes a Codex thread based on session registry and previous conversation. If testing from a clean state, clear the app-server cache/session registry first.

Important rule after server restart:

- If user says "继续刚才实验 / 恢复实验", the system must enter an explicit recovery path.
- First rebuild graph state from records/logs or call graph MCP `redirect_to_step` to the correct step.
- If graph cannot be synchronized, the LLM must not continue purely from chat/log history.
- If records are insufficient, ask the student what step they are actually on, then attach graph state there.
- Existing recorded data should not force the student to redo completed steps.

User mentioned a Codex resume/thread id that may matter for later manual continuation:

`codex resume 019e003c-b502-73a2-9751-4ab7193dda9f`

## Experiment 1: AgNPs Synthesis

Experiment 1 is a per-device experiment. Different students/devices must write separate data and report files.

Typical data path:

`C:\Users\11979\Documents\GitHub\codex_edu\lab_runs\exp1_AgNPs_synthesis\data\<device_id>`

Desired procedure state:

- There are five samples, numbered 1 to 5.
- There is only one magnetic stir bar.
- The workflow is sequential by sample: finish sample 1 through NaBH4 color change/photo/recording, clean stir bar, move to sample 2, and repeat until sample 5.
- Do not ask students to prepare all samples through later reagent stages in parallel.
- Exp1 does not need the UV-Vis MCP.

Latest desired step structure:

- `step_prepare_setup_all` and "1号样品：放入磁转子" can be merged because the first sample needs the stir bar at setup time.
- For every sample:
  - Add sodium citrate, silver nitrate, and hydrogen peroxide as individual reagent steps.
  - Merge "add H2O2" with the instruction to mix/stir the common solution.
  - Merge "add pure water + KBr" with the stir/mix after KBr/water.
  - Merge "add NaBH4, start timer" with "stir after NaBH4 and observe color". This step must explicitly say that after adding NaBH4, the beaker should be placed on the magnetic stirrer or kept stirring while observing color change.
  - Merge photo permission and photo confirmation into one step.
  - Merge "clean stir bar" with "place stir bar into next numbered beaker". This cleanup/move step must not say to place the beaker on the magnetic stirrer.

Photo behavior:

- If user says "拍照 / 可以拍照 / 拍吧 / 现在拍 / 帮我拍一张...", Codex should immediately call `xiaozhi_take_photo`.
- Do not use photo requests to roll graph state backwards.
- If user says "重新拍 N 号样品照片", take a new photo immediately.
- Photo name should include sample number and timestamp, for example `2号样品照片_YYYYMMDD_HHMMSS`.
- New photos must not overwrite old photos.
- There was a bug where "重新拍2号样品照片" produced two photos. Keep same-utterance photo dedupe in place and verify.
- A short beep before capture was desired: tool trigger layer should beep, wait about 500-800 ms, then capture.

Report behavior:

- Experimental graph YAML and PDF should be generated/updated during the experiment, not only after a user asks for a report.
- Empty/null observation rows should be hidden in PDF output.
- Photo records must match the correct sample number. There was a previous bug where sample 4 color/photo record used sample 5 photo.
- Chinese spoken time like "一分二十秒" should be normalized before passing into Codex or before record writing, to avoid "一分二十" becoming wrong values like "一分三十三".

## Experiment 2: UV-Vis Analysis

Experiment 2 is grouped by device and group number. Different groups must write separate data/report files.

Typical data root:

`C:\Users\11979\Documents\GitHub\codex_edu\lab_runs\exp2_UV_Vis_analysis\data\uv_data_common\<device_id>\<group_number>`

There may also be archived/teaching-record paths under:

`C:\Users\11979\Documents\GitHub\codex_edu\data\教学资料\data_record\exp2_UV_Vis_analysis_YYYYMMDD\data\uv_data_common\<device_id>\<group_number>`

Desired procedure:

1. Start instrument.
2. Measure dark current.
3. Measure air energy calibration.
4. Load samples and reference liquids.
5. Measure UV-Vis spectra for samples 1-5.
6. Ask/confirm group number before kinetics. If group number is not spoken, default to group 1.
7. Run kinetics measurement.
8. Next group repeats the sample loading / UV-Vis / kinetics flow, with data saved into the next group folder.

Important current process corrections:

- There is no pure-water blank correction step anymore. Only dark current and air energy calibration before sample/reference loading.
- For UV-Vis scanning over multiple samples and wavelengths, scan one complete sample first, then switch to the next sample. Do not scan wavelength-major across all samples.
- After sample position 5 and final wavelength, the instrument must stop. There was a bug where it looped again after `last_wavelength_nm=580` and `sampler_position=5`.
- Long kinetics calls can exceed common HTTP tool timeouts. Kinetics may run around 70-90 minutes. A plain 3600 second timeout can be too short.

Kinetics placement:

- Position 2: sample 2 reaction liquid.
- Position 3: sample 2 reference liquid.
- Position 4: sample 4 reaction liquid.
- Position 5: sample 4 reference liquid.

Kinetics data layout:

- Kinetics data must be placed inside the active group folder, not in the next group folder.
- Inside `<group_number>\uvvis_measure_kinetic`, use direct sample folders:
  - `2号样品`
  - `4号样品`
- Data files for each sample go inside the corresponding sample folder.
- Temporary sync scripts such as `.codex_tmp\sync_exp2_kinetics_runtime.py` should not be needed for the final flow.

Report behavior:

- Exp2 graph YAML/PDF should be generated per group folder in real time.
- Maximum absorption wavelength data and kinetics data for the same group must be written into the same `experimental_graph_records.pdf`.
- `plot_png` images should be embedded directly into the PDF, large enough to read. A previous fix enlarged embedded plot images, but this should be verified against the Exp1 PDF style.
- Kinetics plots should include the fitted line slope annotated on the plot.

## ASR / VAD / TTS Notes

ASR:

- Current ASR provider discussed: `Qwen3ASRLocal`.
- It is not reliably streaming in the way the user expects; there is noticeable delay from speech end to transcript display.
- Chemistry terms are often misrecognized. Add experiment context/hotwords to `.config.yaml`, `.config_exp1.yaml`, and `.config_exp2.yaml`.
- Codex should use experiment context to infer likely chemical names from ASR errors.

VAD desired strategy:

- Do not simply make silence timeout extremely low.
- Use layered behavior:
  - General fast stop: `min_silence_duration_ms` around 700-800 ms.
  - `threshold` around 0.45.
  - `threshold_low` around 0.18-0.20.
  - Add a 300-500 ms tail/grace merge window after VAD endpoint; if speech resumes during this window, merge into the same utterance.
  - Short confirmation steps can use 600-800 ms endpointing.
  - Data-reporting steps such as "深蓝色，一分二十秒" should allow 1200-1500 ms endpointing.

TTS:

- TTS reads hyphen `-` incorrectly in contexts like `1-5号样品` and `4-硝基苯酚`, sometimes as "减".
- Use text normalization before TTS:
  - Ranges: `1-5号样品` -> `1到5号样品`.
  - Chemical locants: `4-硝基苯酚` -> `4位硝基苯酚` or another domain-appropriate spoken form.
  - Avoid tables or Markdown separators in speech output because `-` may be spoken badly.
- If user asks a non-step question while a step is pending, answer the question first, then ask "我们现在能继续做实验了吗？"
- Backend should keep current graph step pending while answering side questions. Do not repeatedly re-speak the unfinished step, especially at the final step.
- There was an issue where post-photo speech was garbled and returned to listening before speech finished. Check send-audio timing/state handling.

## Device / Photo / Future Pad-Webcam Migration

Current photo path:

- HTTP endpoint: `/mcp/device/take_photo`
- Server handler: `main\xiaozhi-server\core\api\device_mcp_handler.py`
- MCP side: `main\xiaozhi-server\device_trigger_mcp_server.py`
- Local trigger helper: `main\xiaozhi-server\trigger_take_photo.py`

Per-device photo concurrency:

- Add/keep a per-device lock around photo capture.
- Different devices should be allowed to capture in parallel as much as MCP runtime and transport allow.
- Same device must be serialized to avoid duplicate/overlapping captures.
- Log capture timing so remaining bottlenecks can be identified.

Future hardware migration requested by user:

- Replace xiaozhi chip audio/display terminal with a pad.
- Pad should handle ASR/TTS audio relay, bone-conduction headset, and visible step subtitles.
- Camera should become a separate upright PC webcam attached to the Windows host.
- Program still runs from this repo.

Recommended migration path:

- Build a minimal Pad browser/PWA client using the existing browser test client as a base:
  - `main\xiaozhi-server\test\test_page.html`
  - `main\xiaozhi-server\test\js\core\audio\recorder.js`
  - `main\xiaozhi-server\test\js\core\audio\player.js`
  - `main\xiaozhi-server\test\js\core\audio\opus-codec.js`
  - `main\xiaozhi-server\test\js\core\network\websocket.js`
  - `main\xiaozhi-server\test\js\core\mcp\tools.js`
- Show connection state, ASR transcript, assistant/TTS text, and current step subtitle.
- Keep backend protocol compatible first; do not rewrite the whole server before confirming the browser client protocol.
- Add a config-selectable PC webcam capture backend, for example:
  - provider: `pc_webcam`
  - index: `0`
  - warmup frames: configurable
- Route `xiaozhi_take_photo` to PC webcam capture when configured.
- Keep naming, per-device/group data paths, dedupe, and beep-before-capture behavior.

Important caveat:

- The shared ChatGPT URL supplied by the user could not be read without login. Do not claim details from that link unless the user provides content directly.

## Paths That Commonly Need Clearing Before Tests

Clear only with care and preferably back up first:

- `main\xiaozhi-server\data\codex_app`
- `main\xiaozhi-server\data\cache*` or timestamped cache backup folders as relevant
- Experiment device data under:
  - `...\lab_runs\exp1_AgNPs_synthesis\data\<device_id>`
  - `...\lab_runs\exp2_UV_Vis_analysis\data\uv_data_common\<device_id>`

The user frequently asks to preserve devices:

- `94_a9_90_28_e8_d8`
- `94_a9_90_28_ea_b4`

Do not delete preserved device folders unless explicitly asked.

## Known Device IDs Seen

- `94_a9_90_28_e8_d8`
- `94_a9_90_28_ea_b4`
- `94_a9_90_28_eb_58`

## WebSocket / Network Notes

The user changed websocket URLs during testing. Examples used:

- `ws://192.168.1.100:8000/xiaozhi/v1/`
- `ws://192.168.1.106:8000/xiaozhi/v1/`

There was a problem where editing `.config.yaml` websocket address reverted. Check config copy/sync/startup logic before assuming manual edits persist.

There was also a concern that Codex may first try a stale websocket and only later reconnect. Confirm current connection fallback behavior from logs if startup is slow.

## Validation Expectations

After editing YAML:

- Validate YAML parses as UTF-8.
- Check for mojibake/garbled Chinese.
- Check step IDs are unique and in intended order.
- Check prompt text does not ask for report generation at the end.
- Check graph MCP can initialize with the YAML.

After editing server code:

- Run focused tests if available.
- At minimum run Python compile checks on touched Python files.
- Watch logs during live chip tests.
- Do not leave long-running services active after user asks to stop.

