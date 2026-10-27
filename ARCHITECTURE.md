# ARCHITECTURE (for developers and AI assistants)

Desktop app (Tkinter). Python 3.9+. Entry: `run.py` -> `r08/ui.py:main()`.

## Data flow
```
screen --mss--> OcrWorker (thread) --digits.read_cell--> Shared.offer_pose (speed gate) --> Shared.pose
                                                     \--> IKWorker (thread): pose -> joint angles q  (Shared.ik["live"])
UI thread (App.tick ~60 Hz): Shared.snapshot() -> smooth q_vis -> RobotMesh.update(q) -> triangles()
         -> render.Renderer.render(Frame) -> PIL -> Tk canvas
User jogs/Go: App.go_pose -> planner.plan_presses -> Guide (state machine: aim -> user clicks -> verify -> replan)
```

## Modules (all in `r08/`)
| File | Responsibility | Touch when you want to… |
|---|---|---|
| `common.py` | constants, theme `C`, `Config` (r08_config.json), settings defaults, step levels | add a setting / colour |
| `kinematics.py` | robot dimensions, tool spikes, FK, damped-least-squares IK | change robot/tool geometry |
| `digits.py` | digit reader: `segment`, `DigitBank` (templates), `read_cell`, `teach` | fix OCR problems |
| `screen_reader.py` | `Shared` state + speed gate, `OcrWorker`, `IKWorker`, banner colour, step-size text | change scanning logic / gate limits |
| `model3d.py` | `Camera`, `View` (projection), `RobotMesh` (PART_DEFS + `update`), plates/pipes, floor | change how the robot looks |
| `render.py` | `GLBackend` (moderngl), `CPUBackend` (OpenCV fallback), `draw_overlay` (2D HUD) | rendering/perf |
| `planner.py` | fewest jog-button presses (pure logic) | planning |
| `guide.py` | guided click state machine (never clicks for the user) | guided behaviour |
| `inputs.py` | Cursor glide, global clicks/hotkeys (pynput), Beacon ring | input handling |
| `picker.py` | one-time setup overlay | setup flow |
| `widgets.py` | Tk helpers, key map, help text | UI widgets |
| `ui.py` | `App`: window, cards, main loop, object mapping, teach/snapshot tools | UI features |

## Conventions
* **Pose** = `np.array([X,Y,Z (in), Xa,Ya,Za (deg)])` = what the panel prints (FANUC WORLD frame, W/P/R). TCP = tip of spike `TCP_SPIKE`.
* **Base frame** = metres, +Z up, origin on J1 axis at ground. WORLD origin = J1 axis at J2 height (`TABLE_ORIGIN_IN_BASE_M`). `kinematics.table_to_base()` converts.
* **Triangle soup** = `(pos (N,3), normal (N,3), colour (N,3))` float32, N multiple of 3, normals outward, colours 0..1. Renderers only consume soups.
* Threads: workers write `Shared` under `sh.lock`; UI reads `sh.snapshot()`. No Tk calls from threads. The GL context lives in the UI thread.
* `r08_config.json` keys: region, status, step, buttons, level, settings, window, waypoints, objects, tool.

## Digit reader details (most likely place for future fixes)
1. `segment(gray)`: upscale 3x, contrast-normalise, threshold at 50%, remove long table lines (morphology), connected components, drop noise, merge broken pieces, split glued digits, classify each blob: `+` (squarish, ~0.8 digit height), `-` (flat wide), `.` (small, on baseline), `digit`.
2. `DigitBank.match`: cosine similarity of 16x24 normalised glyph vs templates (+ small aspect-ratio penalty). Templates = seeds rendered from system fonts + learned samples from the real screen (`r08_digits.npz`, bonus +0.03).
3. `read_cell`: enforces layout (row 0: one decimal; row 1: integer), min score 0.45, magnitude limits.
4. `OcrWorker._update_cell`: a changed value must be read identically on two consecutive scans; sign can be inferred from the previous value if the glyph is lost.
5. `Shared.offer_pose`: rejects jumps faster than `GATE_*` unless seen `GATE_CONFIRM` times.
Test idea: render text with PIL into 150x52 cells (white on dark, noise, grid lines) and call `read_cell`.

## Adding things
* New robot part: add to `PART_DEFS`, assign `V[S["name"]] = ...` in `RobotMesh.update`.
* New 3D object: build a soup (`model3d.solid_soup`) and append it to `tris` in `App._render`.
* New 2D overlay item: `render.draw_overlay` (uses `View.project`).
* Model accuracy: only `kinematics.py` numbers affect where the twin thinks the arm is; `model3d.py` is cosmetic.
