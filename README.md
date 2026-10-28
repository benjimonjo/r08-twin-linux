# R08 live twin v3 – FANUC R-2000iA/200F

The active robot appearance is imported from
`fanuc-r2000ia-165f.snapshot.2/FANUC_R2000iA165F.STEP` using CadQuery/OpenCascade.
The STEP is a fused CAD assembly rather than separate joint solids. The
existing kinematic chain drives pose calculations, and the imported envelope
translates by the live TCP displacement so detected motion is reflected
without pretending that the fused CAD can articulate. The complete
tessellated assembly is retained for surface continuity and correct rendering.
The folder's original STL is a partial component, not the complete robot.

Reads the Position/Angle table off the stream, shows a live 3D twin, and guides you click-by-click on the jog panel.

## Install & run
```
pip install numpy opencv-python mss pillow pynput moderngl PySide6
python run.py
```
On Linux Wayland, capture uses libportal's GObject bindings and the XDG
ScreenCast portal to open a PipeWire stream. Install the system packages for
PyGObject, libportal's Xdp introspection data, GStreamer, and the GStreamer
PipeWire plugin. Plasma will ask you to choose a screen and grant access. The
Wayland UI uses PySide6/Qt; install your distribution's Qt Wayland platform
plugin if the Qt wheel does not include it. This native frontend currently
uses the live viewer, screen setup, movement planning, object mapping, and digit
teaching. Guided movement can be performed manually. Optional automated panel
input uses the XDG RemoteDesktop portal: enable it in Movement options, set up
the panel button positions, then approve KDE's pointer permission prompt for
each run. The click interval is configurable; automation pauses if the screen
reader cannot verify an action. Install the KDE portal backend (`xdg-desktop-portal-kde`)
and libportal Xdp introspection bindings for your distribution.
Full setup now asks for the panel's step-size text region on the movement-panel
capture. The step OCR worker reads it continuously (about twice per second) and
syncs the selected jog size.
While a coarse jog step is selected, the position reader increases its sampling
rate and rereads all six positions so large movements are not skipped.
Install `pytesseract` and the Tesseract system executable for that feature and
for automatic panel-button recognition; manual button setup remains available.
On X11, Windows, and macOS the app continues to use `mss` directly.
* Put this folder where your old `r08_config.json` is (or copy that file in) – your setup carries over.
* `moderngl` = smooth GPU 3D. Without it the app still works using a slower CPU renderer (it tells you in the log).
* Tesseract is **not needed** for the six position numbers. It is used for continuous step-size reading and optional button-label recognition.

## What changed vs v2
| Problem | Fix |
|---|---|
| Model was wrong | Rebuilt as R-2000iA/200F: reach 2483 mm, J2 offset 312, lower arm 1075, forearm 1096, J4 motor box at the elbow, round wrist cap, flat tool plate + 3 round spikes. TCP = tip of the 14.4 in spike. |
| 5↔9 and other misreads | New digit reader (`r08/digits.py`): glyph segmentation + template matching trained on *your* font, strict layout rules (`+65.3`, `+170`), two-scan confirmation, plus the speed gate. Tested 600/600 on noisy synthetic cells using seed fonts that differ from the real one. |
| Lag | GPU renderer (OpenGL, 4× MSAA) in the main thread; OCR is ~100× cheaper (no Tesseract calls); render only when something changed. |

## First-time steps
1. Setup ▸ Run full setup (same as before).
2. **Digit reader ▸ Teach digits…** – type what the stream shows right now (e.g. `+65.3`, `+170`), press Learn. Repeat at 2–3 poses. Samples are saved in `r08_digits.npz`.
3. If a number is ever wrong: **Save OCR snapshot** → look in `ocr_debug/` (`cell*_seg.png` shows what the reader saw, `report.txt` what it read), then Teach digits with the correct text.

## Tool orientation
If the spikes look rotated/mirrored compared with the photos, use **Flip long-spike side** / **Turn row 90°** (saved in config).
Robot dimensions / tool lengths live at the top of `r08/kinematics.py`.

See `ARCHITECTURE.md` for how the code is organised (written for people and other AIs extending it).
