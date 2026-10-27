# R08 live twin v3 – FANUC R-2000iA/200F

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
supports the live viewer, screen setup, and digit teaching; guided global
pointer automation remains unavailable on Wayland.
On X11, Windows, and macOS the app continues to use `mss` directly.
* Put this folder where your old `r08_config.json` is (or copy that file in) – your setup carries over.
* `moderngl` = smooth GPU 3D. Without it the app still works using a slower CPU renderer (it tells you in the log).
* Tesseract is **no longer needed** for the numbers. (Only the optional "Step size" text uses it.)

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
