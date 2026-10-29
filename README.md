# R08-twin-linux – FANUC R-2000iA/200F

The active robot appearance is loaded from `robot_files`, using the separate
colorized PLY links and `robot.json` DH data. The exported materials preserve
the FANUC-style yellow housings, charcoal base/covers, and red motor details.
Each link is transformed by its joint frame, so detected motion is reflected
in the articulated model. The older STEP and STL assets remain available as
reference files.

Reads the Position/Angle table off the stream, shows a live 3D twin, and guides you click-by-click on the jog panel.

## Install & run

R08-twin-linux currently targets Linux, with the native Qt frontend used on
Wayland sessions.

### 1. Install Linux dependencies

On Debian or Ubuntu:

```bash
sudo apt update
sudo apt install \
  python3-venv python3-dev python3-gi gir1.2-xdp-1.0 \
  gstreamer1.0-pipewire tesseract-ocr \
  xdg-desktop-portal xdg-desktop-portal-kde
```

Make sure the KDE portal backend is running for the desktop session. On other
distributions, install the equivalent packages for Python GObject bindings,
libportal/Xdp introspection, GStreamer PipeWire support, Tesseract, and the
desktop portal backend.

### 2. Create a virtual environment and install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements include the Qt, OCR, capture, input, and rendering packages.
`moderngl` is optional at runtime; without it, R08 uses the CPU renderer.

### 3. Start the application

```bash
source .venv/bin/activate
python run.py
```

On first use, the desktop portal asks you to choose the required screen or
window captures. R08 saves separate PipeWire restore tokens in
`r08_config.json` for the primary R08 stream, movement-controls stream, and
pointer workspace. If a saved token is no longer valid, R08 removes it and
prompts you to select that capture again.

The Wayland UI provides live viewing, screen setup, movement planning, object
mapping, digit teaching, and optional automated panel input. Automated input
requires the KDE RemoteDesktop pointer permission and pauses if the screen
reader cannot verify an action.
Full setup now asks for the panel's step-size text region on the movement-panel
capture. The step OCR worker reads it continuously (about twice per second) and
syncs the selected jog size.
Panel button recognition also detects the camera controls (`Up`, `Down`, `Left`,
`Right`, `Zoom+`, `Zoom-`, `Reset Camera`, and `Switch Camera`) from their
regular layout, including the dark buttons that are not found by color
segmentation.
It also recognizes and exposes the panel safety controls `Abort`, `Backup`,
`Safe J23`, and `Safe J456` in the UI and button review calibration. The two
safe-position buttons are kept separate because the panel contains both.
Camera jog step buttons are exposed as `Camera Step+` and `Camera Step-`,
separate from the robot position step buttons.
Once calibrated, these controls are available in the UI's **Panel camera
controls** group. They use the same manual-confirm or optional automated panel
action flow as robot movement buttons.
The native UI also has a **Configuration** tab containing the capture, number
table, panel-button, camera-video, step-size, and status setup actions. Its
**Find arm reach limits** action samples the configured joint limits and stores
an estimated TCP workspace bounding box in the configuration.
The Live twin view includes **Map camera onto robot**. Configure the Discord
camera region, choose **Configuration -> Calibrate camera using robot points**,
then click the landmarks in this order: base, J1 through J6, flange, and tool
tip. After calibration, the live camera texture is projected onto the
reconstructed robot and can be toggled from the Live twin toolbar.
The main window includes a toggleable **Debug** window. It shows application
logs, OCR pose readouts, continuous step-size OCR status and values, automatic
button recognition results and coordinates, automatic step-region detection,
capture errors, and other recognition diagnostics. Enable **Show live camera
preview** from the Configuration tab to inspect the captured frame, resolution,
calibration landmarks, and latest camera update time.
The robot position table and Discord camera region use the same primary
PipeWire capture session, so they can be selected from one shared screen
stream. Use **Configuration -> Clear saved PipeWire tokens** when you want the
desktop chooser to appear again.
Automated pointer input accepts normal PipeWire/RemoteDesktop scaling
differences between the capture stream and input stream instead of rejecting
the session for a mismatched aspect ratio.
When the panel was captured as a window, the pointer permission prompt also
requests that panel window; select the same window again. When the panel was
captured as a monitor, select that monitor instead. R08 stops before moving
the mouse if the selected stream does not match the calibrated panel source.
Calibrated landmark markers are recomputed from the current live joint pose, so
they move with the robot instead of remaining fixed at their original image
coordinates. The Live twin view also shows the moving `C0`, `C1`, etc. markers
when camera mapping is enabled.
While a coarse jog step is selected, the position reader increases its sampling
rate and rereads all six positions so large movements are not skipped.
`pytesseract` and the Tesseract system executable are required for continuous
step-size OCR and automatic panel-button recognition; manual button setup
remains available.
On X11, Windows, and macOS the app continues to use `mss` directly.
* Put this folder where your old `r08_config.json` is (or copy that file in) – your setup carries over.
* `moderngl` = smooth GPU 3D. Without it the app still works using a slower CPU renderer (it tells you in the log).
* Tesseract is **not needed** for the six position numbers. It is used for continuous step-size reading and optional button-label recognition.

## What changed vs v2
| Problem | Fix |
|---|---|
| Model was wrong | Rebuilt as R-2000iA/200F from the separate `robot_files` links: flat tool plate plus three slim, separated local spikes. The spike lengths follow the last-joint (J6) axis, with the row offset perpendicular to it. The TCP is the tip of the longest 14.4 in spike, and detected coordinates are interpreted at that tool point. |
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
