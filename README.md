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

> **Warning:** R08 only works with the **Excessive Overkill robot stream**.
> It will not work with other streams or layouts.

### 1. Install Linux dependencies

For a one-command install on a supported distribution, run:

```bash
curl -fsSL https://raw.githubusercontent.com/benjimonjo/r08_twin/main/install.sh | sh
```

This detects `apt`, `dnf`, `pacman`, or `zypper`, installs the system
dependencies, clones R08 into `~/.local/share/r08-twin`, creates its virtual
environment, and adds an application-menu shortcut. Review the script before
running it if your local security policy does not allow piping downloaded
scripts to a shell. Set `R08_SKIP_SYSTEM_DEPS=1` to skip package-manager
installation, or set `R08_INSTALL_DIR` to choose another install directory.
The installer checks for Discord because the Excessive Overkill robot stream
is provided through Discord. If Discord is missing, it offers a native
distribution package, the official Flatpak package, or exiting. Use
`R08_INSTALL_DISCORD=1` to install the native package non-interactively, or
`R08_INSTALL_DISCORD=0` to exit.

To install from a checked-out copy instead:

```bash
./install.sh
```

Choose the command for your distribution. Debian-based derivatives such as
Linux Mint and Pop!_OS use the Debian/Ubuntu command.

#### Debian, Ubuntu, Linux Mint, and Pop!_OS

```bash
sudo apt update
sudo apt install \
  build-essential python3-venv python3-dev python3-gi gir1.2-xdp-1.0 \
  gstreamer1.0-pipewire tesseract-ocr \
  xdg-desktop-portal xdg-desktop-portal-kde
```

#### Fedora

```bash
sudo dnf install \
  gcc gcc-c++ make python3 python3-devel python3-gobject \
  libportal-devel pipewire-gstreamer tesseract \
  xdg-desktop-portal xdg-desktop-portal-kde
```

#### Arch Linux and Manjaro

```bash
sudo pacman -S --needed --disable-sandbox \
  base-devel python python-gobject \
  libportal gst-plugin-pipewire tesseract tesseract-data-eng \
  xdg-desktop-portal xdg-desktop-portal-kde \
  qt6-multimedia-gstreamer pipewire-jack
```

#### openSUSE Tumbleweed and Leap

```bash
sudo zypper install \
  gcc gcc-c++ make python3 python3-devel python3-gobject \
  libportal-devel gstreamer-plugin-pipewire tesseract-ocr \
  xdg-desktop-portal xdg-desktop-portal-kde
```

If a package name differs on your release, install the equivalent packages
for Python GObject bindings, libportal/Xdp introspection, GStreamer PipeWire
support, Tesseract, and the desktop portal backend. The KDE portal backend
(`xdg-desktop-portal-kde`) is required for the Wayland screen-share chooser;
GNOME users can substitute `xdg-desktop-portal-gnome`.

### 2. Create a virtual environment and install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements include the Qt, OCR, capture, input, and rendering packages.
`moderngl` is optional at runtime; without it, R08 uses the CPU renderer.
`pygame` is optional and enables the **Controller** menu. Connect a game
controller, open **Controller → Configure button mapping**, and choose the
named buttons for the six position and six angle directions plus Step+/Step−.
For recognized PlayStation controllers, for example, the menu shows names such
as `Triangle`, `Square`, `Circle`, and `Cross` together with their SDL button
numbers, as well as `D-pad Up`, `D-pad Down`, `D-pad Left`, and `D-pad Right`.
Joystick axes are also available, and each mapping row has a **Bind** button:
click it and press the desired controller button, D-pad direction, or move the
desired joystick direction. Holding a mapped button or D-pad direction repeats
the jog, while joystick movement is proportional to stick deflection.
The controller mapping follows the PS5/Xbox layout by default: the sticks jog
position, holding the **Angle modifier** switches them to Xa/Ya/Za, and holding
the **View modifier** switches them to 3D orbit/pan. All modifier buttons and
stick axes can be remapped and the camera axes inverted independently.
**Controller input readout** shows every detected button, D-pad state, and
joystick axis value live.
The **Abort** action can also be assigned with **Bind**; it triggers once when
the selected control is pressed and does not repeat while held.
The held-button/D-pad repeat delay is configurable in **Settings** or under
**Controller → Set held-input repeat delay**. **Controller → Set controller
input mode** switches between planning a target pose and direct mode. Direct
mode repeatedly clicks the calibrated robot-panel buttons and requires
**Automated input (Wayland portal)** plus panel calibration.

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
The Live twin view includes **Map camera onto robot** and **Overlay camera
feed**. Configure the Discord camera region, select Camera 1 or Camera 2,
switch that PTZ camera to its home position, then choose
**Configuration -> Calibrate active camera using robot points** and click the
landmarks in this order: base, J1 through J6, flange, and tool tip. Each camera
has its own saved calibration. The camera texture can be projected onto the
reconstructed robot, or the live camera image can be blended over the 3D view
with the adjustable overlay control in the Live twin toolbar.
After making an initial landmark calibration for a camera, use
**Auto-refine active camera from robot yellow** to refine its pose from the
known robot coordinates and FANUC-yellow silhouette. This works best with the
camera at its saved home position, the robot visible, and stable lighting.
The automatic pass requires the initial calibration as a safe geometric seed;
it reports a failure instead of saving an unreliable transform when too few
yellow landmarks can be matched.
3D camera presets, follow/trail display, camera overlays, active-camera
selection, Gaussian view, axis visibility, performance indicators, and the GPU
or CPU computation method are available from the **3D View** menu. The menu
also includes a toggle for the `GPU (OpenGL)` renderer label shown in the
viewport. The GPU method toggle takes effect after restarting R08.
Configuration actions are available from the **Configuration** tab.
The main window includes a toggleable **Debug** window. It is a scrollable
diagnostics panel containing application logs, OCR snapshots and cell previews,
the Digit reader's live output and teaching controls, OCR pose readouts,
continuous step-size OCR status and values, automatic button
recognition results and coordinates, automatic step-region detection, capture
errors, and other recognition diagnostics. The live camera preview and
calibration status are also available in this Debug window.
The Panel camera controls group continuously OCRs the movement-panel feed for
the active Camera 1/Camera 2 indicator and updates the selected camera
calibration automatically. If the panel does not expose a readable camera
indicator, the readout remains unknown rather than guessing.
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
The Settings dialog includes **Pose smoothing strength** for regular
interpolation between the slower OCR coordinate updates. When a calibrated
camera feed is enabled, **Camera pose contribution** blends visual coordinates
from the tracked FANUC-yellow arm into that smoothing path; if the yellow mask
cannot provide enough reliable landmarks, R08 falls back to the OCR pose.
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
