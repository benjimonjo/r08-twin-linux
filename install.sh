#!/bin/sh
set -eu

REPO_URL="${R08_REPO_URL:-https://github.com/benjimonjo/r08_twin.git}"
REF="${R08_REF:-main}"
INSTALL_DIR="${R08_INSTALL_DIR:-$HOME/.local/share/r08-twin}"
BIN_DIR="${R08_BIN_DIR:-$HOME/.local/bin}"
DESKTOP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
DESKTOP_FILE="$DESKTOP_DIR/r08-twin.desktop"
SOURCE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
CLONE_SOURCE="$REPO_URL"

if [ -d "$SOURCE_DIR/.git" ]; then
    CLONE_SOURCE="$SOURCE_DIR"
fi

info() {
    printf '%s\n' "R08 installer: $*"
}

fail() {
    printf '%s\n' "R08 installer: $*" >&2
    exit 1
}

warn_stream_requirement() {
    printf '%s\n' ""
    printf '%s\n' "WARNING: R08 only works with the Excessive Overkill robot stream."
    printf '%s\n' "The installer cannot make R08 work with other streams or layouts."
    printf '%s\n' ""
}

install_system_dependencies() {
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y \
            git build-essential python3 python3-venv python3-dev python3-gi gir1.2-xdp-1.0 \
            gstreamer1.0-pipewire tesseract-ocr \
            libegl1 libgl1 libglvnd0 libgl1-mesa-dri mesa-utils \
            xdg-desktop-portal xdg-desktop-portal-kde
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y \
            git gcc gcc-c++ make python3 python3-devel python3-gobject \
            libportal-devel pipewire-gstreamer tesseract \
            mesa-libEGL mesa-libGL mesa-dri-drivers libglvnd-egl libglvnd-opengl \
            xdg-desktop-portal xdg-desktop-portal-kde
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -Syu --needed --noconfirm --disable-sandbox \
            git base-devel python python-gobject \
            libportal gst-plugin-pipewire tesseract tesseract-data-eng \
            mesa libglvnd \
            xdg-desktop-portal xdg-desktop-portal-kde \
            qt6-multimedia-gstreamer pipewire-jack
    elif command -v zypper >/dev/null 2>&1; then
        sudo zypper --non-interactive install \
            git gcc gcc-c++ make python3 python3-devel python3-gobject \
            libportal-devel gstreamer-plugin-pipewire tesseract-ocr \
            Mesa-libEGL1 Mesa-libGL1 Mesa-dri-drivers libglvnd \
            xdg-desktop-portal xdg-desktop-portal-kde
    else
        fail "unsupported distribution; install the packages listed in README.md and rerun with R08_SKIP_SYSTEM_DEPS=1"
    fi
}

warn_stream_requirement
if [ "${R08_SKIP_SYSTEM_DEPS:-0}" != "1" ]; then
    install_system_dependencies
fi

command -v python3 >/dev/null 2>&1 || fail "python3 is required"
command -v git >/dev/null 2>&1 || fail "git is required"

mkdir -p "$(dirname "$INSTALL_DIR")"
if [ -d "$INSTALL_DIR/.git" ]; then
    git -C "$INSTALL_DIR" fetch --depth 1 origin "$REF"
    git -C "$INSTALL_DIR" checkout -q FETCH_HEAD
else
    [ ! -e "$INSTALL_DIR" ] ||
        fail "install directory already exists and is not a git checkout: $INSTALL_DIR"
    git -c "safe.directory=$SOURCE_DIR" clone --depth 1 --branch "$REF" "$CLONE_SOURCE" "$INSTALL_DIR"
fi

python3 -m venv "$INSTALL_DIR/.venv"
"$INSTALL_DIR/.venv/bin/python" -m pip install --upgrade pip
"$INSTALL_DIR/.venv/bin/python" -m pip install -r "$INSTALL_DIR/requirements.txt"
"$INSTALL_DIR/.venv/bin/python" -m playwright install chromium

mkdir -p "$BIN_DIR" "$DESKTOP_DIR"
cat > "$BIN_DIR/r08-twin" <<EOF
#!/bin/sh
exec "$INSTALL_DIR/.venv/bin/python" "$INSTALL_DIR/run.py" "\$@"
EOF
chmod +x "$BIN_DIR/r08-twin"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=R08 Live Twin
Comment=FANUC R-2000iA/200F live 3D twin
Exec=$BIN_DIR/r08-twin
Terminal=false
Categories=Science;Engineering;
StartupNotify=true
EOF
chmod 644 "$DESKTOP_FILE"

if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$DESKTOP_DIR" >/dev/null 2>&1 || true
fi

info "installed to $INSTALL_DIR"
info "launcher: $BIN_DIR/r08-twin"
info "desktop shortcut: $DESKTOP_FILE"
info "if '$BIN_DIR' is not on PATH, launch from the application menu or add it to PATH"
