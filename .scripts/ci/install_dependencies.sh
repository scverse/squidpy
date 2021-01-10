#!/usr/bin/env bash

set -euo pipefail

if [[ "$OS" == "ubuntu-latest" ]]; then
    echo "Installing APT dependencies"

    sudo apt-get update -y
    sudo apt-get install automake -y  # leidenalg

    # PyQt5 related
    sudo apt install libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 -y
    sudo Xvfb :42 -screen 0 1920x1080x24 -ac +extension GLX </dev/null &
elif [[ "$OS" == "macos-latest" ]]; then
    brew install automake  # leidenalg
fi
