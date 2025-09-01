#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Install Dependencies ---
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# --- Install xvfb for headless GUI testing ---
echo "Installing xvfb and Qt platform plugin..."
sudo apt-get update
sudo apt-get install -y xvfb libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0 libxcb-xkb1

# --- Run Tests ---
echo "Running all tests..."
# The GUI test needs a virtual screen to run headlessly.
# We use xvfb-run to provide this.
xvfb-run -a pytest tests/
