#!/usr/bin/env bash
# Download DemoGrasp assets + checkpoints from official Google Drive
# Source: https://github.com/BeingBeyond/DemoGrasp#requirements
# Link: https://drive.google.com/drive/folders/1NXDcMI5IXalOwauryWvXCz3h2n2IzBQ6?usp=sharing
set -euo pipefail
pip install --quiet gdown
DEST="${1:-/opt/demograsp}"
gdown --folder "https://drive.google.com/drive/folders/1NXDcMI5IXalOwauryWvXCz3h2n2IzBQ6" -O "$DEST/"
cd "$DEST"
for z in ckpt.zip robots.zip union_ycb_unidex.zip textures.zip; do
  [ -f "$z" ] && unzip -qo "$z" && rm "$z"
done
echo "DemoGrasp assets ready at $DEST"
# Expected structure:
#   /opt/demograsp/ckpt/inspire.pt
#   /opt/demograsp/assets/textures/
#   /opt/demograsp/assets/union_ycb_unidex/
#   /opt/demograsp/assets/franka/
#   /opt/demograsp/assets/inspire_tac/
