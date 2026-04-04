#!/usr/bin/env bash
# Download DemoGrasp assets + checkpoints from official Google Drive
# Source: https://github.com/BeingBeyond/DemoGrasp#requirements
# Link: https://drive.google.com/drive/folders/1NXDcMI5IXalOwauryWvXCz3h2n2IzBQ6?usp=sharing
set -euo pipefail
uv pip install --system --quiet gdown
DEST="${1:-/opt/demograsp}"
gdown --folder "https://drive.google.com/drive/folders/1NXDcMI5IXalOwauryWvXCz3h2n2IzBQ6" -O "$DEST/"
cd "$DEST"
for z in *.zip; do
  [ -f "$z" ] || continue
  unzip -qo "$z" && rm "$z" || echo "WARNING: failed to extract $z"
done
echo "DemoGrasp assets ready at $DEST"
# Expected structure:
#   /opt/demograsp/ckpt/inspire.pt
#   /opt/demograsp/assets/textures/
#   /opt/demograsp/assets/union_ycb_unidex/
#   /opt/demograsp/assets/franka/
#   /opt/demograsp/assets/inspire_tac/
