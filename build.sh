#!/usr/bin/env bash
# Sync the working source into the release folder and the installed extension.
set -euo pipefail

SRC="/home/vidyut/Softwares/CODING/mer_build"
REL="/home/vidyut/Softwares/CODING/Mesh expression retargetter/mesh_expression_retargeter_1.2.12_blender_extension"
INST="/home/vidyut/.config/blender/5.2/extensions/user_default/mesh_expression_retargeter"

python3 -m py_compile "$SRC/__init__.py"
echo "syntax ok"

for dst in "$REL" "$INST"; do
    mkdir -p "$dst"
    rm -rf "$dst/__pycache__"
    cp -a "$SRC/CHANGELOG.md" "$SRC/README.md" "$SRC/LICENSE" \
        "$SRC/blender_manifest.toml" "$dst/"
    cp -a "$SRC/__init__.py" "$dst/"
    # LICENSE is not in the working source; leave any existing copy alone.
    rm -rf "$dst/__pycache__"
    echo "synced -> $dst"
done

echo "--- installed copy matches source ---"
diff -q "$SRC/__init__.py" "$INST/__init__.py" && diff -q "$SRC/__init__.py" "$REL/__init__.py" && echo "identical"

# Extension package: the zip holds the package files at its root, exactly as
# the Blender Extensions platform expects.
VER=$(sed -n 's/^version = "\(.*\)"/\1/p' "$SRC/blender_manifest.toml")
ZIP="$(dirname "$REL")/mesh_expression_retargeter_${VER}_blender_extension.zip"
rm -f "$ZIP"
(
    cd "$REL"
    zip -9 -X "$ZIP" CHANGELOG.md LICENSE README.md __init__.py blender_manifest.toml
)
echo "--- packaged ---"
unzip -l "$ZIP"
