# Mesh Expression Retargeter

Mesh Expression Retargeter is a Blender add-on for transferring shape keys, facial expressions, sculpted deformations, and other mesh deformation states between meshes with different topology.

It is useful for tasks such as:

- transferring shape keys between unrelated meshes
- transferring facial expressions between character systems
- mapping sculpted shape keys or corrective sculpts onto lower-poly animation meshes
- transferring deformations between proxy, high-resolution, and low-resolution meshes
- reusing useful deformation shapes from one stable topology on another

Transferred deformation is stored as a native Blender shape key on the Target mesh.

It provides two workflows.

## Manual Correspondences

Use this when Source and Target meshes have unrelated topology, different proportions, or sit anywhere in the scene.

1. Choose Source Mesh and Target Mesh.
2. Put the Source in its neutral state.
3. Enter Pairing Mode.
4. Click a landmark on Source, then the matching landmark on Target.
5. Repeat for enough landmarks to constrain the fit.
6. Optionally save the mapping for reuse.
7. Fit / Build Transfer Map.
8. Change the Source to the desired expression/deformation.
9. Transfer Current Expression, or use Transfer All Shape Keys to batch-transfer every non-Basis Source shape key.

Use Remove Pair to click a correspondence marker and delete just that pair,
keeping the ones that came after it. Delete Last and Clear All remain for the
coarser edits.

The result is created as a Blender shape key on the Target. Transfer All Shape Keys preserves Source key names; name collisions create Blender-style numbered copies such as `Smile.001`.

A saved mapping is based on stable Source/Target vertex indices, so it can be reused on different character shapes that share those same topologies.

## MPFB Expressions

MPFB is optional. It is found by scanning for a loaded module rather than by a
fixed path, so it works whether MPFB was installed as an extension or run from
source. When it is missing, or loaded but not finished registering, the panel
says which of the two it is instead of quietly hiding the buttons.

MPFB expressions are not shape data. An MPFB expression file stores only
weights over MPFB's 52 ARKit face units, which are themselves fixed
deformations of the MPFB basemesh. So a transferred shape key cannot be
written out as-is; it has to be projected onto those units.

When MPFB is installed and Target is an MPFB basemesh, two further buttons
appear under the transfer buttons:

1. Transfer the expression to a Target shape key as usual.
2. Use Save as MPFB Expression to write it into the MPFB expression library,
   or Save All as MPFB Expressions to export every non-Basis Target shape key.

That projection is a best fit. The panel reports how much of the pose the
face-unit vocabulary could not represent. The Target shape key is unchanged
and remains the exact result, so keep it when the fit is not good enough.

## Overlapping Surfaces

Use this when Source and Target are already spatially aligned.

No landmark pairs are required. The add-on builds nearest-surface barycentric correspondence directly from the overlapping neutral meshes.

This is useful for:

- proxy meshes
- aligned high-poly and low-poly meshes
- sculpt-to-base deformation transfer
- expression meshes already fitted to a target
- meshes with different topology but corresponding neutral surfaces

## Mapping Library

Manual correspondence mappings can be named, saved, loaded, replaced, and deleted.

Mappings can also ship read-only inside the add-on. These appear in the list
marked *(bundled)* and cannot be overwritten or deleted — load one and save it
under a new name to adapt it.

## Limitations

Transferred deformation is only as good as the match between the two neutral
surfaces. Most meshes need nothing extra from you.

Meshes with a **separate inner-lip surface** are the exception. MPFB's
basemesh has one: the inner and outer lips sit about a millimetre apart at
rest, so a nearest-surface bind can glue them together and stretch the mouth
edge. **Match Inner Mouth**, **Relax Mapped Motion** and **Prefer Same-Facing
Bindings** (all on by default) exist to hold this back, and between them they
remove most of it. It is still worth expecting to **touch up the lips by
hand** on the result — particularly for pucker and open-mouth poses.

Those mitigations are heuristics rather than a fix. I have suggested to the
MPFB add-on that it expose an explicit mouth-interior vertex group, which
would let any retargeter separate the two sheets directly instead of inferring
them from face orientation.


## Origin

Mesh Expression Retargeter was originally developed by Vidyut Gore to create and animate characters for the science-fiction series *Abrahamha's Strays*. It grew into a general-purpose deformation retargeting tool and is released as free software in gratitude to Blender and the open-source tools that made the project possible.

More Blender tools, characters, and downloadable assets: https://vidyut.net

## Requirements

- Blender 4.0+ for the standalone/site add-on
- Blender 4.2+ for the official Blender Extensions package
- Source topology must remain stable between its neutral and deformed states
- Target topology must remain stable after building the transfer map

## Installation: standalone/site version

In Blender:

1. Edit > Preferences
2. Add-ons
3. Install from Disk
4. Select `mesh_expression_retargeter.py`
5. Enable it if necessary

The panel appears at:

`3D View > Sidebar > Expression Transfer`

## License

GNU General Public License v3.0 or later.
