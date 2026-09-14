# Mesh Expression Retargeter

Mesh Expression Retargeter is a Blender extension for transferring expressions, shape keys, and other mesh deformations between meshes with different topology.

It was created to solve a simple problem: an expression may already exist on one character or mesh, while the mesh you actually want to animate has different geometry. Mesh Expression Retargeter lets you establish the relationship between the two meshes once, then transfer deformations to native Blender shape keys.

This makes it useful not only for facial expressions, but also for transferring sculpted corrections, deformations between character systems, and shapes between high-resolution, low-resolution, or proxy meshes.

## Features

- Transfer expressions and mesh deformations between different meshes
- Manual landmark correspondence for unrelated meshes
- RBF-based non-rigid fitting
- Save and reuse named correspondence mappings
- Reuse mappings on differently shaped characters with the same underlying topology
- Overlapping Surfaces mode for meshes that are already spatially aligned
- Transfer evaluated/deformed source geometry
- Create the transferred result as a native Blender shape key
- Show or hide correspondence markers for verification
- No requirement for the source and target meshes to have matching topology in Manual Correspondences mode

## How It Works

Mesh Expression Retargeter provides two mapping methods.

### Manual Correspondences

Use this when the source and target have different topology, proportions, positions, or character designs.

Select corresponding landmarks on the two meshes: corners of the eyes, mouth, nose, chin, or other useful structural points. The extension uses those correspondences to fit the source deformation to the target.

Mappings can be named and saved. Once a mapping has been created for two topologies, it can be loaded again for other characters using those same topologies, even when their individual head shapes differ.

This is particularly useful when transferring expressions between character-generation systems.

### Overlapping Surfaces

Use this when the source and target meshes already occupy approximately the same space.

The extension establishes correspondence from the overlapping surfaces directly, avoiding the need to create manual landmark pairs.

This can be useful for workflows such as:

- high-resolution sculpt to animation mesh
- proxy to production mesh
- corrective sculpt transfer
- related meshes with different topology
- transferring sculpted shape keys to lower-resolution meshes

## Basic Usage

### Manual Correspondences

1. Select the Source mesh.
2. Select the Target mesh.
3. Choose `Manual Correspondences`.
4. Enter Pairing Mode.
5. Click matching landmarks on the Source and Target.
6. Add enough correspondences to describe the important structure of the deformation.
7. Save the mapping if you expect to use these mesh topologies again.
8. Build/Fit the transfer map.
9. Put the Source into the expression or deformation you want.
10. Transfer the expression.
11. The result is created as a shape key on the Target.

Correspondence markers can be hidden after the mapping has been verified.

### Reusing a Saved Mapping

For another pair of characters using the same source and target topologies:

1. Select the new Source and Target.
2. Load the saved mapping.
3. Fit the mapping to the current meshes.
4. Apply the desired deformation to the Source.
5. Transfer it to the Target.

The correspondence work therefore does not need to be repeated for every character.

### Overlapping Surfaces

1. Position the Source and Target so their surfaces overlap appropriately.
2. Select `Overlapping Surfaces`.
3. Build the transfer map.
4. Apply the desired expression or deformation to the Source.
5. Transfer it to the Target.

No manually created correspondence set is required.

## What Can It Be Used For?

The extension was initially created for transferring AI-generated facial expressions onto Blender character meshes, but the transfer mechanism is general-purpose.

Possible uses include:

- facial expression retargeting
- transferring generated expressions to production characters
- transferring expressions between different character systems
- building reusable expression libraries
- transferring sculpted shape keys
- corrective shape transfer
- high-poly to low-poly deformation transfer
- low-poly proxy to production mesh transfer
- transferring useful deformations from otherwise unsuitable character meshes

If a mesh has a deformation you want and another mesh is the one you actually want to use, this extension is intended to help bridge the two.

## Installation

### Blender Extensions

Once available through the official Blender Extensions repository, install Mesh Expression Retargeter directly through Blender's Extensions interface.

Official listing:

https://extensions.blender.org/add-ons/mesh-expression-retargeter/

### Manual Installation

Download the extension ZIP from a release and install it in Blender using:

`Edit > Preferences > Add-ons > Install from Disk`

Select the ZIP file. It does not need to be extracted first.

## Requirements

Mesh Expression Retargeter is a Blender extension and requires a compatible current version of Blender.

It does not require the source and target meshes to share topology when using Manual Correspondences mode.

## Support and Bug Reports

Please use the GitHub Issues section of this repository for reproducible bugs, problems, and feature requests.

When reporting a transfer problem, information about the Blender version, source/target mesh setup, mapping method, and steps required to reproduce the problem will be useful.

## Project Origin

Mesh Expression Retargeter was originally developed by Vidyut Gore to create and animate characters for the science-fiction series *Abrahamha's Strays*.

It grew from that specific need into a general-purpose deformation retargeting tool and is released as free software in gratitude to Blender and the open-source tools that made the project possible.

More Blender tools, characters, and downloadable assets:

https://vidyut.net

## License

Mesh Expression Retargeter is free software licensed under the GNU General Public License v3.0 or later.

See `LICENSE` for the full license text.
