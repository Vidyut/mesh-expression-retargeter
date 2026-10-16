# Changelog

## 1.2.12

- Documentation: a Limitations section in the README describing when the
  transferred lips may want a small hand touch-up, and which defaults already
  guard against it.
- Records the inner-mouth mitigation. MPFB-style basemeshes carry a separate
  inner-lip sheet — the inner and outer lips sit about a millimetre apart at
  rest — so a nearest-surface bind can glue one sheet to the other and tear
  the mouth edge. Match Inner Mouth (1.2.10), Relax Mapped Motion (1.2.9) and
  Prefer Same-Facing Bindings (1.2.8) hold that back, and are all on by
  default. They are heuristics, not a fix: I have suggested to the MPFB
  add-on that it expose an explicit mouth-interior vertex group, which would
  let any retargeter separate the two sheets directly instead of inferring
  them from face orientation. Until that exists, the lips can still want a
  little touching up on some meshes.

## 1.2.11

- Never zero identity morphs. Transferring an expression used to zero *every*
  non-Basis shape key, which on a fitted MPFB character wipes its modeling
  morphs ('$md-...', 'head-oval', ...) and collapses the head — the same
  failure as the reset-loop in the demo script. The addon now distinguishes
  expression keys (its own transfers, tagged on creation, plus MPFB '!ex-*'
  face units) from identity keys, and only ever zeroes the former.
- Reset Target to Neutral button: the MPFB counterpart of resetting the GNM
  head. Calls MPFB's clear_applied_expressions (empties the expression stack
  and zeroes every '!ex-*' unit) and zeroes this addon's transferred keys,
  leaving identity morphs and the Basis intact — so you can return to the
  fitted face instead of regenerating it.
- Build Transfer Map is now deterministic: it binds against the Target's rest
  face. Any live expression keys are temporarily zeroed for the build and
  restored afterwards, so the same setup no longer binds differently
  depending on what was transferred before. Identity morphs are untouched.

## 1.2.10

- Match Inner Mouth (on by default): when the Source names an oral-cavity
  bag (the GNM head's `mouth_sock` group) and the Target is an MPFB
  basemesh, the target vertices that face into the mouth cavity are bound
  to that bag and the outer vertices to the skin, so the two sheets cannot
  bind to each other. On a clean GNM→MPFB mouth this trims the worst
  mouth-edge stretch (p99 6.1→5.6 mm, corner max 7.2→6.1 mm); its real job
  is to stop the outer lip gluing to the inner bag when a build does land
  cross-sheet. Automatic and mesh-agnostic: only activates when a source
  inner-mouth group exists, so other meshes are unaffected. Exposed as
  "Match Inner Mouth" in the Fit box.

## 1.2.9

- Relax Mapped Motion (on by default): the transferred motion is now
  smoothed across mapped target vertices (excluded groups pinned). Nearest-
  triangle binding is per-vertex, so two adjacent target vertices can bind
  to opposite sides of a sharp static/moving seam on the source (the inner
  mouth bag right beside the moving outer lip, mouth corners), which showed
  up as spikes/folds. Measured on the GNM→MPFB mouth: worst edge stretch
  drops from max 37.3 / p99 32.1 mm to max 7.5 / p99 5.7 mm while the mouth
  motion is preserved (mean 10.16→10.01 mm). The source field itself is
  smooth, so this fixes the binding, not the expression.
- Correspondence anchors still pin the neutral fit, but are no longer
  locked during Relax Mapped Motion. At a coincident double-sheet (mouth
  corners, eyelid folds) an anchor can land on the wrong sheet while its
  neighbours take the other; locking it there folded the skin. Motion
  anchors now relax with their neighbourhood (19 of 25 drift <1 mm; the two
  corner anchors drift ~10 mm to match their neighbourhood). Anchor-adjacent
  edge stretch at the corners falls from 12.1 mm to 2.9 mm.

## 1.2.8

- Prefer Same-Facing Bindings (on by default): when choosing the source
  surface for a target vertex, a candidate whose normal agrees with the
  vertex's own normal now wins over a merely-visible opposed-normal one.
  Fixes cross-sheet bindings where a closed mouth's inner and outer lip
  sheets (and other coincident double surfaces) are within a millimetre
  of each other at build time — the wrong sheet was glued, producing a
  walled-shut mouth and spikes once the surfaces separated. Mesh-agnostic:
  no extra markers, no assumption about the target. The global
  opposed-normal rejection (`release_across_gap`) stays off; this is the
  local, bounded version of the same signal.

## 1.2.7

- Smooth Unmapped Fill (on by default): after transfer, unmapped vertices
  near the mapped region get their motion inpainted from mapped neighbours
  (hop-limited mesh-Laplacian smoothing, pure numpy). Removes the
  frozen/moved seam artefacts — the sawtooth spikes where the halo-union
  boundary cuts through a moving lip. Distant geometry still stays neutral
  and excluded vertex groups stay frozen at Basis.
- Motion-outlier demotion (part of Smooth Unmapped Fill): mapped verts
  whose motion disagrees with their mapped neighbours (>2.5mm vs local
  mean) are demoted and filled instead. Catches bindings to static inner
  sheets — a lip vert bound across the mouth opening to the mouth bag
  (delta 0 among 10mm movers) is the sawtooth root cause. Anchor verts
  are never demoted.

## 1.2.6

- Local Anchor Influence (on by default): the landmark fit is a compact
  Wendland RBF — every anchor's influence falls to exactly zero at the
  Influence Radius (~4cm default), on top of a strain-free rigid
  translation that moves the source cluster onto the target (the source
  may sit anywhere in the scene). Lip markers move the lips and a small
  skin halo and never pull distant vertices, shrink the model, or strand
  separately-objected hair/eyes. Cover more of the face by adding markers,
  not by raising the radius. Untick to restore the original global
  thin-plate fit.
- Excluded Vertex Groups: comma-separated group names (text box in the Fit
  box) whose vertices are left unmapped. Prefilled with the MPFB basemesh
  helper set (HelperGeometry, JointCubes, tongue, teeth, eyes, eyelashes,
  hair/skirt/tights/genital helpers); unknown names are ignored, so the same
  box works on any other mesh. Replaces the unreliable geometric guess for
  tongue/teeth/eye/joint geometry.
- Included Vertex Groups: optional allow-list in the same style; when set,
  only listed groups are influenced and everything else stays at Basis.
  Exclude wins over include. (MPFB has no face-only group: 'body' is the
  whole visible skin, 'lips' just the lip band.)
- Bind Visible Surfaces Only (on by default): each target vertex binds to
  the nearest source triangle that faces it; backfaces are skipped in favour
  of the next-nearest front-facing candidate. Inner mouth bags, nostril bags
  and eyelid folds bind to their own visible sheet instead of gluing across
  a gap. No boxes, no freezing, works on any mesh.
- Protect Mouth Interior now defaults off: the normal heuristic froze real
  outer-lip skin near the rim (the lips group wraps the rim); keep it only
  as an experiment.

## 1.2.5

- Protect Mouth Interior (on by default): during landmark map building, target
  vertices near the landmark cluster whose normals face into the mouth cavity
  are frozen instead of bound. The inner mouth bag otherwise projects onto the
  moving outer lip across the gap and tears; freezing only the cavity keeps
  the outer lip's correct motion. Unlike Reject Across-Gap Bindings this has
  no collateral: outer lip skin (outward normals) and all anchors are
  unaffected.

## 1.2.4

- Mapping presets saved on a helper-stripped MPFB basemesh now load onto the
  full basemesh (and vice versa). MPFB exposes the same face vertices with or
  without helper geometry; the face indices are shared, so a count mismatch
  between the two variants is accepted with a warning as long as every stored
  target index exists on the selected mesh. Non-MPFB targets keep the strict
  topology check.

## 1.2.3

- Reject Across-Gap Bindings (on by default): when the surface a vertex binds
  to faces away from the vertex's own sheet (opposed normals), the vertex is
  left unmapped instead of bound to a facing sheet across a gap. This is what
  kept an MPFB target's upper inner lip stuck to the lower inner lip: the two
  sheets touch across the mouth gap and plain nearest-surface projection flips
  between them. Only same-facing, surface-connected projections move now.
  Disable the option to restore the original nearest-surface behaviour
  exactly.
- The build report states how many bindings were released.


## 1.2.2

- Transferring an expression now zeroes every other Target shape key first.
  Previously each transfer activated its own key at 1.0 while every earlier one
  stayed at 1.0 too, so the Target stacked all transferred expressions into a
  single mangled face. The panel reports how many keys it zeroed.
- Transfer All Shape Keys starts from a neutral Target as well, and restores
  the previous slider values if the batch fails part-way through.


## 1.2.1

- Correspondence markers now respect object visibility: hiding the Source mesh
  no longer leaves its markers drawn over the Target
- Correspondence count notice: above 50 pairs the panel adds a single quiet
  line, "More isn't necessarily better, unless you're capturing a specific
  deformation." It is a nudge, never a limit.


## 1.2.0

When MPFB is installed and Target is an MPFB basemesh, a transferred shape key
can be saved as a first-class MPFB expression in addition to the shape key.

- Save as MPFB Expression — projects the Target shape key onto MPFB's ARKit
  face units and writes an expression JSON into the MPFB expression library
- Save All as MPFB Expressions — same, one file per non-Basis Target shape key
- Optional immediate refresh of the MPFB expression library after saving
- Configurable expression filename, tags, and author
- Reports how much of the pose the face-unit vocabulary could not represent

The projection onto ARKit face units is a best fit, not a copy: an MPFB
expression stores only weights over those 52 base deformations. The Target
shape key remains the exact result.

Also:

- Remove Pair tool: click a correspondence marker to delete that pair and its
  partner, leaving the rest of the list untouched
- Correspondence mappings can ship read-only inside the add-on alongside the
  per-user library; bundled mappings cannot be overwritten or deleted


## 1.1.0

- Add Transfer All Shape Keys batch operation
- Preserve Source shape-key names on batch transfer
- Auto-increment duplicate Target names using Blender-style `.001`, `.002`, etc. suffixes
- Leave batch-created Target shape keys inactive after transfer


## 1.0.0

Initial public release.

- Manual Source/Target landmark correspondence workflow
- Continuous pairing mode with visible numbered correspondence markers
- Non-rigid RBF fitting between unrelated mesh shapes
- Reusable named topology-mapping library
- Direct overlapping-surface barycentric mapping mode
- Transfer current Source deformation to a Target shape key
- Evaluated-mesh support for deformed/generated target geometry
