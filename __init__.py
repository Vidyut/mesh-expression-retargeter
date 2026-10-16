# SPDX-License-Identifier: GPL-3.0-or-later
#
# Mesh Expression Retargeter
# Copyright (C) 2026 Vidyut Gore
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


import bpy
import json
import os
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras import view3d_utils
import gpu
from gpu_extras.batch import batch_for_shader
import blf


# -----------------------------------------------------------------------------
# Runtime cache
# -----------------------------------------------------------------------------

_CACHE = {
    "source_name": None,
    "target_name": None,
    "source_neutral_local": None,
    "source_triangles": None,
    "mapping": None,
    "rbf": None,
}

_DRAW_HANDLE = None


# Correspondence count is not a quality dial.  On one pair of meshes the fit
# residual was flat from 10 to 3500 pairs while solve time and spline overshoot
# both climbed: the landmark warp interpolates every pair exactly, so surplus
# pairs tend to add cost and oscillation rather than accuracy.  That will not
# hold for every mesh pair -- dense anchors earn their keep when the source
# deforms in ways a handful of landmarks cannot describe -- so this is only
# ever a nudge in the panel, never a limit.
PAIR_COUNT_NOTICE_AT = 50


def pair_count_notice(count):
    """A single quiet line, shown only when it applies."""
    if count <= PAIR_COUNT_NOTICE_AT:
        return None
    return (
        "More isn't necessarily better,",
        "unless you're capturing a specific deformation.",
    )


def ensure_draw_handler():
    """Create the 3D View correspondence overlay only when it is needed."""
    global _DRAW_HANDLE
    if _DRAW_HANDLE is None:
        _DRAW_HANDLE = bpy.types.SpaceView3D.draw_handler_add(
            draw_callback,
            (),
            'WINDOW',
            'POST_VIEW'
        )


def remove_draw_handler():
    """Remove the correspondence overlay handler if it exists."""
    global _DRAW_HANDLE
    if _DRAW_HANDLE is not None:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(
                _DRAW_HANDLE,
                'WINDOW'
            )
        except Exception:
            # Blender may already have destroyed the region during shutdown,
            # file/window changes, or add-on reload.
            pass
        _DRAW_HANDLE = None


def update_show_correspondences(self, context):
    """
    Tie the draw handler to correspondence visibility instead of keeping
    a global handler alive for the lifetime of the add-on.
    """
    if self.show_correspondences:
        ensure_draw_handler()
    elif not self.pairing_mode:
        remove_draw_handler()

    if context and context.area:
        try:
            context.area.tag_redraw()
        except Exception:
            pass


def update_mapping_method(self, context):
    """
    Leaving Manual Correspondences mode ends temporary pairing/display state.
    """
    if self.mapping_method != "LANDMARKS":
        self.pairing_mode = False
        self.show_correspondences = False
        remove_draw_handler()

    if context and context.area:
        try:
            context.area.tag_redraw()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# MPFB expression export (optional integration)
# -----------------------------------------------------------------------------

_MPFB_ABSENT = "absent"
_MPFB_UNINITIALIZED = "not_initialized"
_MPFB_OK = "ok"

_MPFB_STATE_MESSAGES = {
    _MPFB_ABSENT: (
        "MPFB is not loaded in this Blender: no loaded module name ends in "
        "'mpfb'. Install and enable the MPFB extension."
    ),
    _MPFB_UNINITIALIZED: (
        "MPFB is loaded but did not finish initializing, so none of its "
        "services are available. Check the console for errors raised during "
        "MPFB registration."
    ),
    _MPFB_OK: None,
}

# Which MakeHuman-space component feeds each Blender axis, and its sign.
# MPFB's .target files store offsets in MakeHuman's Y-up, Z-forward convention
# while Blender shape keys are Z-up, so the swap is (x, -z, y). The basemesh's
# local space is MakeHuman units times MPFB_TARGET_SCALE. Both values were
# verified against a freshly created Human: applying them reproduces the
# shape-key delta MPFB itself loads, to a residual of ~3e-6.
MPFB_TARGET_AXIS = (0, 2, 1)
MPFB_TARGET_SIGN = (1.0, -1.0, 1.0)
MPFB_TARGET_SCALE = 0.1

_FACEUNIT_BASIS = {
    "vectors": None,
    "units": (),
    "vertex_count": 0,
    "active": None,
}


def _wrap_label(text, width=46):
    """Break a sentence into lines short enough for a panel label."""
    words = str(text).split()
    lines = []
    current = ""

    for word in words:
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = word if not current else current + " " + word

    if current:
        lines.append(current)

    return lines


def resolve_mpfb():
    """
    Find MPFB the way a companion add-on is meant to.

    Scans sys.modules for a module whose name ends in "mpfb" rather than
    hardcoding a package path, because the name depends on which extension
    repository the user installed MPFB from -- bl_ext.user_default.mpfb,
    bl_ext.some_repo.mpfb, or a plain "mpfb" when run from source. Then reads
    the SERVICES dict MPFB publishes on the last line of its own register().

    Deliberately uncached. MPFB can be disabled and re-enabled inside one
    Blender session, and its unregister() never resets
    MPFB_CONTEXTUAL_INFORMATION, so a cached services dict is exactly the
    stale state that is hardest to diagnose. The scan is a pass over
    sys.modules and two getattr calls; there is nothing here worth saving.
    """
    import sys

    candidates = sorted(name for name in list(sys.modules) if name.endswith("mpfb"))
    if not candidates:
        return {"state": _MPFB_ABSENT, "services": None, "candidates": candidates}

    module = sys.modules[candidates[0]]
    info = getattr(module, "MPFB_CONTEXTUAL_INFORMATION", None)
    if not isinstance(info, dict):
        # Still None: register() never ran, or raised before it got this far.
        return {"state": _MPFB_UNINITIALIZED, "services": None, "candidates": candidates}

    services = info.get("SERVICES")
    if not isinstance(services, dict):
        # register() ran but did not reach its last line.
        return {"state": _MPFB_UNINITIALIZED, "services": None, "candidates": candidates}

    return {"state": _MPFB_OK, "services": services, "candidates": candidates}


def mpfb_service(name):
    """One MPFB service class by name, or None when MPFB is unusable."""
    resolution = resolve_mpfb()
    if resolution["state"] != _MPFB_OK:
        return None
    service = resolution["services"].get(name)
    return service if service is not None else None


def mpfb_available():
    """
    True when MPFB is loaded, initialized, and exposes what this needs.

    This add-on must keep working on its own, so MPFB is never imported at
    module level and is looked up through resolve_mpfb() instead.
    """
    return (
        mpfb_service("FaceService") is not None
        and mpfb_service("TargetService") is not None
        and mpfb_service("ObjectService") is not None
    )


def mpfb_detect():
    """Why MPFB expression export is not on offer, or None when it is.

    Reports the state in the user's words rather than a bare failure, so the
    panel can say which of the two is happening: MPFB is not installed, or it
    is installed but did not finish registering.
    """
    resolution = resolve_mpfb()
    if resolution["state"] == _MPFB_OK:
        return None
    return _MPFB_STATE_MESSAGES[resolution["state"]]


def faceunit_names():
    """The 52 ARKit face-unit names, sorted, or () when MPFB is unusable.

    ARKIT_FACEUNITS is a module-level constant of faceservice, not an
    attribute of the FaceService class, so the module has to be reached
    through the class rather than imported by a hardcoded path.
    """
    face_service = mpfb_service("FaceService")
    if face_service is None:
        return ()

    import importlib

    try:
        face_module = importlib.import_module(face_service.__module__)
        return tuple(sorted(face_module.ARKIT_FACEUNITS))
    except Exception as exc:
        print("[mesh_expression_retargeter] could not read ARKIT_FACEUNITS: " + repr(exc))
        return ()



def target_is_mpfb_basemesh(target):
    """True when Target is an MPFB basemesh, which is what expressions need."""
    object_service = mpfb_service("ObjectService")
    if target is None or object_service is None:
        return False
    try:
        return bool(object_service.object_is_basemesh(target))
    except Exception:
        return False


def mpfb_mapping_can_ignore_vertex_count(target, pairs):
    """Whether an MPFB helper-geometry count mismatch is harmless.

    MPFB can expose the same basemesh with or without helper geometry. The
    face vertices retain their indices, so a landmark preset remains valid as
    long as every stored target index exists in the selected variant.
    """
    if not target_is_mpfb_basemesh(target):
        return False

    target_count = len(target.data.vertices)
    try:
        return all(
            0 <= int(pair["target_vertex"]) < target_count
            for pair in pairs
        )
    except (KeyError, TypeError, ValueError):
        return False


def load_faceunit_basis():
    """
    Parse MPFB's ARKit face-unit .target files into a (verts, 3, units) array.

    The files are sparse "index dx dy dz" text, so one dense array is built and
    cached for the rest of the session. Returns (vectors, units, vertex_count),
    or (None, (), 0) when nothing could be read.
    """
    if _FACEUNIT_BASIS["vectors"] is not None:
        return (
            _FACEUNIT_BASIS["vectors"],
            _FACEUNIT_BASIS["units"],
            _FACEUNIT_BASIS["vertex_count"],
        )

    if not mpfb_available():
        return None, (), 0

    units = list(faceunit_names())
    if not units:
        return None, (), 0

    target_service = mpfb_service("TargetService")
    if target_service is None:
        return None, (), 0

    columns = []
    vertex_count = 0

    for unit in units:
        column = {}
        path = target_service.target_full_path(unit)
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as handle:
                    for line in handle:
                        parts = line.split()
                        if len(parts) != 4:
                            continue
                        try:
                            index = int(parts[0])
                            column[index] = (
                                float(parts[1]),
                                float(parts[2]),
                                float(parts[3]),
                            )
                        except ValueError:
                            continue
            except OSError:
                column = {}
        columns.append(column)
        if column:
            vertex_count = max(vertex_count, max(column) + 1)

    if vertex_count == 0:
        return None, (), 0

    vectors = np.zeros((vertex_count, 3, len(units)), dtype=np.float64)
    axis = MPFB_TARGET_AXIS
    signs = MPFB_TARGET_SIGN
    scale = MPFB_TARGET_SCALE
    for col_index, column in enumerate(columns):
        for index, (dx, dy, dz) in column.items():
            source = (dx, dy, dz)
            vectors[index, 0, col_index] = source[axis[0]] * signs[0] * scale
            vectors[index, 1, col_index] = source[axis[1]] * signs[1] * scale
            vectors[index, 2, col_index] = source[axis[2]] * signs[2] * scale

    _FACEUNIT_BASIS["vectors"] = vectors
    _FACEUNIT_BASIS["units"] = tuple(units)
    _FACEUNIT_BASIS["vertex_count"] = vertex_count
    _FACEUNIT_BASIS["active"] = None

    return vectors, tuple(units), vertex_count


def faceunit_active_vertices(max_vertices=512):
    """
    Indices of the vertices that carry the face units, best first.

    Face units only deform the head, and the files never touch a vertex the
    units leave still. Fitting on the strongest movers is both far cheaper
    than the whole head and better conditioned: the columns of the design
    matrix actually differ there, instead of being thousands of identical
    zero rows. A few hundred vertices is plenty for 52 weights; the hard
    floor is 52, since each vertex contributes three equations.
    """
    basis, _units, vertex_count = load_faceunit_basis()
    if basis is None:
        return None

    if _FACEUNIT_BASIS["active"] is not None:
        return _FACEUNIT_BASIS["active"]

    energy = np.einsum("vci,vci->v", basis, basis)
    movers = np.flatnonzero(energy > 0.0)
    if movers.size == 0:
        return None

    limit = max(52, min(int(max_vertices), movers.size))
    order = movers[np.argsort(energy[movers])[::-1][:limit]]
    order.sort()

    _FACEUNIT_BASIS["active"] = order
    return order


def fit_faceunits(displacement, *, iterations=3, max_vertices=512):
    """
    Express a per-vertex displacement as weights over MPFB's ARKit face units.

    An MPFB expression stores nothing but weights over these 52 base
    deformations, so a transferred shape key has to be projected onto them.
    The projection is lossy by construction: the residual says how well the
    pose is expressible in this vocabulary, not how faithful it is to the
    source mesh.

    Solves min ||U w - d||^2 subject to 0 <= w <= 1 with least squares followed
    by clamping and re-solving on the surviving support. Returns
    (weights_by_unit, relative_residual).
    """
    basis, units, vertex_count = load_faceunit_basis()
    if basis is None:
        raise RuntimeError("MPFB face-unit targets could not be loaded")

    active_vertices = faceunit_active_vertices(max_vertices)
    if active_vertices is None or len(active_vertices) == 0:
        raise RuntimeError("There is no displacement to fit")

    available = min(len(displacement), vertex_count)
    use = active_vertices[active_vertices < available]
    if use.size == 0:
        raise RuntimeError("There is no displacement to fit")

    design = basis[use].reshape(use.size * 3, len(units))
    rhs = np.asarray(displacement)[use].reshape(use.size * 3).astype(np.float64)

    weights = np.zeros(len(units), dtype=np.float64)
    active = np.arange(len(units))

    for _ in range(max(1, int(iterations))):
        if active.size == 0:
            break
        solved, *_ = np.linalg.lstsq(design[:, active], rhs, rcond=None)
        candidate = np.zeros(len(units), dtype=np.float64)
        candidate[active] = np.clip(solved, 0.0, 1.0)
        weights = candidate
        surviving = np.flatnonzero(weights > 1e-9)
        if surviving.size == active.size:
            break
        active = surviving

    residual = float(np.linalg.norm(design @ weights - rhs))
    total = float(np.linalg.norm(rhs))
    relative = residual / total if total > 0.0 else 0.0

    face_units = {
        unit: round(float(weights[index]), 4)
        for index, unit in enumerate(units)
        if weights[index] > 0.0
    }
    return face_units, relative


def shape_key_displacement(target, key):
    """Per-vertex displacement of a Target shape key relative to its Basis."""
    blocks = target.data.shape_keys.key_blocks
    basis = blocks["Basis"]
    return [key.data[i].co - basis.data[i].co for i in range(len(key.data))]


def save_shape_key_as_mpfb_expression(target, key, filename, tags, author, refresh):
    """
    Project one Target shape key onto the face units and write an MPFB
    expression JSON. Returns (absolute_path, unit_count, relative_residual).
    """
    face_units, relative = fit_faceunits(shape_key_displacement(target, key))
    if not face_units:
        raise RuntimeError(
            "The deformation did not map onto any MPFB face unit "
            "(residual %.0f%%)" % (relative * 100.0)
        )

    stem = os.path.splitext(os.path.basename(filename))[0].strip()
    if not stem:
        stem = key.name

    face_service = mpfb_service("FaceService")
    if face_service is None:
        raise RuntimeError("MPFB is not available, so the expression cannot be written")

    path = face_service.save_expression(stem + ".json", face_units, {
        "name": stem.replace("_", " "),
        "description": "Transferred from a source mesh by Mesh Expression Retargeter.",
        "tags": tags,
        "author": author,
    })

    if refresh:
        try:
            from bl_ext.blender_org.mpfb.ui.apply_assets.useexpression import (
                refresh_expression_sliders,
            )
            refresh_expression_sliders(force_redraw=False)
        except Exception:
            # The file is already on disk; the library can be refreshed manually.
            pass

    return path, len(face_units), relative


# -----------------------------------------------------------------------------
# Mapping preset library
# -----------------------------------------------------------------------------



def mapping_library_dir():
    """Writable per-user data directory for the official Blender Extension."""
    root = bpy.utils.extension_path_user(__package__, create=True)
    path = os.path.join(root, "mappings")
    os.makedirs(path, exist_ok=True)
    return path


def bundled_mapping_dir():
    """Read-only correspondence mappings shipped inside the add-on."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "mappings")


def _list_mapping_files(folder):
    try:
        return sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith(".json")
        )
    except Exception:
        return []


def sanitize_mapping_name(name):
    safe = "".join(
        c if c.isalnum() or c in (" ", "-", "_", ".") else "_"
        for c in name.strip()
    )
    safe = safe.strip(" .")
    return safe or "Source to Target"


def mapping_preset_items(self, context):
    items = []
    index = 0

    for origin, folder in (
        ("bundled", bundled_mapping_dir()),
        ("user", mapping_library_dir()),
    ):
        for filename in _list_mapping_files(folder):
            stem = os.path.splitext(filename)[0]
            label = stem if origin == "user" else f"{stem} (bundled)"
            items.append((f"{origin}:{filename}", label, filename, index))
            index += 1

    if not items:
        items.append(("__NONE__", "(no saved mappings)", "", 0))

    return items


def mapping_origin(identifier):
    """Return 'bundled' or 'user' for a preset identifier."""
    origin = str(identifier).partition(":")[0]
    return origin if origin in ("bundled", "user") else "user"


def mapping_filename(identifier):
    """Strip the origin prefix from a preset identifier."""
    origin, separator, filename = str(identifier).partition(":")
    return filename if separator and origin in ("bundled", "user") else str(identifier)


def mapping_is_bundled(identifier):
    return mapping_origin(identifier) == "bundled"


def mapping_filepath_from_name(name):
    safe = sanitize_mapping_name(name)
    return os.path.join(mapping_library_dir(), safe + ".json")


def find_mapping_file(identifier):
    origin = mapping_origin(identifier)
    folder = bundled_mapping_dir() if origin == "bundled" else mapping_library_dir()
    path = os.path.join(folder, mapping_filename(identifier))
    return path if os.path.isfile(path) else None


def normalize_mapping_data(data):
    if data.get("format") == "mesh_expression_retargeter_mapping":
        return data
    return None


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def triangulate_polygons(mesh):
    """
    Return triangle tuples using simple fan triangulation for polygons with
    more than three vertices. The source topology must remain stable.
    """
    tris = []
    for poly in mesh.polygons:
        verts = list(poly.vertices)
        if len(verts) == 3:
            tris.append(tuple(verts))
        elif len(verts) > 3:
            root = verts[0]
            for i in range(1, len(verts) - 1):
                tris.append((root, verts[i], verts[i + 1]))
    return tris


def barycentric_coords(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a

    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-14:
        return (1.0, 0.0, 0.0)

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v
    u -= w
    return (u, v, w)


def ensure_basis(obj):
    if obj.data.shape_keys is None:
        obj.shape_key_add(name="Basis", from_mix=False)


def zero_other_shape_keys(target, keep=None):
    """
    Zero every Target shape key slider except Basis and `keep`.

    A single transfer activates its own key at 1.0. Left alone, every key
    previously transferred stays at 1.0 too, so the Target stacks every
    expression it has ever received into one mangled face.
    """
    shape_keys = target.data.shape_keys
    if shape_keys is None:
        return 0
    cleared = 0
    for key in shape_keys.key_blocks:
        # RNA wrappers are not interned: `is` would miss the kept key, so
        # compare pointers with == like the reference-key check above.
        if key == shape_keys.reference_key or key == keep:
            continue
        if key.value != 0.0:
            key.value = 0.0
            cleared += 1
    return cleared


TRANSFERRED_KEY_PROP = "mer_transferred_expression_keys"


def transferred_key_names(target):
    raw = target.get(TRANSFERRED_KEY_PROP)
    if not raw:
        return set()
    try:
        return set(json.loads(raw))
    except Exception:
        return set()


def mark_transferred_key(target, key):
    """Record an addon-created expression key name on the Target object.

    ShapeKey is not a Blender ID, so a custom property cannot live on the key
    itself; the names are stored on the Target (an ID) instead.
    """
    names = transferred_key_names(target)
    names.add(key.name)
    try:
        target[TRANSFERRED_KEY_PROP] = json.dumps(sorted(names))
    except Exception:
        pass


def is_expression_key(target, key):
    """
    True for shape keys that are expressions, not identity.

    Expressions are MPFB face-unit keys ('!ex-*') and this addon's own
    transfers (recorded on the Target). Modeling/identity morphs (MPFB
    '$md-...', user morphs like 'head-oval') are NOT expressions: zeroing them
    collapses the character to its raw basis, which for a fitted MPFB head
    means regenerating it.
    """
    if key.name.startswith("!ex-"):
        return True
    return key.name in transferred_key_names(target)


def expression_key_blocks(target):
    shape_keys = target.data.shape_keys
    if shape_keys is None or not shape_keys.key_blocks:
        return []
    reference = shape_keys.reference_key
    return [
        key for key in shape_keys.key_blocks
        if key != reference and is_expression_key(target, key)
    ]


def zero_expression_keys(target, keep=None):
    """Zero only expression keys; identity morphs and Basis are left alone."""
    cleared = 0
    for key in expression_key_blocks(target):
        if key == keep:
            continue
        if key.value != 0.0:
            key.value = 0.0
            cleared += 1
    return cleared


def capture_expression_values(target):
    return [(key, key.value) for key in expression_key_blocks(target)]


def restore_key_values(saved):
    for key, value in saved:
        try:
            key.value = value
        except Exception:
            pass


def reset_to_neutral(target):
    """
    Return an MPFB target to its rest face without collapsing it.

    The MPFB equivalent of the GNM head's reset: MPFB's own
    clear_applied_expressions empties the expression stack and zeroes every
    '!ex-*' face unit; this addon's transferred expression keys are zeroed too.
    Identity/modeling morphs and the Basis are deliberately untouched, so the
    fitted head shape survives instead of needing a regenerate.
    """
    face_service = mpfb_service("FaceService")
    if face_service is not None:
        try:
            face_service.clear_applied_expressions(target)
        except Exception:
            try:
                face_service.clear_expression(target)
            except Exception:
                pass
    return zero_expression_keys(target)


def evaluated_object(context, obj):
    depsgraph = context.evaluated_depsgraph_get()
    return obj.evaluated_get(depsgraph)


def raycast_object_nearest_vertex(context, event, obj):
    """
    Raycast against evaluated geometry and snap the click to the nearest vertex
    on the hit polygon. Stable vertex indices are stored in the mapping.
    """
    region = context.region
    rv3d = context.region_data
    coord = (event.mouse_region_x, event.mouse_region_y)

    origin_world = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
    direction_world = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)

    depsgraph = context.evaluated_depsgraph_get()
    eobj = obj.evaluated_get(depsgraph)

    inv = eobj.matrix_world.inverted()
    origin_obj = inv @ origin_world
    direction_obj = inv.to_3x3() @ direction_world
    direction_obj.normalize()

    hit, loc_obj, normal, poly_index = eobj.ray_cast(
        origin_obj,
        direction_obj,
        depsgraph=depsgraph
    )
    if not hit:
        return None

    poly = eobj.data.polygons[poly_index]
    nearest = min(
        poly.vertices,
        key=lambda vi: (eobj.data.vertices[vi].co - loc_obj).length_squared
    )

    world = eobj.matrix_world @ eobj.data.vertices[nearest].co
    return nearest, world


# -----------------------------------------------------------------------------
# RBF warp
# -----------------------------------------------------------------------------

class RBFWarp:
    """
    3D polyharmonic RBF warp with phi(r) = r plus affine terms.
    Correspondence points constrain a non-rigid source-to-target fit.
    """

    def __init__(self, src, dst, regularization=1e-8):
        self.src = np.asarray(src, dtype=np.float64)
        self.dst = np.asarray(dst, dtype=np.float64)

        n = len(self.src)
        if n < 4:
            raise ValueError("At least 4 correspondence pairs are required.")

        K = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            diff = self.src[i][None, :] - self.src
            K[i, :] = np.linalg.norm(diff, axis=1)

        K += np.eye(n) * regularization

        P = np.ones((n, 4), dtype=np.float64)
        P[:, 1:] = self.src

        L = np.zeros((n + 4, n + 4), dtype=np.float64)
        L[:n, :n] = K
        L[:n, n:] = P
        L[n:, :n] = P.T

        Y = np.zeros((n + 4, 3), dtype=np.float64)
        Y[:n, :] = self.dst

        try:
            params = np.linalg.solve(L, Y)
        except np.linalg.LinAlgError:
            params = np.linalg.lstsq(L, Y, rcond=None)[0]

        self.w = params[:n, :]
        self.a = params[n:, :]

    def transform_array(self, pts):
        pts = np.asarray(pts, dtype=np.float64)
        diff = pts[:, None, :] - self.src[None, :, :]
        R = np.linalg.norm(diff, axis=2)

        P = np.ones((len(pts), 4), dtype=np.float64)
        P[:, 1:] = pts

        return R @ self.w + P @ self.a


class RBFLocalWarp:
    """
    Compact-support RBF: anchor influence falls off to exactly zero at
    influence_radius (Wendland C2 kernel). Lip anchors move the lips and a
    small halo of surrounding skin; distant vertices are not pulled at all.
    Displacement formulation: d(x) = sum w_i * phi(|x - s_i|), d == 0 beyond
    the radius, anchors satisfied exactly.
    """

    def __init__(self, src, dst, influence_radius, regularization=1e-8):
        self.src = np.asarray(src, dtype=np.float64)
        self.dst = np.asarray(dst, dtype=np.float64)
        self.radius = float(influence_radius)

        # Strain-free pre-alignment: a pure translation moves the source
        # cluster onto the target cluster (source may sit anywhere in the
        # scene); the compact residual below only carries the shape change.
        self.translation = self.dst.mean(axis=0) - self.src.mean(axis=0)
        src_a = self.src + self.translation

        n = len(self.src)
        if n < 4:
            raise ValueError("At least 4 correspondence pairs are required.")

        diff = src_a[None, :, :] - src_a[:, None, :]
        rho = np.linalg.norm(diff, axis=2) / self.radius
        K = self._wendland(rho)
        K += np.eye(n) * regularization

        delta = self.dst - src_a

        try:
            self.w = np.linalg.solve(K, delta)
        except np.linalg.LinAlgError:
            self.w = np.linalg.lstsq(K, delta, rcond=None)[0]

    @staticmethod
    def _wendland(rho):
        """Wendland C2: (1-rho)^4 * (4*rho+1) inside the radius, else 0."""
        r = np.clip(rho, 0.0, 1.0)
        return np.where(rho < 1.0, (1.0 - r) ** 4 * (4.0 * r + 1.0), 0.0)

    def transform_array(self, pts):
        pts = np.asarray(pts, dtype=np.float64)
        pts_a = pts + self.translation
        diff = pts_a[:, None, :] - self.src[None, :, :] - self.translation[None, None, :]
        rho = np.linalg.norm(diff, axis=2) / self.radius
        phi = self._wendland(rho)
        return pts_a + phi @ self.w


# -----------------------------------------------------------------------------
# Persistent scene data
# -----------------------------------------------------------------------------

class MER_Correspondence(bpy.types.PropertyGroup):
    source_vertex: bpy.props.IntProperty(default=-1)
    target_vertex: bpy.props.IntProperty(default=-1)


class MER_Settings(bpy.types.PropertyGroup):
    mapping_method: bpy.props.EnumProperty(
        name="Mapping Method",
        description="How Source and Target correspondence is established",
        items=[
            (
                "LANDMARKS",
                "Manual Correspondences",
                "Use manually paired Source/Target vertices and an RBF fit. Meshes may be anywhere."
            ),
            (
                "OVERLAP",
                "Overlapping Surfaces",
                "Use nearest-surface correspondence directly. Source and Target must already overlap spatially."
            ),
        ],
        default="LANDMARKS",
        update=update_mapping_method
    )

    source_obj: bpy.props.PointerProperty(
        name="Source Mesh",
        description="Neutral/expression donor mesh",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH'
    )

    target_obj: bpy.props.PointerProperty(
        name="Target Mesh",
        description="Mesh that will receive the transferred expression",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'MESH'
    )

    shapekey_name: bpy.props.StringProperty(
        name="Shape Key",
        default="Transferred_Expression"
    )

    expression_filename: bpy.props.StringProperty(
        name="Expression File",
        description="Filename for the saved MPFB expression. Blank uses the shape key name.",
        default=""
    )

    expression_tags: bpy.props.StringProperty(
        name="Tags",
        description="Comma-separated tags stored in the expression file",
        default="retargeted"
    )

    expression_author: bpy.props.StringProperty(
        name="Author",
        description="Author stored in the expression file",
        default=""
    )

    refresh_library: bpy.props.BoolProperty(
        name="Add to MPFB Library",
        description="Refresh the MPFB expression library so the saved file appears immediately",
        default=True
    )

    max_distance: bpy.props.FloatProperty(
        name="Max Fit Distance",
        description="Ignore target vertices farther than this from the fitted source surface. 0 means unlimited.",
        default=0.0,
        min=0.0,
        unit='LENGTH'
    )

    release_across_gap: bpy.props.BoolProperty(
        name="Reject Across-Gap Bindings",
        description=(
            "Leave target vertices unmapped when the surface they bind to faces "
            "away from their own sheet (opposed normals). This is what kept an "
            "MPFB target's upper inner lip stuck to the lower inner lip: the two "
            "sheets touch across the mouth gap and plain nearest-surface "
            "projection flips between them. Only same-facing, surface-connected "
            "projections move; disable to restore plain nearest-surface behaviour"
        ),
        default=True
    )

    protect_mouth_interior: bpy.props.BoolProperty(
        name="Protect Mouth Interior",
        description=(
            "Freeze target vertices inside the mouth cavity (their normals "
            "face into the cavity, unlike the outer lip skin). Heuristic: "
            "can freeze real lip skin near the rim; prefer Excluded Vertex "
            "Groups and Bind Visible Surfaces Only when possible"
        ),
        default=False
    )

    bind_visible_only: bpy.props.BoolProperty(
        name="Bind Visible Surfaces Only",
        description=(
            "Bind each target vertex to the nearest source surface that "
            "faces it (front-facing = visible from that point); if the "
            "closest triangle is a backface, the next-nearest front-facing "
            "one is used instead. Inner mouth bags, nostril bags and eyelid "
            "folds then bind to their own visible sheet instead of gluing "
            "to a facing surface across the gap"
        ),
        default=True
    )

    bind_same_facing: bpy.props.BoolProperty(
        name="Prefer Same-Facing Bindings",
        description=(
            "Among candidate surfaces facing the target vertex, prefer the "
            "one whose normal agrees with the vertex's own normal. Keeps a "
            "closed mouth's inner and outer sheets (and other coincident "
            "double surfaces) from binding to each other across the gap, "
            "without the over-culling of a global opposed-normal rejection"
        ),
        default=True
    )

    match_mouth_sock: bpy.props.BoolProperty(
        name="Match Inner Mouth",
        description=(
            "When the Source names an inner-mouth bag (GNM head's "
            "'mouth_sock') and the Target is an MPFB basemesh, bind the "
            "target's cavity-facing vertices to that bag and its outer "
            "vertices to the skin. Keeps the outer lip from gluing to the "
            "inner bag and vice versa, which otherwise folds the mouth "
            "corners. Ignored when the Source has no inner-mouth group"
        ),
        default=True
    )

    exclude_vertex_groups: bpy.props.StringProperty(
        name="Excluded Vertex Groups",
        description=(
            "Comma-separated vertex group names to exclude from the "
            "transfer map (left unmapped / frozen at Basis). Prefilled "
            "with MPFB basemesh helper groups (tongue, teeth, eyes, "
            "eyelashes, joint cubes, hair helpers); unknown names are "
            "ignored, so edit the list for other meshes or clear it"
        ),
        default=(
            "HelperGeometry, JointCubes, helper-tongue, "
            "helper-upper-teeth, helper-lower-teeth, "
            "helper-l-eye, helper-r-eye, "
            "helper-l-eyelashes-1, helper-l-eyelashes-2, "
            "helper-r-eyelashes-1, helper-r-eyelashes-2, "
            "helper-hair, helper-tights, helper-skirt, helper-genital"
        )
    )

    include_vertex_groups: bpy.props.StringProperty(
        name="Included Vertex Groups",
        description=(
            "Comma-separated vertex group names to influence: when set, "
            "only vertices in these groups are mapped and everything else "
            "stays at Basis. Empty = all vertices (minus Excluded Groups). "
            "Unknown names are ignored"
        ),
        default=""
    )

    local_anchor_influence: bpy.props.BoolProperty(
        name="Local Anchor Influence",
        description=(
            "Anchors influence only within the Influence Radius (compact "
            "Wendland falloff to zero): lip markers move the lips and a "
            "small halo of skin, and never pull distant vertices. Beyond "
            "the radius the mesh keeps its neutral shape"
        ),
        default=True
    )

    influence_radius: bpy.props.FloatProperty(
        name="Influence Radius",
        description=(
            "Distance over which an anchor's influence falls to zero. "
            "~4cm covers lips plus a natural skin halo around them. "
            "Add more markers to widen coverage instead of raising this"
        ),
        default=0.04,
        min=0.005,
        soft_max=0.2,
        unit='LENGTH'
    )

    displacement_scale: bpy.props.FloatProperty(
        name="Expression Scale",
        description="Scale the transferred deformation",
        default=1.0,
        soft_min=0.0,
        soft_max=2.0
    )

    harmonic_inpaint: bpy.props.BoolProperty(
        name="Smooth Unmapped Fill",
        description=(
            "Fill motion for unmapped vertices from their mapped "
            "neighbours (mesh-Laplacian smoothing, fading out over a "
            "dozen edge rings). Removes frozen/moved seam artefacts at "
            "the edge of the mapped region; distant geometry stays "
            "neutral and excluded vertex groups stay frozen"
        ),
        default=True
    )

    smooth_mapped_motion: bpy.props.BoolProperty(
        name="Relax Mapped Motion",
        description=(
            "Smooth the transferred motion across mapped vertices "
            "(anchors and excluded groups pinned). Nearest-triangle binding "
            "is per-vertex, so adjacent target verts can bind to opposite "
            "sides of a sharp static/moving seam on the source (inner mouth "
            "bag beside the moving outer lip), producing spikes. This relaxes "
            "those binding discontinuities without softening the expression"
        ),
        default=True
    )

    rbf_regularization: bpy.props.FloatProperty(
        name="Fit Smoothness",
        description="Small regularization value for the landmark warp. Increase slightly only if fitting is unstable.",
        default=1e-8,
        min=0.0,
        max=1e-2,
        precision=8
    )

    show_correspondences: bpy.props.BoolProperty(
        name="Show Correspondences",
        default=False,
        update=update_show_correspondences
    )

    pairing_mode: bpy.props.BoolProperty(
        name="Pairing Mode",
        default=False
    )

    remove_mode: bpy.props.BoolProperty(
        name="Remove Mode",
        description="Click correspondence markers to delete their pairs",
        default=False
    )

    mapping_name: bpy.props.StringProperty(
        name="Mapping Name",
        default="Source to Target"
    )

    mapping_preset: bpy.props.EnumProperty(
        name="Saved Mapping",
        items=mapping_preset_items
    )

    correspondences: bpy.props.CollectionProperty(type=MER_Correspondence)


# -----------------------------------------------------------------------------
# Pairing mode
# -----------------------------------------------------------------------------

def remove_correspondence_near_click(context, event, settings, threshold=28.0):
    """
    Delete the correspondence whose marker sits closest to the click.

    Matching is done in screen space rather than by raycast so that clicking a
    drawn marker works even where the marker is not exactly on a hit polygon.
    Returns True when a pair was removed.
    """
    region = context.region
    rv3d = context.region_data

    if region is None or rv3d is None:
        return False

    click = Vector((event.mouse_region_x, event.mouse_region_y))

    best_index = None
    best_distance = float(threshold)

    for index, pair in enumerate(settings.correspondences):
        for world in pair_world_points(context, settings, pair):
            if world is None:
                continue
            co2d = view3d_utils.location_3d_to_region_2d(region, rv3d, world)
            if co2d is None:
                continue
            distance = (Vector(co2d) - click).length
            if distance < best_distance:
                best_distance = distance
                best_index = index

    if best_index is None:
        return False

    settings.correspondences.remove(best_index)
    _CACHE["mapping"] = None
    _CACHE["rbf"] = None
    return True


class MER_OT_pairing_mode(bpy.types.Operator):
    bl_idname = "mer.pairing_mode"
    bl_label = "Pairing Mode"
    bl_description = "Continuously click Source then Target to create correspondence pairs"
    bl_options = {'REGISTER'}

    _stage = 0
    _source_vertex = None

    remove_only: bpy.props.BoolProperty(
        name="Remove Only",
        description="Delete correspondence pairs instead of creating them",
        options={'SKIP_SAVE', 'HIDDEN'}
    )

    def invoke(self, context, event):
        s = context.scene.mer_settings

        if not s.source_obj or not s.target_obj:
            self.report({'ERROR'}, "Choose both Source and Target meshes.")
            return {'CANCELLED'}

        if context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Run pairing mode from the 3D View.")
            return {'CANCELLED'}

        if self.remove_only:
            self._stage = 0
            self._source_vertex = None
            if s.pairing_mode:
                # The already-running modal picks the flag up on the next click.
                s.remove_mode = True
                self.report({'INFO'}, "Remove mode on: click a marker to delete its pair.")
                context.area.tag_redraw()
                return {'FINISHED'}

            s.pairing_mode = True
            s.show_correspondences = True
            s.remove_mode = True
            ensure_draw_handler()
            context.window_manager.modal_handler_add(self)
            self.report(
                {'INFO'},
                "Remove mode: click a correspondence marker to delete that pair. "
                "Esc/right-click exits."
            )
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        s.pairing_mode = True
        s.show_correspondences = True
        s.remove_mode = False
        ensure_draw_handler()
        self._stage = 0
        self._source_vertex = None

        context.window_manager.modal_handler_add(self)
        self.report(
            {'INFO'},
            "Pairing mode: click Source, then matching Target point; repeat. Esc/right-click exits."
        )
        context.area.tag_redraw()
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        s = context.scene.mer_settings

        if not s.pairing_mode or event.type in {'ESC', 'RIGHTMOUSE'}:
            s.pairing_mode = False
            s.remove_mode = False
            if not s.show_correspondences:
                remove_draw_handler()
            self.report({'INFO'}, "Pairing mode off.")
            try:
                context.area.tag_redraw()
            except Exception:
                pass
            return {'FINISHED'}

        if s.remove_mode:
            # Staged half-pairs from pairing mode are irrelevant while removing.
            self._stage = 0
            self._source_vertex = None

            if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
                if remove_correspondence_near_click(context, event, s):
                    try:
                        context.area.tag_redraw()
                    except Exception:
                        pass
                return {'RUNNING_MODAL'}
            return {'PASS_THROUGH'}

        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if self._stage == 0:
                pick = raycast_object_nearest_vertex(context, event, s.source_obj)
                if pick is None:
                    return {'PASS_THROUGH'}

                self._source_vertex = pick[0]
                self._stage = 1
                self.report(
                    {'INFO'},
                    f"Source vertex {self._source_vertex} selected. Click matching Target point."
                )
                context.area.tag_redraw()
                return {'RUNNING_MODAL'}

            pick = raycast_object_nearest_vertex(context, event, s.target_obj)
            if pick is None:
                return {'PASS_THROUGH'}

            item = s.correspondences.add()
            item.source_vertex = self._source_vertex
            item.target_vertex = pick[0]

            _CACHE["mapping"] = None
            _CACHE["rbf"] = None

            self.report(
                {'INFO'},
                f"Pair #{len(s.correspondences)}: Source {item.source_vertex} -> Target {item.target_vertex}"
            )

            self._stage = 0
            self._source_vertex = None
            context.area.tag_redraw()
            return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}


class MER_OT_delete_last_pair(bpy.types.Operator):
    bl_idname = "mer.delete_last_pair"
    bl_label = "Delete Last"

    def execute(self, context):
        s = context.scene.mer_settings
        n = len(s.correspondences)

        if n:
            s.correspondences.remove(n - 1)
            _CACHE["mapping"] = None
            _CACHE["rbf"] = None
            self.report({'INFO'}, "Deleted last correspondence.")

        return {'FINISHED'}


class MER_OT_remove_pair_mode(bpy.types.Operator):
    bl_idname = "mer.remove_pair_mode"
    bl_label = "Remove Pair"
    bl_description = (
        "Click a correspondence marker to delete that pair, leaving the rest "
        "of the list untouched. Esc or right-click to stop."
    )
    bl_options = {'REGISTER'}

    def invoke(self, context, event):
        if not context.scene.mer_settings.source_obj or not context.scene.mer_settings.target_obj:
            self.report({'ERROR'}, "Choose both Source and Target meshes.")
            return {'CANCELLED'}

        if context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Run remove mode from the 3D View.")
            return {'CANCELLED'}

        return bpy.ops.mer.pairing_mode('INVOKE_DEFAULT', remove_only=True)


class MER_OT_clear_pairs(bpy.types.Operator):
    bl_idname = "mer.clear_pairs"
    bl_label = "Clear All"

    def execute(self, context):
        s = context.scene.mer_settings
        s.correspondences.clear()
        _CACHE["mapping"] = None
        _CACHE["rbf"] = None

        if not s.pairing_mode:
            s.show_correspondences = False
            remove_draw_handler()

        self.report({'INFO'}, "Cleared all correspondences.")
        return {'FINISHED'}


# -----------------------------------------------------------------------------
# Mapping library
# -----------------------------------------------------------------------------

class MER_OT_save_mapping(bpy.types.Operator):
    bl_idname = "mer.save_mapping"
    bl_label = "Save Mapping"
    bl_description = "Save current correspondence pairs as a named reusable mapping"

    overwrite: bpy.props.BoolProperty(default=False)

    def execute(self, context):
        s = context.scene.mer_settings
        source = s.source_obj
        target = s.target_obj

        if not source or not target:
            self.report({'ERROR'}, "Choose Source and Target meshes before saving.")
            return {'CANCELLED'}

        if not s.correspondences:
            self.report({'ERROR'}, "There are no correspondence pairs to save.")
            return {'CANCELLED'}

        name = sanitize_mapping_name(s.mapping_name)
        filepath = mapping_filepath_from_name(name)

        if os.path.exists(filepath) and not self.overwrite:
            self.report(
                {'ERROR'},
                f"Mapping '{name}' already exists. Use Replace Selected or choose another name."
            )
            return {'CANCELLED'}

        data = {
            "format": "mesh_expression_retargeter_mapping",
            "version": 1,
            "name": name,
            "source": {
                "vertex_count": len(source.data.vertices),
                "polygon_count": len(source.data.polygons),
            },
            "target": {
                "vertex_count": len(target.data.vertices),
                "polygon_count": len(target.data.polygons),
            },
            "pairs": [
                {
                    "source_vertex": int(pair.source_vertex),
                    "target_vertex": int(pair.target_vertex),
                }
                for pair in s.correspondences
            ],
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.report({'ERROR'}, f"Could not save mapping: {e}")
            return {'CANCELLED'}

        try:
            s.mapping_preset = "user:" + os.path.basename(filepath)
        except Exception:
            pass

        self.report(
            {'INFO'},
            f"Saved mapping '{name}' with {len(data['pairs'])} pairs."
        )
        return {'FINISHED'}


class MER_OT_replace_mapping(bpy.types.Operator):
    bl_idname = "mer.replace_mapping"
    bl_label = "Replace Selected"

    def execute(self, context):
        s = context.scene.mer_settings
        preset = s.mapping_preset

        if not preset or preset == "__NONE__":
            self.report({'ERROR'}, "Select a saved mapping first.")
            return {'CANCELLED'}

        if mapping_is_bundled(preset):
            self.report(
                {'ERROR'},
                "Bundled mappings cannot be replaced. Load it and save a new one instead."
            )
            return {'CANCELLED'}

        s.mapping_name = os.path.splitext(mapping_filename(preset))[0]
        return bpy.ops.mer.save_mapping('EXEC_DEFAULT', overwrite=True)


class MER_OT_load_mapping(bpy.types.Operator):
    bl_idname = "mer.load_mapping"
    bl_label = "Load Selected"
    bl_description = "Load the selected saved correspondence mapping"

    def execute(self, context):
        s = context.scene.mer_settings
        source = s.source_obj
        target = s.target_obj
        preset = s.mapping_preset

        if not source or not target:
            self.report({'ERROR'}, "Choose Source and Target meshes before loading.")
            return {'CANCELLED'}

        if not preset or preset == "__NONE__":
            self.report({'ERROR'}, "No saved mapping selected.")
            return {'CANCELLED'}

        filepath = find_mapping_file(preset)
        if not filepath:
            self.report({'ERROR'}, "Saved mapping file could not be found.")
            return {'CANCELLED'}

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except Exception as e:
            self.report({'ERROR'}, f"Could not load mapping: {e}")
            return {'CANCELLED'}

        data = normalize_mapping_data(raw_data)
        if data is None:
            self.report({'ERROR'}, "This is not a recognized mapping file.")
            return {'CANCELLED'}

        pairs = data.get("pairs", [])
        if not pairs:
            self.report({'ERROR'}, "Saved mapping contains no correspondence pairs.")
            return {'CANCELLED'}

        source_count = len(source.data.vertices)
        target_count = len(target.data.vertices)

        expected_source = data.get("source", {}).get("vertex_count")
        expected_target = data.get("target", {}).get("vertex_count")

        if expected_source is not None and int(expected_source) != source_count:
            self.report(
                {'ERROR'},
                f"Source topology mismatch: mapping expects {expected_source} vertices; selected mesh has {source_count}."
            )
            return {'CANCELLED'}

        if expected_target is not None and int(expected_target) != target_count:
            if mpfb_mapping_can_ignore_vertex_count(target, pairs):
                self.report(
                    {'WARNING'},
                    f"MPFB helper geometry changed the vertex count "
                    f"({expected_target} in the mapping, {target_count} selected); "
                    "all mapped face vertices are still valid, so the mapping was accepted."
                )
            else:
                self.report(
                    {'ERROR'},
                    f"Target topology mismatch: mapping expects {expected_target} vertices; selected mesh has {target_count}."
                )
                return {'CANCELLED'}

        validated = []
        try:
            for p in pairs:
                sv = int(p["source_vertex"])
                tv = int(p["target_vertex"])

                if sv < 0 or sv >= source_count:
                    raise ValueError(f"Source vertex {sv} is out of range.")

                if tv < 0 or tv >= target_count:
                    raise ValueError(f"Target vertex {tv} is out of range.")

                validated.append((sv, tv))
        except Exception as e:
            self.report({'ERROR'}, f"Invalid mapping data: {e}")
            return {'CANCELLED'}

        s.correspondences.clear()

        for sv, tv in validated:
            item = s.correspondences.add()
            item.source_vertex = sv
            item.target_vertex = tv

        s.mapping_name = data.get("name", os.path.splitext(preset)[0])
        s.show_correspondences = True

        _CACHE["mapping"] = None
        _CACHE["rbf"] = None

        self.report(
            {'INFO'},
            f"Loaded mapping '{s.mapping_name}' with {len(validated)} pairs."
        )
        return {'FINISHED'}


class MER_OT_delete_mapping(bpy.types.Operator):
    bl_idname = "mer.delete_mapping"
    bl_label = "Delete Selected"

    def execute(self, context):
        s = context.scene.mer_settings
        preset = s.mapping_preset

        if not preset or preset == "__NONE__":
            self.report({'ERROR'}, "No saved mapping selected.")
            return {'CANCELLED'}

        if mapping_is_bundled(preset):
            self.report(
                {'ERROR'},
                "Bundled mappings ship with the add-on and cannot be deleted."
            )
            return {'CANCELLED'}

        filepath = find_mapping_file(preset)

        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            self.report({'ERROR'}, f"Could not delete mapping: {e}")
            return {'CANCELLED'}

        self.report(
            {'INFO'},
            f"Deleted mapping '{os.path.splitext(mapping_filename(preset))[0]}'."
        )
        return {'FINISHED'}


# -----------------------------------------------------------------------------
# Fit and transfer
# -----------------------------------------------------------------------------

def get_anchor_arrays(context, settings, source_neutral_local):
    source = settings.source_obj
    target = settings.target_obj

    depsgraph = context.evaluated_depsgraph_get()
    etarget = target.evaluated_get(depsgraph)

    src = []
    dst = []

    for pair in settings.correspondences:
        if pair.source_vertex < 0 or pair.target_vertex < 0:
            continue

        sp = source.matrix_world @ source_neutral_local[pair.source_vertex]

        if pair.target_vertex >= len(etarget.data.vertices):
            continue

        tp = etarget.matrix_world @ etarget.data.vertices[pair.target_vertex].co

        src.append([sp.x, sp.y, sp.z])
        dst.append([tp.x, tp.y, tp.z])

    return np.asarray(src), np.asarray(dst)


def is_across_gap(target_normal, surface_normal):
    """
    True when the bound surface faces away from the target vertex's own sheet.

    The nearest-surface projection is only trustworthy when the surface it
    lands on faces the same way as the vertex's own surface. Opposed normals
    mean the hit belongs to a facing sheet across a gap (inner lip sheets
    touching across the mouth gap, nostril bags, eyelid folds): the straight
    path to it jumps the gap and the binding must be released.
    """
    return surface_normal.dot(target_normal) < 0.0


def is_mouth_interior_normal(normal_world, face_out):
    """
    True when a target vertex normal faces into the mouth cavity.

    Outer lip skin faces the same way as the face (dot with face_out > 0);
    the bag behind the lips faces the opposite way, into the oral cavity.
    """
    if normal_world.length_squared == 0.0:
        return False
    return normal_world.normalized().dot(face_out) < -0.1


def source_mouth_sock_vertices(source):
    """
    Base-mesh indices of the source's inner-mouth bag, if it names one.

    The GNM head marks its oral cavity as ``mouth_sock``. Other meshes may
    not; an empty set means "no inner-mouth surface to match".
    """
    group = source.vertex_groups.get("mouth_sock")
    if group is None:
        return set()
    gi = group.index
    return {
        v.index for v in source.data.vertices
        if any(g.group == gi for g in v.groups)
    }


def target_mouth_interior_mask(etarget, target_world, target_normal_mat, dst):
    """
    Boolean mask (evaluated target vertex count) of vertices facing into the
    oral cavity. Same test as Protect Mouth Interior, exposed so the inner
    bag can be bound to the source's mouth sock instead of being frozen.
    """
    count = len(etarget.data.vertices)
    mask = np.zeros(count, dtype=bool)
    if len(dst) == 0:
        return mask
    tverts_world = np.array(
        [tuple(target_world @ tv.co) for tv in etarget.data.vertices],
        dtype=np.float64,
    )
    center_np = np.asarray(dst, dtype=np.float64).mean(axis=0)
    near = tverts_world[
        np.linalg.norm(tverts_world - center_np, axis=1) < 0.25
    ]
    if len(near) == 0:
        return mask
    face_out = center_np - near.mean(axis=0)
    nlen = float(np.linalg.norm(face_out))
    if nlen <= 1e-6:
        return mask
    face_out = Vector((face_out / nlen).tolist())
    center = Vector(center_np.tolist())
    for i, tv in enumerate(etarget.data.vertices):
        p_world = target_world @ tv.co
        if abs(p_world.x - center.x) > 0.05:
            continue
        if abs(p_world.z - center.z) > 0.05:
            continue
        if (p_world - center).dot(face_out) > 0.02:
            continue
        if is_mouth_interior_normal(target_normal_mat @ tv.normal, face_out):
            mask[i] = True
    return mask


def pick_visible_candidate(candidates, p_world, target_normal=None):
    """
    Choose the best source surface point for a target vertex.

    Candidates are BVH find_nearest_range results: (hit, normal, tri, dist),
    nearest first. A point is visible from p_world when its normal points
    toward p_world. When target_normal is given, a same-facing candidate
    (surface normal agreeing with the target vertex normal) is preferred
    over an opposed one -- this is what keeps a closed mouth's inner and
    outer sheets from binding across the gap, without the over-culling of
    a global opposed-normal veto. Falls back to visible, then nearest.
    """
    if not candidates:
        return None
    visible = []
    same_facing = []
    for cand in candidates:
        hit, normal, _tri, _dist = cand
        if hit is None or normal is None:
            continue
        if normal.length_squared <= 0.0 or normal.dot(p_world - hit) <= 0.0:
            continue
        visible.append(cand)
        if (
            target_normal is not None
            and target_normal.length_squared > 0.0
            and normal.normalized().dot(target_normal.normalized()) > 0.0
        ):
            same_facing.append(cand)
    if same_facing:
        return same_facing[0]
    if visible:
        return visible[0]
    return candidates[0]


def collect_excluded_vertices(target, groups_text):
    """
    Base-mesh vertex indices belonging to any comma-separated group name.

    Unknown names are ignored, so the same list works across meshes.
    """
    if not groups_text:
        return set()
    wanted = {n.strip() for n in groups_text.split(",") if n.strip()}
    if not wanted:
        return set()
    out = set()
    for g in target.vertex_groups:
        if g.name not in wanted:
            continue
        gidx = g.index
        for v in target.data.vertices:
            for gi in v.groups:
                if gi.group == gidx:
                    out.add(v.index)
                    break
    return out


def vert_is_included(vi, included, excluded):
    """Include/exclude gate for a vertex index (include empty = all)."""
    if vi in excluded:
        return False
    if included and vi not in included:
        return False
    return True


INPAINT_HOP_LIMIT = 12


def demote_motion_outliers(deltas, mapped, edges, protect=(),
                           threshold_mm=2.5):
    """
    Demote mapped verts whose motion disagrees with their mapped
    neighbours to fill status. Catches bindings to static inner sheets
    (a lip vert bound to the mouth bag across the opening) that show up
    as motion discontinuities against the local neighbourhood. Anchor
    verts are never demoted. Returns the updated mask.
    """
    thr = threshold_mm / 1000.0
    out = mapped.copy()
    for p in protect:
        if 0 <= p < len(out):
            out[p] = True
    if len(edges) == 0:
        return out

    e = np.asarray(edges, dtype=np.int64)
    u, v = e[:, 0], e[:, 1]
    protect_mask = np.zeros(len(out), dtype=bool)
    for p in protect:
        if 0 <= p < len(out):
            protect_mask[p] = True

    for _ in range(2):
        sums = np.zeros_like(deltas)
        cnt = np.zeros(len(deltas))
        both = out[u] & out[v]
        if not both.any():
            break
        np.add.at(sums, u[both], deltas[v[both]])
        np.add.at(sums, v[both], deltas[u[both]])
        np.add.at(cnt, u[both], 1)
        np.add.at(cnt, v[both], 1)
        ok = out & (cnt >= 2) & (~protect_mask)
        if not ok.any():
            break
        mean = sums[ok] / cnt[ok][:, None]
        diff = np.linalg.norm(deltas[ok] - mean, axis=1)
        bad_idx = np.where(ok)[0][diff > thr]
        if bad_idx.size == 0:
            break
        out[bad_idx] = False

    return out


def harmonic_fill_displacements(deltas, fixed, edges,
                                hop_limit=INPAINT_HOP_LIMIT):
    """
    Inpaint displacements for unfixed verts from fixed neighbours.

    Hop-limited Jacobi relaxation of the mesh Laplacian: unfixed verts
    within hop_limit edges of the fixed set converge to neighbour
    averages, faded linearly toward zero at the limit; beyond the limit
    they stay zero. Pure numpy (Blender ships without scipy).
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    n = len(deltas)
    if len(edges) == 0 or not fixed.any():
        return deltas.copy()

    e = np.asarray(edges, dtype=np.int64)
    u, v = e[:, 0], e[:, 1]

    # BFS distance from the fixed set, capped at hop_limit.
    dist = np.full(n, -1, dtype=np.int64)
    dist[fixed] = 0
    frontier = fixed.copy()
    for h in range(1, hop_limit + 1):
        touch = frontier[u] | frontier[v]
        if not touch.any():
            break
        nu, nv = u[touch], v[touch]
        other = np.where(frontier[nu], nv, nu)
        fresh = other[dist[other] < 0]
        if fresh.size == 0:
            break
        dist[fresh] = h
        frontier = np.zeros(n, dtype=bool)
        frontier[fresh] = True

    fill = (~fixed) & (dist > 0)
    if not fill.any():
        return deltas.copy()

    x = deltas.copy()
    x[~fixed] = 0.0
    deg = np.bincount(np.concatenate([u, v]), minlength=n).astype(np.float64)
    deg[deg == 0.0] = 1.0

    for _ in range(400):
        sums = np.zeros_like(x)
        np.add.at(sums, u, x[v])
        np.add.at(sums, v, x[u])
        nxt = sums / deg[:, None]
        step = np.abs(nxt[fill] - x[fill]).max()
        x[fill] = nxt[fill]
        if step < 1e-8:
            break

    weights = 1.0 - (dist[fill].astype(np.float64) - 1.0) / float(hop_limit)
    x[fill] *= weights[:, None]
    return x


SMOOTH_MAPPED_ITERATIONS = 6
SMOOTH_MAPPED_FACTOR = 0.5


def smooth_mapped_displacements(deltas, mapped, edges, protect=(),
                                iterations=SMOOTH_MAPPED_ITERATIONS,
                                factor=SMOOTH_MAPPED_FACTOR):
    """
    Relax the transferred motion across mapped target vertices.

    Nearest-triangle binding is per-vertex, so two adjacent target verts can
    bind to opposite sides of a sharp static/moving seam on the Source (the
    inner mouth bag next to the moving outer lip, eyelid folds, nostril
    bags). The Source field is smooth; the binding is not. A few mouse-
    Laplacian passes over the mapped set remove that binding discontinuity
    while anchors and excluded groups stay pinned, preserving the expression.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    n = len(deltas)
    if len(edges) == 0 or iterations <= 0:
        return deltas.copy()

    e = np.asarray(edges, dtype=np.int64)
    u, v = e[:, 0], e[:, 1]

    movable = np.asarray(mapped, dtype=bool).copy()
    for p in protect:
        if 0 <= p < n:
            movable[p] = False
    if not movable.any():
        return deltas.copy()

    deg = np.bincount(np.concatenate([u, v]), minlength=n).astype(np.float64)
    deg[deg == 0.0] = 1.0

    x = deltas.copy()
    for _ in range(int(iterations)):
        sums = np.zeros_like(x)
        np.add.at(sums, u, x[v])
        np.add.at(sums, v, x[u])
        avg = sums / deg[:, None]
        x[movable] = (1.0 - factor) * x[movable] + factor * avg[movable]
    return x


class MER_OT_build_fit_map(bpy.types.Operator):
    bl_idname = "mer.build_fit_map"
    bl_label = "Build Transfer Map"
    bl_description = "Capture current Source as neutral and build the Source-to-Target deformation map"
    bl_options = {'REGISTER'}

    def execute(self, context):
        s = context.scene.mer_settings

        # Fitting is not an input operation. End any temporary pairing state.
        s.pairing_mode = False

        source = s.source_obj
        target = s.target_obj

        if not source or not target:
            self.report({'ERROR'}, "Choose Source and Target meshes.")
            return {'CANCELLED'}

        # Bind against the Target's rest face, not whatever expression keys
        # happen to be live. Otherwise the map depends on prior transfers and
        # the same setup binds differently run to run. Identity/modeling morphs
        # are left alone: they define the fitted head shape.
        saved_expressions = [
            (key, key.value) for key in expression_key_blocks(target)
            if key.value != 0.0
        ]
        for key, _value in saved_expressions:
            key.value = 0.0
        if saved_expressions:
            context.view_layer.update()

        def _done(result):
            restore_key_values(saved_expressions)
            return result

        source_neutral_local = [v.co.copy() for v in source.data.vertices]
        triangles = triangulate_polygons(source.data)

        if not triangles:
            self.report({'ERROR'}, "Source mesh has no usable polygon surface.")
            return _done({'CANCELLED'})

        depsgraph = context.evaluated_depsgraph_get()
        etarget = target.evaluated_get(depsgraph)
        target_world = etarget.matrix_world
        target_normal_mat = target_world.to_3x3()

        mapping = []
        mapped_count = 0
        released_count = 0
        max_dist = s.max_distance
        excluded = collect_excluded_vertices(target, s.exclude_vertex_groups)
        included = collect_excluded_vertices(target, s.include_vertex_groups)

        # ----------------------------------------------------------
        # Mode 1: Manual landmark correspondences + non-rigid RBF fit
        # ----------------------------------------------------------
        if s.mapping_method == "LANDMARKS":
            if len(s.correspondences) < 4:
                self.report(
                    {'ERROR'},
                    "Manual Correspondences mode needs at least 4 pairs. Facial transfers usually benefit from 12-30+."
                )
                return _done({'CANCELLED'})

            src, dst = get_anchor_arrays(context, s, source_neutral_local)

            try:
                if s.local_anchor_influence:
                    rbf = RBFLocalWarp(
                        src,
                        dst,
                        influence_radius=s.influence_radius,
                        regularization=s.rbf_regularization
                    )
                else:
                    rbf = RBFWarp(
                        src,
                        dst,
                        regularization=s.rbf_regularization
                    )
            except Exception as e:
                self.report({'ERROR'}, f"Fit failed: {e}")
                return _done({'CANCELLED'})

            source_neutral_world_np = np.array(
                [[*(source.matrix_world @ co)] for co in source_neutral_local],
                dtype=np.float64
            )

            warped_np = rbf.transform_array(source_neutral_world_np)
            fitted_source_world = [Vector(v.tolist()) for v in warped_np]

            bvh = BVHTree.FromPolygons(
                fitted_source_world,
                triangles,
                all_triangles=True
            )

            sock_set = set()
            interior_mask = None
            if (
                getattr(s, "match_mouth_sock", True)
                and target_is_mpfb_basemesh(target)
            ):
                sock_set = source_mouth_sock_vertices(source)
                if sock_set:
                    interior_mask = target_mouth_interior_mask(
                        etarget, target_world, target_normal_mat, dst
                    )

            for vi, tv in enumerate(etarget.data.vertices):
                if not vert_is_included(vi, included, excluded):
                    mapping.append(None)
                    continue

                p_world = target_world @ tv.co
                if s.bind_visible_only:
                    if max_dist > 0.0:
                        near = bvh.find_nearest(p_world, max_dist)
                    else:
                        near = bvh.find_nearest(p_world)
                    near_hit, _n0, _i0, d0 = near
                    if near_hit is None:
                        mapping.append(None)
                        continue
                    radius = min(max(d0 * 1.5, d0 + 0.005), 0.02)
                    cands = bvh.find_nearest_range(p_world, radius) or []
                    cands = sorted(cands, key=lambda c: c[3])[:8]
                    if interior_mask is not None and cands:
                        want_sock = bool(interior_mask[vi])
                        same_sheet = [
                            c for c in cands
                            if all(x in sock_set for x in triangles[c[2]])
                            == want_sock
                        ]
                        if same_sheet:
                            cands = same_sheet
                    tn = target_normal_mat @ tv.normal
                    picked = pick_visible_candidate(
                        cands, p_world, tn if s.bind_same_facing else None
                    )
                    if picked is None:
                        mapping.append(None)
                        continue
                    hit, normal, tri_index, dist = picked
                else:
                    hit, normal, tri_index, dist = bvh.find_nearest(p_world)

                if hit is None or tri_index is None:
                    mapping.append(None)
                    continue

                if max_dist > 0.0 and dist > max_dist:
                    mapping.append(None)
                    continue

                if s.release_across_gap:
                    tn = target_normal_mat @ tv.normal
                    if (
                        tn.length_squared > 0.0
                        and normal.length_squared > 0.0
                        and is_across_gap(tn.normalized(), normal.normalized())
                    ):
                        mapping.append(None)
                        released_count += 1
                        continue

                ia, ib, ic = triangles[tri_index]
                a = fitted_source_world[ia]
                b = fitted_source_world[ib]
                c = fitted_source_world[ic]

                bary = barycentric_coords(hit, a, b, c)
                offset_world = p_world - hit

                mapping.append((ia, ib, ic, bary, offset_world))
                mapped_count += 1

            _CACHE["rbf"] = rbf

            mode_message = (
                f"Manual fit built from {len(s.correspondences)} landmarks"
            )
            if s.local_anchor_influence:
                mode_message += f" (local, radius {s.influence_radius:.3f}m)"

        # ----------------------------------------------------------
        # Mode 2: Direct overlapping-surface barycentric map
        # ----------------------------------------------------------
        else:
            # No fitting: use Source neutral exactly where it currently sits.
            source_neutral_world = [
                source.matrix_world @ co for co in source_neutral_local
            ]

            bvh = BVHTree.FromPolygons(
                source_neutral_world,
                triangles,
                all_triangles=True
            )

            for vi, tv in enumerate(etarget.data.vertices):
                if not vert_is_included(vi, included, excluded):
                    mapping.append(None)
                    continue

                p_world = target_world @ tv.co
                if s.bind_visible_only:
                    if max_dist > 0.0:
                        near = bvh.find_nearest(p_world, max_dist)
                    else:
                        near = bvh.find_nearest(p_world)
                    near_hit, _n0, _i0, d0 = near
                    if near_hit is None:
                        mapping.append(None)
                        continue
                    radius = min(max(d0 * 1.5, d0 + 0.005), 0.02)
                    cands = bvh.find_nearest_range(p_world, radius) or []
                    cands = sorted(cands, key=lambda c: c[3])[:8]
                    tn = target_normal_mat @ tv.normal
                    picked = pick_visible_candidate(
                        cands, p_world, tn if s.bind_same_facing else None
                    )
                    if picked is None:
                        mapping.append(None)
                        continue
                    hit, normal, tri_index, dist = picked
                else:
                    hit, normal, tri_index, dist = bvh.find_nearest(p_world)

                if hit is None or tri_index is None:
                    mapping.append(None)
                    continue

                if max_dist > 0.0 and dist > max_dist:
                    mapping.append(None)
                    continue

                if s.release_across_gap:
                    tn = target_normal_mat @ tv.normal
                    if (
                        tn.length_squared > 0.0
                        and normal.length_squared > 0.0
                        and is_across_gap(tn.normalized(), normal.normalized())
                    ):
                        mapping.append(None)
                        released_count += 1
                        continue

                ia, ib, ic = triangles[tri_index]
                a = source_neutral_world[ia]
                b = source_neutral_world[ib]
                c = source_neutral_world[ic]

                bary = barycentric_coords(hit, a, b, c)
                offset_world = p_world - hit

                mapping.append((ia, ib, ic, bary, offset_world))
                mapped_count += 1

            # Explicitly mark that no RBF warp is used.
            _CACHE["rbf"] = None

            mode_message = "Overlapping-surface map built"

        # ----------------------------------------------------------
        # Protect the mouth interior: freeze cavity verts (normals face
        # into the mouth) near the landmark cluster so the inner bag can
        # not bind to the moving outer lip across the mouth gap.
        # ----------------------------------------------------------
        interior_count = 0
        if (
            s.protect_mouth_interior
            and s.mapping_method == "LANDMARKS"
            and len(s.correspondences) >= 4
        ):
            mouth_center_np = dst.mean(axis=0)
            mouth_center = Vector(mouth_center_np.tolist())
            tverts_world = np.array(
                [tuple(target_world @ tv.co) for tv in etarget.data.vertices],
                dtype=np.float64,
            )
            near = tverts_world[
                np.linalg.norm(tverts_world - mouth_center_np, axis=1) < 0.25
            ]
            if len(near):
                face_out = mouth_center_np - near.mean(axis=0)
                nlen = float(np.linalg.norm(face_out))
                if nlen > 1e-6:
                    face_out = Vector((face_out / nlen).tolist())
                    for i, tv in enumerate(etarget.data.vertices):
                        if mapping[i] is None:
                            continue
                        p_world = target_world @ tv.co
                        if abs(p_world.x - mouth_center.x) > 0.05:
                            continue
                        if abs(p_world.z - mouth_center.z) > 0.05:
                            continue
                        if (p_world - mouth_center).dot(face_out) > 0.02:
                            continue
                        tn = target_normal_mat @ tv.normal
                        if is_mouth_interior_normal(tn, face_out):
                            mapping[i] = None
                            mapped_count -= 1
                            interior_count += 1

        _CACHE["source_name"] = source.name
        _CACHE["target_name"] = target.name
        _CACHE["source_neutral_local"] = source_neutral_local
        _CACHE["source_triangles"] = triangles
        _CACHE["mapping"] = mapping

        released_note = (
            f"; released {released_count} across-gap bindings"
            if released_count else ""
        )
        interior_note = (
            f"; froze {interior_count} mouth-interior vertices"
            if interior_count else ""
        )

        self.report(
            {'INFO'},
            f"{mode_message}; mapped {mapped_count}/{len(etarget.data.vertices)} "
            f"target vertices{released_note}{interior_note}."
        )
        return _done({'FINISHED'})


def validate_runtime_transfer_map(operator, settings, source, target):
    """Validate that the cached transfer map can be used for these objects."""
    if _CACHE["mapping"] is None:
        operator.report({'ERROR'}, "Build the transfer map first while Source is neutral.")
        return False

    if (
        _CACHE["source_name"] != source.name
        or _CACHE["target_name"] != target.name
    ):
        operator.report(
            {'ERROR'},
            "The runtime transfer map belongs to different Source/Target objects."
        )
        return False

    if len(_CACHE["source_neutral_local"]) != len(source.data.vertices):
        operator.report({'ERROR'}, "Source topology changed after mapping.")
        return False

    if settings.mapping_method == "LANDMARKS" and _CACHE["rbf"] is None:
        operator.report(
            {'ERROR'},
            "This runtime map was not built with Manual Correspondences. Rebuild the map."
        )
        return False

    return True


def fitted_source_positions(settings, source, source_current_local):
    """
    Return neutral/current Source coordinates in the same fitted world space
    used when the runtime transfer map was built.
    """
    source_neutral_local = _CACHE["source_neutral_local"]

    if len(source_neutral_local) != len(source_current_local):
        raise ValueError("Source topology changed after mapping.")

    if settings.mapping_method == "LANDMARKS":
        neutral_world_np = np.array(
            [[*(source.matrix_world @ co)] for co in source_neutral_local],
            dtype=np.float64
        )
        current_world_np = np.array(
            [[*(source.matrix_world @ co)] for co in source_current_local],
            dtype=np.float64
        )

        warped_neutral_np = _CACHE["rbf"].transform_array(neutral_world_np)
        warped_current_np = _CACHE["rbf"].transform_array(current_world_np)

        fitted_neutral = [Vector(v.tolist()) for v in warped_neutral_np]
        fitted_current = [Vector(v.tolist()) for v in warped_current_np]
    else:
        fitted_neutral = [
            source.matrix_world @ co for co in source_neutral_local
        ]
        fitted_current = [
            source.matrix_world @ co for co in source_current_local
        ]

    return fitted_neutral, fitted_current


def transfer_positions_to_key(settings, source, target, source_current_local, key):
    """Transfer explicit Source vertex positions into an existing Target key."""
    fitted_neutral, fitted_current = fitted_source_positions(
        settings,
        source,
        source_current_local
    )

    keys = target.data.shape_keys.key_blocks
    basis = keys["Basis"]
    inv_target_world = target.matrix_world.inverted()
    scale = settings.displacement_scale

    n = min(len(key.data), len(_CACHE["mapping"]))
    deltas = np.zeros((n, 3), dtype=np.float64)
    mapped = np.zeros(n, dtype=bool)

    for i in range(n):
        entry = _CACHE["mapping"][i]
        if entry is None:
            continue

        ia, ib, ic, bary, offset_world = entry
        u, v, w = bary

        neu = (
            fitted_neutral[ia] * u
            + fitted_neutral[ib] * v
            + fitted_neutral[ic] * w
        )
        cur = (
            fitted_current[ia] * u
            + fitted_current[ib] * v
            + fitted_current[ic] * w
        )

        delta = (cur - neu) * scale
        deltas[i] = delta[:]
        mapped[i] = True

    transferred = int(mapped.sum())

    if settings.harmonic_inpaint and mapped.any():
        edges = [
            (e.vertices[0], e.vertices[1]) for e in target.data.edges
            if e.vertices[0] < n and e.vertices[1] < n
        ]
        anchors = [
            c.target_vertex for c in settings.correspondences
            if 0 <= c.target_vertex < n
        ]
        before = int(mapped.sum())
        fixed0 = mapped.copy()
        # Pass 1: demote obvious discontinuities against mapped context.
        mapped = demote_motion_outliers(deltas, fixed0, edges, protect=anchors)
        # Pass 2: judge again against the filled (smooth) context, so a
        # coherent static island among movers — whose members agree with
        # each other and survived pass 1 — finally stands out.
        smooth = harmonic_fill_displacements(deltas, mapped, edges)
        mapped = demote_motion_outliers(smooth, fixed0, edges, protect=anchors)
        _CACHE["demoted"] = before - int((mapped & fixed0).sum())

        # Excluded groups are frozen at Basis: fixed with zero delta so
        # the fill flows around them instead of dragging them along.
        excluded_frozen = collect_excluded_vertices(
            target, settings.exclude_vertex_groups
        )
        for vi in excluded_frozen:
            if vi < n and not mapped[vi]:
                deltas[vi] = 0.0
                mapped[vi] = True
        deltas = harmonic_fill_displacements(deltas, mapped, edges)

        if settings.smooth_mapped_motion:
            # Anchors pin the neutral fit, not the transferred motion: at a
            # coincident double-sheet (mouth corners, eyelid folds) an anchor
            # can land on the wrong sheet while its neighbours take the other,
            # so locking it here would fold the skin. Let motion anchors relax
            # with their neighbourhood; excluded groups stay frozen.
            deltas = smooth_mapped_displacements(
                deltas, mapped, edges,
                protect=[vi for vi in excluded_frozen if vi < n]
            )

    for i in range(n):
        basis_world = target.matrix_world @ basis.data[i].co
        key.data[i].co = inv_target_world @ (Vector(deltas[i].tolist()) + basis_world)

    return transferred


def add_uniquely_named_shape_key(target, requested_name):
    """
    Always create a new Target shape key. Blender's own name allocator gives
    duplicate names its standard .001, .002, ... suffixes, including filling
    the first available suffix.
    """
    return target.shape_key_add(name=requested_name, from_mix=False)


class MER_OT_transfer_expression(bpy.types.Operator):
    bl_idname = "mer.transfer_expression"
    bl_label = "Transfer Current Expression"
    bl_description = "Transfer the Source mesh's current deformation onto Target as a shape key"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.mer_settings
        source = s.source_obj
        target = s.target_obj

        if not source or not target:
            self.report({'ERROR'}, "Choose Source and Target meshes.")
            return {'CANCELLED'}

        if not validate_runtime_transfer_map(self, s, source, target):
            return {'CANCELLED'}

        source_current_local = [v.co.copy() for v in source.data.vertices]

        ensure_basis(target)
        keys = target.data.shape_keys.key_blocks
        name = s.shapekey_name.strip() or "Transferred_Expression"

        # Preserve the original single-transfer behaviour: an explicitly named
        # existing key is updated rather than duplicated.
        if name in keys:
            key = keys[name]
        else:
            key = target.shape_key_add(name=name, from_mix=False)
        mark_transferred_key(target, key)

        try:
            transferred = transfer_positions_to_key(
                s,
                source,
                target,
                source_current_local,
                key
            )
        except Exception as e:
            self.report({'ERROR'}, f"Transfer failed: {e}")
            return {'CANCELLED'}

        cleared = zero_expression_keys(target, keep=key)
        key.value = 1.0

        message = f"Transferred '{key.name}' to {transferred} target vertices."
        if cleared:
            message += f" Zeroed {cleared} previous expression key(s)."

        self.report(
            {'INFO'},
            message
        )
        return {'FINISHED'}


class MER_OT_transfer_all_shape_keys(bpy.types.Operator):
    bl_idname = "mer.transfer_all_shape_keys"
    bl_label = "Transfer All Shape Keys"
    bl_description = (
        "Transfer every non-Basis Source shape key to Target, preserving names "
        "and creating Blender-style numbered names when a name already exists"
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.mer_settings
        source = s.source_obj
        target = s.target_obj

        if not source or not target:
            self.report({'ERROR'}, "Choose Source and Target meshes.")
            return {'CANCELLED'}

        if not validate_runtime_transfer_map(self, s, source, target):
            return {'CANCELLED'}

        source_shape_keys = source.data.shape_keys
        if source_shape_keys is None or len(source_shape_keys.key_blocks) <= 1:
            self.report({'ERROR'}, "Source has no non-Basis shape keys to transfer.")
            return {'CANCELLED'}

        source_keys = [
            key for key in source_shape_keys.key_blocks
            if key != source_shape_keys.reference_key
        ]

        ensure_basis(target)

        # A batch leaves every new Target key at 0.0, so the Target must start
        # neutral: keys a previous single transfer left at 1.0 would otherwise
        # stay active underneath the whole batch.
        prior_values = {
            key: key.value
            for key in target.data.shape_keys.key_blocks
            if key != target.data.shape_keys.reference_key
        }
        zeroed = zero_expression_keys(target)

        created_keys = []
        total_vertices = 0

        try:
            for source_key in source_keys:
                if len(source_key.data) != len(source.data.vertices):
                    raise ValueError(
                        f"Source shape key '{source_key.name}' does not match Source topology."
                    )

                source_positions = [point.co.copy() for point in source_key.data]
                target_key = add_uniquely_named_shape_key(target, source_key.name)
                mark_transferred_key(target, target_key)
                created_keys.append(target_key)

                total_vertices += transfer_positions_to_key(
                    s,
                    source,
                    target,
                    source_positions,
                    target_key
                )

                # Batch-created expressions should not all become active at once.
                target_key.value = 0.0

        except Exception as e:
            # Do not leave a partially-created batch behind if one key fails.
            if target.data.shape_keys is not None:
                for key in reversed(created_keys):
                    try:
                        target.shape_key_remove(key)
                    except Exception:
                        pass
                # The pre-batch zeroing happened before the failure; put the
                # Target back exactly as it was found.
                for key, value in prior_values.items():
                    try:
                        key.value = value
                    except Exception:
                        pass
            self.report({'ERROR'}, f"Transfer All Shape Keys failed: {e}")
            return {'CANCELLED'}

        message = (
            f"Transferred {len(created_keys)} Source shape keys "
            f"({total_vertices} mapped vertex writes)."
        )
        if zeroed:
            message += f" Zeroed {zeroed} previous shape key(s)."
        self.report({'INFO'}, message)
        return {'FINISHED'}


def _expression_tags(settings):
    return [t.strip() for t in settings.expression_tags.split(",") if t.strip()]


def _validate_expression_target(operator, settings):
    """Shared guards for the MPFB export operators."""
    if not mpfb_available():
        operator.report({'ERROR'}, "MPFB is not installed.")
        return None

    target = settings.target_obj
    if not target_is_mpfb_basemesh(target):
        operator.report({'ERROR'}, "Target must be an MPFB basemesh.")
        return None

    if target.data.shape_keys is None:
        operator.report({'ERROR'}, "Target has no shape keys. Transfer first.")
        return None

    return target


class MER_OT_save_mpfb_expression(bpy.types.Operator):
    bl_idname = "mer.save_mpfb_expression"
    bl_label = "Save as MPFB Expression"
    bl_description = (
        "Project the Target shape key onto MPFB's ARKit face units and write it "
        "as an MPFB expression. Requires MPFB and an MPFB basemesh as Target."
    )
    bl_options = {'REGISTER'}

    def execute(self, context):
        s = context.scene.mer_settings
        target = _validate_expression_target(self, s)
        if target is None:
            return {'CANCELLED'}

        keys = target.data.shape_keys.key_blocks
        name = s.shapekey_name.strip() or "Transferred_Expression"
        key = keys.get(name)

        if key is None or key == target.data.shape_keys.reference_key:
            self.report({'ERROR'}, f"Target has no '{name}' shape key. Transfer first.")
            return {'CANCELLED'}

        try:
            path, count, residual = save_shape_key_as_mpfb_expression(
                target,
                key,
                s.expression_filename,
                _expression_tags(s),
                s.expression_author,
                s.refresh_library
            )
        except Exception as e:
            self.report({'ERROR'}, f"Could not save expression: {e}")
            return {'CANCELLED'}

        self.report(
            {'INFO'},
            f"Saved '{os.path.basename(path)}' with {count} face units "
            f"({residual * 100.0:.0f}% not representable)."
        )
        return {'FINISHED'}


class MER_OT_save_all_mpfb_expressions(bpy.types.Operator):
    bl_idname = "mer.save_all_mpfb_expressions"
    bl_label = "Save All as MPFB Expressions"
    bl_description = (
        "Project every non-Basis Target shape key onto MPFB's ARKit face units "
        "and write one MPFB expression per key. Requires MPFB and an MPFB "
        "basemesh as Target."
    )
    bl_options = {'REGISTER'}

    def execute(self, context):
        s = context.scene.mer_settings
        target = _validate_expression_target(self, s)
        if target is None:
            return {'CANCELLED'}

        shape_keys = target.data.shape_keys
        keys = [
            key for key in shape_keys.key_blocks
            if key != shape_keys.reference_key
        ]
        if not keys:
            self.report({'ERROR'}, "Target has no non-Basis shape keys to export.")
            return {'CANCELLED'}

        tags = _expression_tags(s)
        saved = []
        skipped = []
        residuals = []

        for key in keys:
            try:
                path, count, residual = save_shape_key_as_mpfb_expression(
                    target,
                    key,
                    key.name + ".json",
                    tags,
                    s.expression_author,
                    False
                )
            except Exception as e:
                skipped.append(f"{key.name} ({e})")
                continue
            saved.append(os.path.basename(path))
            residuals.append(residual)

        if s.refresh_library and saved:
            try:
                from bl_ext.blender_org.mpfb.ui.apply_assets.useexpression import (
                    refresh_expression_sliders,
                )
                refresh_expression_sliders(force_redraw=False)
            except Exception:
                pass

        if not saved:
            detail = "; ".join(skipped[:3])
            self.report({'ERROR'}, f"No expressions saved. {detail}")
            return {'CANCELLED'}

        worst = max(residuals) if residuals else 0.0
        message = f"Saved {len(saved)} expressions"
        if skipped:
            message += f", skipped {len(skipped)}"
        message += f" (worst {worst * 100.0:.0f}% not representable)"
        self.report({'INFO'}, message)

        if skipped:
            self.report({'WARNING'}, "; ".join(skipped[:5]))

        return {'FINISHED'}


class MER_OT_clear_runtime(bpy.types.Operator):
    bl_idname = "mer.clear_runtime"
    bl_label = "Clear Runtime Transfer Map"
    bl_description = "Clear the fitted runtime map; saved landmark mappings are unaffected"

    def execute(self, context):
        for k in _CACHE:
            _CACHE[k] = None

        self.report(
            {'INFO'},
            "Runtime transfer map cleared. Saved mappings and correspondence pairs remain."
        )
        return {'FINISHED'}


class MER_OT_reset_neutral(bpy.types.Operator):
    bl_idname = "mer.reset_neutral"
    bl_label = "Reset Target to Neutral"
    bl_description = (
        "Return the Target to its rest face without collapsing it: zero MPFB "
        "'!ex-' face units and this addon's transferred expressions, while "
        "keeping identity/modeling morphs and the Basis. The MPFB equivalent "
        "of resetting the GNM head"
    )
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        s = context.scene.mer_settings
        target = s.target_obj
        if not target:
            self.report({'ERROR'}, "Choose a Target mesh.")
            return {'CANCELLED'}
        cleared = reset_to_neutral(target)
        context.view_layer.update()
        self.report(
            {'INFO'},
            "Target reset to neutral (%d expression key(s) zeroed)." % cleared
        )
        return {'FINISHED'}


# -----------------------------------------------------------------------------
# Correspondence overlay
# -----------------------------------------------------------------------------

def pair_world_points(context, settings, pair):
    source = settings.source_obj
    target = settings.target_obj

    if not source or not target:
        return None, None

    try:
        if pair.source_vertex < 0 or pair.target_vertex < 0:
            return None, None

        esource = evaluated_object(context, source)
        etarget = evaluated_object(context, target)

        if pair.source_vertex >= len(esource.data.vertices):
            return None, None

        if pair.target_vertex >= len(etarget.data.vertices):
            return None, None

        sp = esource.matrix_world @ esource.data.vertices[pair.source_vertex].co
        tp = etarget.matrix_world @ etarget.data.vertices[pair.target_vertex].co

        return sp, tp
    except Exception:
        return None, None


def draw_callback():
    """
    Draw correspondence markers only in a valid 3D View WINDOW region.

    The handler is installed lazily, but Blender can still redraw regions
    while windows/areas are being created or destroyed, so all context access
    is deliberately defensive.
    """
    try:
        context = bpy.context

        if context is None or context.scene is None:
            return

        if not hasattr(context.scene, "mer_settings"):
            return

        area = context.area
        region = context.region
        space = context.space_data

        if area is None or area.type != 'VIEW_3D':
            return

        if region is None or region.type != 'WINDOW':
            return

        if space is None or getattr(space, "type", None) != 'VIEW_3D':
            return

        rv3d = getattr(space, "region_3d", None)
        if rv3d is None:
            return

        s = context.scene.mer_settings

        if not s.show_correspondences:
            return

        if not s.source_obj or not s.target_obj:
            return

        if not s.correspondences:
            return

        shader = gpu.shader.from_builtin('UNIFORM_COLOR')

        show_source = bool(s.source_obj.visible_get())
        show_target = bool(s.target_obj.visible_get())

        for idx, pair in enumerate(s.correspondences, start=1):
            sp, tp = pair_world_points(context, s, pair)

            if sp is None and tp is None:
                continue

            # A hidden mesh should not still pepper the viewport with its
            # markers, so only the points of visible objects are drawn.
            pts = [p for p, ok in ((sp, show_source), (tp, show_target))
                   if ok and p is not None]
            if not pts:
                continue

            batch = batch_for_shader(
                shader,
                'POINTS',
                {"pos": pts}
            )

            shader.bind()
            shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
            gpu.state.point_size_set(8.0)
            batch.draw(shader)

            for p in pts:
                co2d = view3d_utils.location_3d_to_region_2d(
                    region,
                    rv3d,
                    p
                )

                if co2d:
                    blf.position(0, co2d.x + 6, co2d.y + 6, 0)
                    blf.size(0, 12)
                    blf.draw(0, str(idx))

    except (ReferenceError, RuntimeError, AttributeError):
        # Window/area destruction can invalidate RNA objects between checks.
        # Drawing is optional, so silently skip that redraw.
        return


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

class MER_PT_panel(bpy.types.Panel):
    bl_label = "Mesh Expression Retargeter"
    bl_idname = "MER_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Expression Transfer"

    def draw(self, context):
        layout = self.layout
        s = context.scene.mer_settings

        mode_box = layout.box()
        mode_box.label(text="Mapping Method")
        mode_box.prop(s, "mapping_method", text="")

        objects_box = layout.box()
        objects_box.label(text="Meshes")
        objects_box.prop(s, "source_obj")
        objects_box.prop(s, "target_obj")

        if s.mapping_method == "LANDMARKS":
            pairs_box = layout.box()
            pairs_box.label(text=f"Correspondences: {len(s.correspondences)}")

            notice = pair_count_notice(len(s.correspondences))
            if notice:
                notice_col = pairs_box.column(align=True)
                notice_col.scale_y = 0.8
                for line in notice:
                    notice_col.label(text=line)

            row = pairs_box.row()
            row.alert = s.pairing_mode
            row.operator(
                "mer.pairing_mode",
                text="Exit Pairing Mode" if s.pairing_mode else "Enter Pairing Mode"
            )

            row = pairs_box.row(align=True)
            row.operator("mer.delete_last_pair", text="Delete Last")
            row.operator("mer.clear_pairs", text="Clear All")

            remove_row = pairs_box.row(align=True)
            remove_row.alert = s.remove_mode
            remove_row.operator(
                "mer.remove_pair_mode",
                text="Remove Pair" if not s.remove_mode else "Removing - Esc to stop"
            )
            pairs_box.prop(s, "show_correspondences")

            mapping_box = layout.box()
            mapping_box.label(text="Mapping Library")
            mapping_box.prop(s, "mapping_name")
            mapping_box.operator("mer.save_mapping", text="Save New Mapping")
            mapping_box.separator()
            mapping_box.prop(s, "mapping_preset")

            row = mapping_box.row(align=True)
            row.operator("mer.load_mapping", text="Load Selected")
            row.operator("mer.replace_mapping", text="Replace Selected")

            mapping_box.operator("mer.delete_mapping", text="Delete Selected")

            fit_box = layout.box()
            fit_box.label(text="Fit")
            fit_box.prop(s, "rbf_regularization")
            fit_box.prop(s, "local_anchor_influence")
            if s.local_anchor_influence:
                fit_box.prop(s, "influence_radius")
            fit_box.prop(s, "max_distance")
            fit_box.prop(s, "release_across_gap")
            fit_box.prop(s, "bind_visible_only")
            fit_box.prop(s, "bind_same_facing")
            fit_box.prop(s, "match_mouth_sock")
            fit_box.prop(s, "protect_mouth_interior")
            fit_box.prop(s, "include_vertex_groups")
            fit_box.prop(s, "exclude_vertex_groups")
            fit_box.operator("mer.build_fit_map", text="Fit / Build Transfer Map")

        else:
            overlap_box = layout.box()
            overlap_box.label(text="Overlapping Surface Mapping")
            overlap_box.label(text="Source and Target must already overlap.")
            overlap_box.label(text="No landmark pairs are required.")
            overlap_box.prop(s, "max_distance")
            overlap_box.prop(s, "release_across_gap")
            overlap_box.prop(s, "bind_visible_only")
            overlap_box.prop(s, "bind_same_facing")
            overlap_box.prop(s, "include_vertex_groups")
            overlap_box.prop(s, "exclude_vertex_groups")
            overlap_box.operator("mer.build_fit_map", text="Build Surface Map")

        expression_box = layout.box()
        expression_box.label(text="Expression Transfer")
        expression_box.prop(s, "shapekey_name")
        expression_box.prop(s, "displacement_scale")
        expression_box.prop(s, "harmonic_inpaint")
        expression_box.prop(s, "smooth_mapped_motion")
        expression_box.operator(
            "mer.transfer_expression",
            text="Transfer Current Expression"
        )
        expression_box.operator(
            "mer.transfer_all_shape_keys",
            text="Transfer All Shape Keys"
        )
        expression_box.operator(
            "mer.reset_neutral",
            text="Reset Target to Neutral"
        )

        if mpfb_available():
            expression_box.separator()
            if target_is_mpfb_basemesh(s.target_obj):
                expression_box.prop(s, "expression_filename")
                expression_box.operator(
                    "mer.save_mpfb_expression",
                    text="Save as MPFB Expression"
                )
                expression_box.operator(
                    "mer.save_all_mpfb_expressions",
                    text="Save All as MPFB Expressions"
                )
                expression_box.prop(s, "refresh_library")

                details = expression_box.box()
                details.label(text="File Details")
                details.prop(s, "expression_tags")
                details.prop(s, "expression_author")
            else:
                expression_box.label(text="MPFB found: set Target to an MPFB basemesh")
        else:
            # Silently hiding the buttons leaves a user with no way to tell
            # "MPFB is not installed" from "MPFB is installed but broken".
            detected = mpfb_detect()
            if detected:
                alert = expression_box.box()
                alert.alert = True
                alert.label(text="MPFB expressions not available")
                for line in _wrap_label(detected, 46):
                    alert.label(text=line)

        layout.operator("mer.clear_runtime")

        help_box = layout.box()
        help_box.label(text="Workflow")

        if s.mapping_method == "LANDMARKS":
            help_box.label(text="1. Put Source in neutral.")
            help_box.label(text="2. Load/create correspondences.")
            help_box.label(text="3. Fit / Build Transfer Map.")
            help_box.label(text="4. Change Source expression.")
            help_box.label(text="5. Transfer to Target shape key.")
        else:
            help_box.label(text="1. Overlap Source and Target.")
            help_box.label(text="2. Put Source in neutral.")
            help_box.label(text="3. Build Surface Map.")
            help_box.label(text="4. Change Source deformation.")
            help_box.label(text="5. Transfer to Target shape key.")

        if mpfb_available() and target_is_mpfb_basemesh(s.target_obj):
            help_box.label(text="6. Save as MPFB Expression to keep it.")


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

classes = (
    MER_Correspondence,
    MER_Settings,
    MER_OT_pairing_mode,
    MER_OT_delete_last_pair,
    MER_OT_remove_pair_mode,
    MER_OT_clear_pairs,
    MER_OT_save_mapping,
    MER_OT_replace_mapping,
    MER_OT_load_mapping,
    MER_OT_delete_mapping,
    MER_OT_build_fit_map,
    MER_OT_transfer_expression,
    MER_OT_transfer_all_shape_keys,
    MER_OT_save_mpfb_expression,
    MER_OT_save_all_mpfb_expressions,
    MER_OT_clear_runtime,
    MER_OT_reset_neutral,
    MER_PT_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.mer_settings = bpy.props.PointerProperty(
        type=MER_Settings
    )


def unregister():
    # Mark any surviving modal pairing operators for termination on their
    # next event before removing the property definition.
    try:
        for scene in bpy.data.scenes:
            if hasattr(scene, "mer_settings"):
                scene.mer_settings.pairing_mode = False
    except Exception:
        pass

    remove_draw_handler()

    if hasattr(bpy.types.Scene, "mer_settings"):
        del bpy.types.Scene.mer_settings

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
