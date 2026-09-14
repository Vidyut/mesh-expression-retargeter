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
# Mapping preset library
# -----------------------------------------------------------------------------



def mapping_library_dir():
    """Writable per-user data directory for the official Blender Extension."""
    root = bpy.utils.extension_path_user(__package__, create=True)
    path = os.path.join(root, "mappings")
    os.makedirs(path, exist_ok=True)
    return path


def sanitize_mapping_name(name):
    safe = "".join(
        c if c.isalnum() or c in (" ", "-", "_", ".") else "_"
        for c in name.strip()
    )
    safe = safe.strip(" .")
    return safe or "Source to Target"


def mapping_preset_items(self, context):
    folder = mapping_library_dir()
    items = []

    try:
        files = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith(".json")
        )
    except Exception:
        files = []

    for i, filename in enumerate(files):
        stem = os.path.splitext(filename)[0]
        items.append((filename, stem, filename, i))

    if not items:
        items.append(("__NONE__", "(no saved mappings)", "", 0))

    return items


def mapping_filepath_from_name(name):
    safe = sanitize_mapping_name(name)
    return os.path.join(mapping_library_dir(), safe + ".json")


def find_mapping_file(filename):
    path = os.path.join(mapping_library_dir(), filename)
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

    max_distance: bpy.props.FloatProperty(
        name="Max Fit Distance",
        description="Ignore target vertices farther than this from the fitted source surface. 0 means unlimited.",
        default=0.0,
        min=0.0,
        unit='LENGTH'
    )

    displacement_scale: bpy.props.FloatProperty(
        name="Expression Scale",
        description="Scale the transferred deformation",
        default=1.0,
        soft_min=0.0,
        soft_max=2.0
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

class MER_OT_pairing_mode(bpy.types.Operator):
    bl_idname = "mer.pairing_mode"
    bl_label = "Pairing Mode"
    bl_description = "Continuously click Source then Target to create correspondence pairs"
    bl_options = {'REGISTER'}

    _stage = 0
    _source_vertex = None

    def invoke(self, context, event):
        s = context.scene.mer_settings

        if not s.source_obj or not s.target_obj:
            self.report({'ERROR'}, "Choose both Source and Target meshes.")
            return {'CANCELLED'}

        if context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, "Run pairing mode from the 3D View.")
            return {'CANCELLED'}

        if s.pairing_mode:
            s.pairing_mode = False
            if not s.show_correspondences:
                remove_draw_handler()
            self.report({'INFO'}, "Pairing mode off.")
            context.area.tag_redraw()
            return {'FINISHED'}

        s.pairing_mode = True
        s.show_correspondences = True
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
            if not s.show_correspondences:
                remove_draw_handler()
            self.report({'INFO'}, "Pairing mode off.")
            try:
                context.area.tag_redraw()
            except Exception:
                pass
            return {'FINISHED'}

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
            s.mapping_preset = os.path.basename(filepath)
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

        s.mapping_name = os.path.splitext(preset)[0]
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

        filepath = find_mapping_file(preset)

        try:
            if filepath and os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            self.report({'ERROR'}, f"Could not delete mapping: {e}")
            return {'CANCELLED'}

        self.report(
            {'INFO'},
            f"Deleted mapping '{os.path.splitext(preset)[0]}'."
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

        source_neutral_local = [v.co.copy() for v in source.data.vertices]
        triangles = triangulate_polygons(source.data)

        if not triangles:
            self.report({'ERROR'}, "Source mesh has no usable polygon surface.")
            return {'CANCELLED'}

        depsgraph = context.evaluated_depsgraph_get()
        etarget = target.evaluated_get(depsgraph)
        target_world = etarget.matrix_world

        mapping = []
        mapped_count = 0
        max_dist = s.max_distance

        # ----------------------------------------------------------
        # Mode 1: Manual landmark correspondences + non-rigid RBF fit
        # ----------------------------------------------------------
        if s.mapping_method == "LANDMARKS":
            if len(s.correspondences) < 4:
                self.report(
                    {'ERROR'},
                    "Manual Correspondences mode needs at least 4 pairs. Facial transfers usually benefit from 12-30+."
                )
                return {'CANCELLED'}

            src, dst = get_anchor_arrays(context, s, source_neutral_local)

            try:
                rbf = RBFWarp(
                    src,
                    dst,
                    regularization=s.rbf_regularization
                )
            except Exception as e:
                self.report({'ERROR'}, f"Fit failed: {e}")
                return {'CANCELLED'}

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

            for tv in etarget.data.vertices:
                p_world = target_world @ tv.co
                hit, normal, tri_index, dist = bvh.find_nearest(p_world)

                if hit is None or tri_index is None:
                    mapping.append(None)
                    continue

                if max_dist > 0.0 and dist > max_dist:
                    mapping.append(None)
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

            for tv in etarget.data.vertices:
                p_world = target_world @ tv.co
                hit, normal, tri_index, dist = bvh.find_nearest(p_world)

                if hit is None or tri_index is None:
                    mapping.append(None)
                    continue

                if max_dist > 0.0 and dist > max_dist:
                    mapping.append(None)
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

        _CACHE["source_name"] = source.name
        _CACHE["target_name"] = target.name
        _CACHE["source_neutral_local"] = source_neutral_local
        _CACHE["source_triangles"] = triangles
        _CACHE["mapping"] = mapping

        self.report(
            {'INFO'},
            f"{mode_message}; mapped {mapped_count}/{len(etarget.data.vertices)} target vertices."
        )
        return {'FINISHED'}


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

        if _CACHE["mapping"] is None:
            self.report({'ERROR'}, "Build the transfer map first while Source is neutral.")
            return {'CANCELLED'}

        if (
            _CACHE["source_name"] != source.name
            or _CACHE["target_name"] != target.name
        ):
            self.report(
                {'ERROR'},
                "The runtime transfer map belongs to different Source/Target objects."
            )
            return {'CANCELLED'}

        source_neutral_local = _CACHE["source_neutral_local"]
        source_current_local = [v.co.copy() for v in source.data.vertices]

        if len(source_neutral_local) != len(source_current_local):
            self.report({'ERROR'}, "Source topology changed after mapping.")
            return {'CANCELLED'}

        # ----------------------------------------------------------
        # Convert neutral/current Source into the same fitted space
        # used when the map was built.
        # ----------------------------------------------------------
        if s.mapping_method == "LANDMARKS":
            if _CACHE["rbf"] is None:
                self.report(
                    {'ERROR'},
                    "This runtime map was not built with Manual Correspondences. Rebuild the map."
                )
                return {'CANCELLED'}

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
            # Direct mode: Source is already spatially aligned.
            fitted_neutral = [
                source.matrix_world @ co for co in source_neutral_local
            ]
            fitted_current = [
                source.matrix_world @ co for co in source_current_local
            ]

        ensure_basis(target)

        keys = target.data.shape_keys.key_blocks
        name = s.shapekey_name.strip() or "Transferred_Expression"

        if name in keys:
            key = keys[name]
        else:
            key = target.shape_key_add(name=name, from_mix=False)

        basis = keys["Basis"]
        inv_target_world = target.matrix_world.inverted()
        scale = s.displacement_scale
        transferred = 0

        for i, entry in enumerate(_CACHE["mapping"]):
            if i >= len(key.data):
                break

            if entry is None:
                key.data[i].co = basis.data[i].co
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
            basis_world = target.matrix_world @ basis.data[i].co
            new_world = basis_world + delta

            key.data[i].co = inv_target_world @ new_world
            transferred += 1

        key.value = 1.0

        self.report(
            {'INFO'},
            f"Transferred '{name}' to {transferred} target vertices."
        )
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

        for idx, pair in enumerate(s.correspondences, start=1):
            sp, tp = pair_world_points(context, s, pair)

            if sp is None or tp is None:
                continue

            batch = batch_for_shader(
                shader,
                'POINTS',
                {"pos": [sp, tp]}
            )

            shader.bind()
            shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
            gpu.state.point_size_set(8.0)
            batch.draw(shader)

            for p in (sp, tp):
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

            row = pairs_box.row()
            row.alert = s.pairing_mode
            row.operator(
                "mer.pairing_mode",
                text="Exit Pairing Mode" if s.pairing_mode else "Enter Pairing Mode"
            )

            row = pairs_box.row(align=True)
            row.operator("mer.delete_last_pair", text="Delete Last")
            row.operator("mer.clear_pairs", text="Clear All")
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
            fit_box.prop(s, "max_distance")
            fit_box.operator("mer.build_fit_map", text="Fit / Build Transfer Map")

        else:
            overlap_box = layout.box()
            overlap_box.label(text="Overlapping Surface Mapping")
            overlap_box.label(text="Source and Target must already overlap.")
            overlap_box.label(text="No landmark pairs are required.")
            overlap_box.prop(s, "max_distance")
            overlap_box.operator("mer.build_fit_map", text="Build Surface Map")

        expression_box = layout.box()
        expression_box.label(text="Expression Transfer")
        expression_box.prop(s, "shapekey_name")
        expression_box.prop(s, "displacement_scale")
        expression_box.operator(
            "mer.transfer_expression",
            text="Transfer Current Expression"
        )

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


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

classes = (
    MER_Correspondence,
    MER_Settings,
    MER_OT_pairing_mode,
    MER_OT_delete_last_pair,
    MER_OT_clear_pairs,
    MER_OT_save_mapping,
    MER_OT_replace_mapping,
    MER_OT_load_mapping,
    MER_OT_delete_mapping,
    MER_OT_build_fit_map,
    MER_OT_transfer_expression,
    MER_OT_clear_runtime,
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
