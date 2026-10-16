"""Headless verification for Mesh Expression Retargeter 1.2.12 MPFB export."""

import importlib
import importlib.util
import json
import os
import sys

import bpy
import addon_utils

RESULTS = {}

INSTALLED = (
    "/home/vidyut/.config/blender/5.2/extensions/user_default/"
    "mesh_expression_retargeter/__init__.py"
)


def check(name, condition, detail=""):
    RESULTS[name] = {"ok": bool(condition), "detail": str(detail)}
    print(("PASS " if condition else "FAIL ") + name + " :: " + str(detail), flush=True)


def load_module():
    """Load the installed extension, preferring its real module name."""
    try:
        return importlib.import_module("bl_ext.user_default.mesh_expression_retargeter")
    except Exception:
        pass

    spec = importlib.util.spec_from_file_location("mer", INSTALLED)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mer"] = module
    spec.loader.exec_module(module)
    return module


def main():
    try:
        addon_utils.enable("bl_ext.user_default.mesh_expression_retargeter", default_set=False)
    except Exception as exc:
        print("enable note:", repr(exc), flush=True)

    mer = load_module()
    importlib.reload(mer)
    check("module_loaded", mer is not None)

    try:
        mer.register()
    except Exception as exc:
        check("register", False, repr(exc))
        return RESULTS
    check("register", True)

    # Shape-key isolation (1.2.2): activating one expression must not leave
    # every previously transferred expression active underneath it.
    import bmesh

    probe_mesh = bpy.data.meshes.new("_mer_selftest_mesh")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(probe_mesh)
    bm.free()
    probe = bpy.data.objects.new("_mer_selftest_target", probe_mesh)
    bpy.context.collection.objects.link(probe)

    probe.shape_key_add(name="Basis", from_mix=False)
    key_a = probe.shape_key_add(name="expr_a", from_mix=False)
    key_a.value = 1.0
    key_b = probe.shape_key_add(name="expr_b", from_mix=False)
    key_b.value = 1.0

    cleared = mer.zero_other_shape_keys(probe, keep=key_b)
    check("zero_clears_others", cleared == 1 and key_a.value == 0.0,
          "cleared=%s a=%s" % (cleared, key_a.value))
    check("zero_keeps_target", key_b.value == 1.0, "b=%s" % key_b.value)
    check("zero_keeps_basis",
          probe.data.shape_keys.key_blocks["Basis"].value == 1.0)

    # Across-gap rejection (1.2.3): a binding whose surface faces away from
    # the target vertex's own sheet has jumped a gap (e.g. inner lip sheets
    # touching across the mouth) and must be released.
    from mathutils import Vector
    import numpy as np

    check("across_gap_opposed",
          mer.is_across_gap(Vector((0, 0, -1)), Vector((0, 0, 1))))
    check("across_gap_aligned_ok",
          not mer.is_across_gap(Vector((0, 0, 1)), Vector((0, 0, 1))))
    check("across_gap_tangent_ok",
          not mer.is_across_gap(Vector((1, 0, 0)), Vector((0, 0, 1))))
    check("across_gap_setting_default",
          bool(bpy.context.scene.mer_settings.release_across_gap))
    check("interior_opposed",
          mer.is_mouth_interior_normal(Vector((0, 1, 0)), Vector((0, -1, 0))))
    check("interior_aligned_ok",
          not mer.is_mouth_interior_normal(Vector((0, -1, 0)), Vector((0, -1, 0))))
    check("interior_zero_safe",
          not mer.is_mouth_interior_normal(Vector((0, 0, 0)), Vector((0, -1, 0))))
    check("protect_mouth_interior_default",
          not bool(bpy.context.scene.mer_settings.protect_mouth_interior))
    check("bind_visible_only_default",
          bool(bpy.context.scene.mer_settings.bind_visible_only))
    check("exclude_groups_default",
          "HelperGeometry" in bpy.context.scene.mer_settings.exclude_vertex_groups)

    # pick_visible_candidate: first front-facing wins, else nearest.
    check("visible_pick_front",
          mer.pick_visible_candidate(
              [(Vector((0, 0, 1)), Vector((0, 0, -1)), 1, 1.0),
               (Vector((0, 0, 2)), Vector((0, 0, 1)), 2, 2.0)],
              Vector((0, 0, 0)))[2] == 1)
    check("visible_pick_fallback",
          mer.pick_visible_candidate(
              [(Vector((0, 0, 1)), Vector((0, 0, 1)), 7, 1.0)],
              Vector((0, 0, 0)))[2] == 7)
    check("visible_pick_empty",
          mer.pick_visible_candidate([], Vector((0, 0, 0))) is None)

    # collect_excluded_vertices on the MPFB basemesh.
    axbody = bpy.data.objects.get("Aaxyl.body")
    check("exclude_helper_geometry",
          axbody is not None
          and len(mer.collect_excluded_vertices(axbody, "HelperGeometry")) > 0)
    check("exclude_empty_text",
          axbody is not None
          and len(mer.collect_excluded_vertices(axbody, "")) == 0)
    check("exclude_unknown_ok",
          axbody is not None
          and len(mer.collect_excluded_vertices(axbody, "no-such-group")) == 0)
    check("match_mouth_sock_default",
          bool(bpy.context.scene.mer_settings.match_mouth_sock))
    check("sock_missing_group_empty",
          axbody is not None and mer.source_mouth_sock_vertices(axbody) == set())
    check("include_gate_include",
          mer.vert_is_included(5, {5}, set()))
    check("include_gate_exclude_wins",
          not mer.vert_is_included(5, {5}, {5}))
    check("include_gate_empty_included",
          mer.vert_is_included(9, set(), set())
          and not mer.vert_is_included(9, {1, 2}, set()))

    # Local compact warp: anchors exact, beyond-radius field is identity.
    a_src = np.array([[0, 0, 0], [0.03, 0, 0], [0, 0.03, 0],
                      [0, 0, 0.03], [0.015, 0.015, 0]], dtype=np.float64)
    a_dst = a_src + np.array([[0, 0, 0.01], [0, 0.01, 0], [0.005, 0, 0],
                              [0, 0, 0.008], [0.002, 0.002, 0.005]],
                             dtype=np.float64)
    lw = mer.RBFLocalWarp(a_src, a_dst, 0.04)
    far = np.array([[5.0, 5.0, 5.0], [5.1, 5.0, 5.0]], dtype=np.float64)
    far_t = lw.transform_array(far)
    check("local_far_rigid",
          bool(np.allclose(far_t[1] - far_t[0], far[1] - far[0], atol=1e-9)
               and np.allclose(far_t[0] - far[0], lw.translation, atol=1e-9)))
    check("local_anchor_exact",
          bool(np.allclose(lw.transform_array(a_src[:1])[0], a_dst[0],
                           atol=1e-6)))
    mid = lw.transform_array(np.array([[0.015, 0.0, 0.0]]))
    check("local_midpoint_between",
          0.0 < float(mid[0][1]) < 0.01 and float(mid[0][2]) > 0.0)

    # Harmonic inpaint: fixed boundary respected, interior smoothed,
    # outside the hop limit untouched.
    chain_d = np.zeros((7, 3), dtype=np.float64)
    chain_d[0] = [0.01, 0.0, 0.0]
    chain_f = np.zeros(7, dtype=bool)
    chain_f[0] = True
    chain_e = [(i, i + 1) for i in range(6)]
    filled = mer.harmonic_fill_displacements(chain_d, chain_f, chain_e,
                                             hop_limit=3)
    check("inpaint_fixed_exact",
          bool(np.allclose(filled[0], [0.01, 0.0, 0.0])))
    check("inpaint_falloff",
          0.0 < filled[3][0] < filled[2][0] < filled[1][0] < 0.01)
    check("inpaint_outside_limit",
          bool(np.allclose(filled[5], 0.0) and np.allclose(filled[6], 0.0)))

    # Motion-outlier demotion: a static vert among moving neighbours is
    # demoted; protected (anchor) verts never are.
    dm_d = np.zeros((3, 3), dtype=np.float64)
    dm_d[0] = [0.01, 0.0, 0.0]
    dm_d[2] = [0.01, 0.0, 0.0]
    dm_m = np.array([True, True, True])
    dm_e = [(0, 1), (1, 2)]
    dm = mer.demote_motion_outliers(dm_d, dm_m, dm_e, threshold_mm=2.5)
    check("demote_disagreement", bool(not dm[1] and dm[0] and dm[2]))
    dm_p = mer.demote_motion_outliers(dm_d, dm_m, dm_e, protect=[1],
                                      threshold_mm=2.5)
    check("demote_protect", bool(dm_p[1]))

    # Same-facing candidate preference: an opposed-normal candidate that is
    # merely visible must lose to a same-facing one (closed-mouth sheets).
    from mathutils import Vector as V
    tn = V((0.0, 0.0, -1.0))
    p = V((0.0, 0.0, 0.0))
    opposed = (V((0.0, 0.0, -0.01)), V((0.0, 0.0, 1.0)), 0, 0.001)
    same = (V((0.0, 0.0, 0.02)), V((0.0, 0.0, -1.0)), 1, 0.002)
    picked = mer.pick_visible_candidate([opposed, same], p, tn)
    check("same_facing_preferred", picked is same)
    picked_no = mer.pick_visible_candidate([opposed, same], p, None)
    check("same_facing_off_falls_back", picked_no is opposed)

    # Mapped-motion relaxation (1.2.9): a per-vertex binding spike between two
    # moving verts is blended out, anchors stay exact, the motion survives.
    sm_d = np.zeros((4, 3), dtype=np.float64)
    sm_d[0] = [0.01, 0.0, 0.0]
    sm_d[1] = [0.0, 0.0, 0.0]
    sm_d[2] = [0.01, 0.0, 0.0]
    sm_d[3] = [0.01, 0.0, 0.0]
    sm_m = np.array([True, True, True, True])
    sm_e = [(0, 1), (1, 2), (2, 3)]
    sm = mer.smooth_mapped_displacements(sm_d, sm_m, sm_e, protect=[0],
                                         iterations=4)
    check("smooth_mapped_fills_spike", bool(sm[1][0] > 0.004), float(sm[1][0]))
    check("smooth_mapped_anchor_pinned", bool(np.allclose(sm[0], sm_d[0])))
    check("smooth_mapped_motion_kept", bool(sm[3][0] > 0.007), float(sm[3][0]))
    check("smooth_mapped_setting_default",
          bool(bpy.context.scene.mer_settings.smooth_mapped_motion))

    key_a.value = 1.0
    key_b.value = 1.0
    cleared_all = mer.zero_other_shape_keys(probe)
    check("zero_clears_all",
          cleared_all == 2 and key_a.value == 0.0 and key_b.value == 0.0,
          "cleared=%s" % cleared_all)

    # Expression vs identity keys (1.2.11): only expressions may be zeroed.
    # Identity/morph keys zeroed by a transfer would collapse a fitted head.
    ident = probe.shape_key_add(name="$md-identity", from_mix=False)
    ident.value = 0.4
    ex = probe.shape_key_add(name="!ex-jawOpen", from_mix=False)
    ex.value = 0.3
    marked = probe.shape_key_add(name="my_expression", from_mix=False)
    marked.value = 0.7
    mer.mark_transferred_key(probe, marked)
    check("is_expression_mpfb", mer.is_expression_key(probe, ex))
    check("is_expression_marked", mer.is_expression_key(probe, marked))
    check("is_expression_identity_false", not mer.is_expression_key(probe, ident))
    expr_zeroed = mer.zero_expression_keys(probe)
    check("zero_expr_keeps_identity",
          expr_zeroed == 2 and abs(ident.value - 0.4) < 1e-5
          and ex.value == 0.0 and marked.value == 0.0,
          "zeroed=%s ident=%s" % (expr_zeroed, ident.value))

    bpy.data.objects.remove(probe, do_unlink=True)
    bpy.data.meshes.remove(probe_mesh)

    check("mpfb_available", mer.mpfb_available())

    vectors, units, vertex_count = mer.load_faceunit_basis()
    check("faceunit_basis_shape",
          vectors is not None and len(units) == 52 and vertex_count == 15120,
          "units=%s verts=%s" % (len(units), vertex_count))
    active = mer.faceunit_active_vertices()
    check("faceunit_subset", active is not None and 52 <= len(active) <= 1024,
          "active=%s" % (None if active is None else len(active)))

    body = bpy.data.objects.get("Aaxyl.body")
    check("basemesh_present", body is not None)
    check("target_is_mpfb_basemesh", mer.target_is_mpfb_basemesh(body))

    # Helper-geometry count tolerance (1.2.4): a preset authored on the
    # helper-stripped basemesh stays valid on the full basemesh as long as
    # every stored target index exists on the selected mesh.
    nverts = len(body.data.vertices) if body else 0
    good_pairs = [{"target_vertex": 0}, {"target_vertex": nverts - 1}]
    check("helper_count_in_range_ok",
          mer.mpfb_mapping_can_ignore_vertex_count(body, good_pairs))
    check("helper_count_out_of_range_rejected",
          not mer.mpfb_mapping_can_ignore_vertex_count(
              body, [{"target_vertex": nverts}]))
    check("helper_count_non_mpfb_rejected",
          not mer.mpfb_mapping_can_ignore_vertex_count(probe, good_pairs))

    # Round-trip: fitting a face unit's own displacement must recover that unit.
    from bl_ext.blender_org.mpfb.services.targetservice import TargetService

    full_path = TargetService.target_full_path("jawOpen")
    TargetService.load_target(body, full_path, weight=1.0, name="!ex-jawOpen")
    key = body.data.shape_keys.key_blocks["!ex-jawOpen"]
    displacement = mer.shape_key_displacement(body, key)
    weights, residual = mer.fit_faceunits(displacement)

    check("roundtrip_unit", "jawOpen" in weights and weights["jawOpen"] > 0.99,
          "jawOpen=%s" % weights.get("jawOpen"))
    check("roundtrip_residual", residual < 0.01, "residual=%.6f" % residual)
    # Anatomically adjacent units share basis columns, so least squares leaks a
    # little weight into them. That is expected; what matters is that jawOpen
    # dominates and nothing else comes close.
    others = {k: v for k, v in weights.items() if k != "jawOpen"}
    check("roundtrip_dominant",
          not others or max(others.values()) < 0.05,
          "max_other=%s" % (max(others.values()) if others else 0.0))

    # A real expression is a blend, not a single unit. Build one by summing two
    # independently loaded targets and check both come back.
    basis_key = body.data.shape_keys.key_blocks["Basis"]
    TargetService.load_target(body, TargetService.target_full_path("jawOpen"),
                              weight=1.0, name="probeJaw")
    TargetService.load_target(body, TargetService.target_full_path("mouthSmileLeft"),
                              weight=1.0, name="probeSmile")
    jk = body.data.shape_keys.key_blocks["probeJaw"]
    sk = body.data.shape_keys.key_blocks["probeSmile"]
    blend = [
        (jk.data[i].co - basis_key.data[i].co)
        + (sk.data[i].co - basis_key.data[i].co)
        for i in range(len(jk.data))
    ]
    blend_weights, blend_residual = mer.fit_faceunits(blend)
    check("blend_jawOpen",
          abs(blend_weights.get("jawOpen", 0.0) - 1.0) < 0.05,
          "jawOpen=%s" % blend_weights.get("jawOpen"))
    check("blend_mouthSmileLeft",
          abs(blend_weights.get("mouthSmileLeft", 0.0) - 1.0) < 0.05,
          "mouthSmileLeft=%s" % blend_weights.get("mouthSmileLeft"))
    check("blend_residual", blend_residual < 0.02, "residual=%.6f" % blend_residual)

    # End-to-end write into the MPFB expression library.
    out_name = "_mer_selftest_expression"
    out_path, unit_count, relative = mer.save_shape_key_as_mpfb_expression(
        body, key, out_name, ["selftest"], "mer selftest", False
    )
    check("expression_written", os.path.isfile(out_path), out_path)
    # The raw fit carries the correlated units too; FaceService drops the ones
    # too small to matter, which is why the read-back below is just jawOpen.
    check("expression_units", unit_count >= 1, unit_count)

    from bl_ext.blender_org.mpfb.services.faceservice import FaceService
    loaded, meta = FaceService.load_expression(out_path)
    check("expression_readable", loaded == {"jawOpen": 1.0}, loaded)
    check("expression_metadata", meta.get("tags") == ["selftest"], meta.get("tags"))

    if os.path.isfile(out_path):
        os.remove(out_path)

    print(json.dumps(RESULTS, indent=1), flush=True)


if __name__ == "__main__":
    main()
