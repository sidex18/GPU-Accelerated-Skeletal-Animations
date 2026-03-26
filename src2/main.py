import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import warp as wp
from typing import Optional
import copy
from scipy.spatial.transform import Rotation, Slerp
import time 
 
from cli import parse_args
from skeleton import Skeleton
from mesh import Mesh
from forward_kinematics import compose_local_transforms_func, compute_world_transforms_func
from skinning import compute_material_coordinates, linear_blend_skin
from ik_solver import create_ik_chains, solve_ik
from utils import load_weights
 
MAX_BIND_BONES = 4
ENABLE_IK = True
 
BONE_GROUPS = {
    "Spine": ["c_spine0", "c_spine1", "c_spine2", "c_spine3"],
    "Head": ["c_neck", "c_head"],
    "Left Arm": ["l_clavicle", "l_uparm", "l_lowarm"],
    "Right Arm": ["r_clavicle", "r_uparm", "r_lowarm"],
    "Left Leg": ["l_upleg", "l_lowleg", "l_foot"],
    "Right Leg": ["r_upleg", "r_lowleg", "r_foot"],
}
 
# --- Keyframe animation state (globals) ---
keyframes = []          # list of (time_float, euler_dict_copy)
playback_time = 0.0
playing = False
last_frame_time = 0.0 
playback_speed = [1.0]  # list so it's mutable inside callback
 
 
def get_interpolated_pose(t: float, bone_names: list[str]) -> dict:
    """Interpolate between keyframes at time t.
    
    Uses linear interpolation for Euler angles converted to quaternions,
    then SLERP between them for smooth rotation. Falls back to linear
    for the trivial cases (0 or 1 keyframe).
    """
    if not keyframes:
        return {name: [0.0, 0.0, 0.0] for name in bone_names}
 
    if len(keyframes) == 1:
        return copy.deepcopy(keyframes[0][1])
 
    # Loop playback: wrap t into [t_start, t_end]
    t_start = keyframes[0][0]
    t_end   = keyframes[-1][0]
    duration = t_end - t_start
    if duration <= 0:
        return copy.deepcopy(keyframes[0][1])
 
    t_loop = (t - t_start) % duration + t_start
 
    # Find the two surrounding keyframes
    # After looping, t_loop is always in [t_start, t_end)
    kf_before = keyframes[0]
    kf_after  = keyframes[-1]
    for i in range(len(keyframes) - 1):
        if keyframes[i][0] <= t_loop <= keyframes[i + 1][0]:
            kf_before = keyframes[i]
            kf_after  = keyframes[i + 1]
            break
 
    t0, pose0 = kf_before
    t1, pose1 = kf_after
 
    # Compute interpolation factor alpha in [0, 1]
    segment_duration = t1 - t0
    if segment_duration < 1e-6:
        return copy.deepcopy(pose0)
    alpha = (t_loop - t0) / segment_duration
 
    # Interpolate each bone using SLERP on quaternions
    result = {}
    for name in bone_names:
        angles0 = pose0.get(name, [0.0, 0.0, 0.0])
        angles1 = pose1.get(name, [0.0, 0.0, 0.0])
 
        # Convert Euler XYZ -> quaternion for each keyframe
        q0 = Rotation.from_euler("XYZ", angles0)
        q1 = Rotation.from_euler("XYZ", angles1)
 
        # SLERP between the two quaternions
        # scipy Slerp expects a list of times and a list of rotations
        slerp = Slerp([0.0, 1.0], Rotation.concatenate([q0, q1]))
        q_interp = slerp(alpha)
 
        # Convert back to Euler XYZ
        result[name] = list(q_interp.as_euler("XYZ").astype(np.float32))
 
    return result
 
 
def record_keyframe(euler_angles: dict, t: float):
    """Record the current pose as a keyframe at time t."""
    pose_copy = {k: v[:] for k, v in euler_angles.items()}
    keyframes.append((t, pose_copy))
    # Keep keyframes sorted by time
    keyframes.sort(key=lambda kf: kf[0])
    print(f"Recorded keyframe at t={t:.2f}s  ({len(keyframes)} total)")
 
 
def pack_euler_angles(euler_dict, controlled_indices, num_bones):
    """Pack {bone_name: [rx,ry,rz]} dict into (num_bones, 3) numpy array."""
    out = np.zeros((num_bones, 3), dtype=np.float32)
    for name, idx in controlled_indices.items():
        out[idx] = euler_dict[name]
    return out
 
 
def main(argv: Optional[list[str]] = None) -> int:
    global playback_time, playing, last_frame_time
    last_frame_time = time.time()
    args = parse_args(argv)
    ENABLE_IK = args.IK
    wp.init()
    device = wp.get_device(args.device) if args.device else wp.get_preferred_device()
 
    # --- Load data ---
    skeleton  = Skeleton("data/skeleton_bind.json", device=device)
    mesh_data = Mesh("data/base_mesh.obj", device=device)
 
    vert_bone_indices, vert_bone_weights = load_weights(
        filename="data/vertex_weights.npz",
        num_verts=mesh_data.num_verts,
        name_to_idx=skeleton.name_to_idx,
        max_bind_bones=MAX_BIND_BONES)
    mesh_data.set_skinning_data(vert_bone_indices, vert_bone_weights)
 
    # Bind-pose world transforms
    euler_np = np.zeros((skeleton.num_bones, 3), dtype=np.float32)
    bind_local_transforms = compose_local_transforms_func(
        skeleton.bind_local_transforms, euler_np, device)
    world_bind_transforms = compute_world_transforms_func(
        skeleton.parent_indices, bind_local_transforms, device)
 
    # Skeleton visualization edges
    edges = [[skeleton.parent_indices[i], i]
             for i in range(skeleton.num_bones) if skeleton.parent_indices[i] != -1]
    bone_edges = np.array(edges, dtype=np.int32)
 
    wp.synchronize()
    world_np = world_bind_transforms.numpy()
    bind_bone_positions = world_np[:, :3].copy()
 
    # Material coordinates (computed once at bind pose)
    mesh_data.material_coords = compute_material_coordinates(
        mesh_data.verts, mesh_data.bone_indices, world_bind_transforms, device)
 
    # Initial vertex positions
    updated_vertex_positions = linear_blend_skin(
        mesh_data.material_coords, mesh_data.bone_indices, mesh_data.bone_weights,
        world_bind_transforms, device)
    wp.synchronize()
    verts = updated_vertex_positions.numpy()
 
    # Controlled bones
    controlled_indices = {}
    for bones in BONE_GROUPS.values():
        for name in bones:
            if name in skeleton.name_to_idx:
                controlled_indices[name] = skeleton.name_to_idx[name]
 
    euler_angles = {name: [0.0, 0.0, 0.0] for name in controlled_indices}
    bone_names   = list(controlled_indices.keys())
 
    # --- Polyscope setup ---
    ps.init()
    ps.set_up_dir("y_up")
    mesh = ps.register_surface_mesh("man", verts, mesh_data.faces, smooth_shade=True)
    skeleton_net = ps.register_curve_network(
        "skeleton", bind_bone_positions, bone_edges, radius=0.002)
    skeleton_net.set_color((0.2, 0.6, 1.0))
    ps.look_at((0.0, 80.0, 250.0), (0.0, 80.0, 0.0))
 
    # --- IK setup ---
    ik_chains, ik_point_clouds = [], {}
    ik_lr_exp  = [-4.0]
    ik_max_iter = [20]
 
    if ENABLE_IK:
        ik_chains = create_ik_chains(skeleton.name_to_idx)
        for chain in ik_chains:
            chain.target_position = world_np[chain.end_effector_index, :3].copy()
            pc = ps.register_point_cloud(
                f"IK: {chain.name}", np.array([[0.0, 0.0, 0.0]]), radius=0.008)
            pc.set_position(tuple(chain.target_position))
            pc.set_transform_gizmo_enabled(True)
            pc.set_enabled(False)
            ik_point_clouds[chain.name] = pc
 
    # -----------------------------------------------------------------------
    def callback():
        global playback_time, playing, last_frame_time
 
        changed    = False
        ik_changed = False
 
        # ── Skeleton toggle ──────────────────────────────────────────────
        c, show_skel = psim.Checkbox("Show Skeleton", skeleton_net.is_enabled())
        if c:
            skeleton_net.set_enabled(show_skel)
        psim.Separator()
 
        # ── IK controls ──────────────────────────────────────────────────
        if ENABLE_IK:
            if psim.TreeNode("Inverse Kinematics"):
                for chain in ik_chains:
                    c, chain.enabled = psim.Checkbox(f"Enable {chain.name} IK", chain.enabled)
                    if c:
                        pc = ik_point_clouds[chain.name]
                        if chain.enabled:
                            local_t = compose_local_transforms_func(
                                skeleton.bind_local_transforms,
                                pack_euler_angles(euler_angles, controlled_indices, skeleton.num_bones),
                                device)
                            world_t = compute_world_transforms_func(
                                skeleton.parent_indices, local_t, device)
                            wp.synchronize()
                            ee_pos = world_t.numpy()[chain.end_effector_index, :3].copy()
                            chain.target_position = ee_pos
                            pc.set_position(tuple(ee_pos))
                            pc.set_enabled(True)
                            ik_changed = True
                        else:
                            pc.set_enabled(False)
                            changed = True
 
                c, ik_lr_exp[0] = psim.SliderFloat(
                    "Learning Rate (1e)", ik_lr_exp[0], v_min=-5.0, v_max=-2.0)
                psim.TextUnformatted(f"  lr = {10.0 ** ik_lr_exp[0]:.6f}")
                c, ik_max_iter[0] = psim.SliderInt(
                    "Max Iterations", ik_max_iter[0], v_min=1, v_max=50)
                psim.TreePop()
            psim.Separator()
 
        # ── FK sliders ───────────────────────────────────────────────────
        if psim.Button("Reset All"):
            for name in euler_angles:
                euler_angles[name] = [0.0, 0.0, 0.0]
            changed = True
 
        for group_name, bone_names_group in BONE_GROUPS.items():
            if psim.TreeNode(group_name):
                for bone_name in bone_names_group:
                    if bone_name not in euler_angles:
                        continue
                    if psim.TreeNode(bone_name):
                        for axis_i, axis_label in enumerate(["X", "Y", "Z"]):
                            c, euler_angles[bone_name][axis_i] = psim.SliderAngle(
                                f"{axis_label}##{bone_name}",
                                euler_angles[bone_name][axis_i],
                                v_degrees_min=-180.0, v_degrees_max=180.0)
                            if c:
                                changed = True
                        psim.TreePop()
                psim.TreePop()
 
        # ── Keyframe animation ───────────────────────────────────────────
        psim.Separator()
        if psim.TreeNode("Keyframe Animation"):
 
            if psim.Button("Record Pose"):
                record_keyframe(euler_angles, playback_time)
                # Advance time by 1 second for the next keyframe slot
                playback_time += 1.0
 
            psim.SameLine()
            if psim.Button("Clear All"):
                keyframes.clear()
                playback_time = 0.0
                playing = False
 
            # Play / pause
            c, playing = psim.Checkbox("Play", playing)
            if playing and not keyframes:
                playing = False     # can't play with no keyframes
 
            # Speed slider
            c, playback_speed[0] = psim.SliderFloat(
                "Speed", playback_speed[0], v_min=0.1, v_max=5.0)
 
            # Timeline scrubber (only shown when keyframes exist)
            if keyframes:
                max_t = keyframes[-1][0]
                c, new_t = psim.SliderFloat("Timeline", playback_time, 0.0, max(max_t, 0.01))
                if c:
                    playback_time = new_t
                    playing = False     # pause when scrubbing manually
 
            psim.TextUnformatted(
                f"Keyframes: {len(keyframes)}   t = {playback_time:.2f}s")
 
            psim.TreePop()
 
        # ── Determine which pose to render ──────────────────────────────
        current_time = time.time()
        dt = current_time - last_frame_time
        last_frame_time = current_time
 
        if playing and keyframes:
            playback_time += dt * playback_speed[0]
            use_euler = get_interpolated_pose(playback_time, bone_names)
        else:
            use_euler = euler_angles
 
        # ── IK (disabled during playback) ────────────────────────────────
        any_ik_active = ENABLE_IK and any(ch.enabled for ch in ik_chains)
        if any_ik_active and not playing:
            for chain in ik_chains:
                if chain.enabled:
                    pc = ik_point_clouds[chain.name]
                    new_pos = np.array(pc.get_position(), dtype=np.float32)
                    if not np.allclose(new_pos, chain.target_position, atol=1e-4):
                        chain.target_position = new_pos.copy()
                        ik_changed = True
                    if changed or ik_changed:
                        solve_ik(chain, use_euler,
                                 pack_euler_angles(use_euler, controlled_indices, skeleton.num_bones),
                                 skeleton.parent_indices,
                                 skeleton.bind_local_transforms,
                                 device,
                                 max_iterations=ik_max_iter[0],
                                 learning_rate=10.0 ** ik_lr_exp[0])
 
        # ── FK + skinning ────────────────────────────────────────────────
        euler_np = pack_euler_angles(use_euler, controlled_indices, skeleton.num_bones)
        local_t  = compose_local_transforms_func(skeleton.bind_local_transforms, euler_np, device)
        world_t  = compute_world_transforms_func(skeleton.parent_indices, local_t, device)
        new_verts = linear_blend_skin(
            mesh_data.material_coords, mesh_data.bone_indices, mesh_data.bone_weights,
            world_t, device)
        wp.synchronize()
 
        wt_np = world_t.numpy()
        skeleton_net.update_node_positions(wt_np[:, :3].copy())
        mesh.update_vertex_positions(new_verts.numpy())
 
    # -----------------------------------------------------------------------
    ps.set_user_callback(callback)
    ps.show()
    return 0
 
 
if __name__ == "__main__":
    main()