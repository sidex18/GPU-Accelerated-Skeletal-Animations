import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import warp as wp
from typing import Optional

from cli import parse_args

from skeleton import Skeleton
from mesh import Mesh
from forward_kinematics import compose_local_transforms_func, compute_world_transforms_func
from skinning import compute_material_coordinates, linear_blend_skin
from ik_solver import create_ik_chains, solve_ik

from utils import (
    load_weights
)

MAX_BIND_BONES = 4

# Set to True when you are ready to work on Part 3 (Inverse Kinematics)
ENABLE_IK = True

# Bone groups for the UI sliders
BONE_GROUPS = {
    "Spine": ["c_spine0", "c_spine1", "c_spine2", "c_spine3"],
    "Head": ["c_neck", "c_head"],
    "Left Arm": ["l_clavicle", "l_uparm", "l_lowarm"],
    "Right Arm": ["r_clavicle", "r_uparm", "r_lowarm"],
    "Left Leg": ["l_upleg", "l_lowleg", "l_foot"],
    "Right Leg": ["r_upleg", "r_lowleg", "r_foot"],
}


def pack_euler_angles(euler_dict, controlled_indices, num_bones):
    """Pack {bone_name: [rx,ry,rz]} dict into (num_bones, 3) numpy array."""
    out = np.zeros((num_bones, 3), dtype=np.float32)
    for name, idx in controlled_indices.items():
        out[idx] = euler_dict[name]
    return out


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    ENABLE_IK = args.IK
    wp.init()
    if args.device is None:
        device = wp.get_preferred_device()
    else:
        device = wp.get_device(args.device)

    # Load skeleton
    skeleton = Skeleton("data/skeleton_bind.json", device=device)

    # Load mesh vertex and face data
    mesh_data = Mesh("data/base_mesh.obj", device=device)

    # Load top-weighted bone indices and weights for each vertex
    vert_bone_indices, vert_bone_weights = load_weights(
        filename="data/vertex_weights.npz",
        num_verts=mesh_data.num_verts,
        name_to_idx=skeleton.name_to_idx,
        max_bind_bones=MAX_BIND_BONES)

    # Set vertex weights
    mesh_data.set_skinning_data(vert_bone_indices, vert_bone_weights)

    # Compute bind-pose world transforms (zero Euler angles)
    euler_np = np.zeros((skeleton.num_bones, 3), dtype=np.float32)
    bind_local_transforms = compose_local_transforms_func(
        skeleton.bind_local_transforms, euler_np, device)
    world_bind_transforms = compute_world_transforms_func(
        skeleton.parent_indices, bind_local_transforms, device)

    # Build bone edges from parent hierarchy (for skeleton visualization)
    edges = []
    for i in range(skeleton.num_bones):
        if skeleton.parent_indices[i] != -1:
            edges.append([skeleton.parent_indices[i], i])
    bone_edges = np.array(edges, dtype=np.int32)

    # Extract bind-pose bone positions from transforms
    # wp.transform numpy layout is (N, 7): [px, py, pz, qx, qy, qz, qw]
    wp.synchronize()
    world_np = world_bind_transforms.numpy()  # (num_bones, 7)
    bind_bone_positions = world_np[:, :3].copy()  # (num_bones, 3)

    # Compute material coordinates using bind-pose world transforms
    mesh_data.material_coords = compute_material_coordinates(
        mesh_data.verts, mesh_data.bone_indices, world_bind_transforms, device)

    # Initial rest-pose LBS using bind-pose world transforms
    updated_vertex_positions = linear_blend_skin(
        mesh_data.material_coords, mesh_data.bone_indices, mesh_data.bone_weights,
        world_bind_transforms, device)

    wp.synchronize()
    verts = updated_vertex_positions.numpy()
    faces = mesh_data.faces

    # --- Interactive state ---
    # Map controlled bone names -> skeleton index
    controlled_indices = {}
    for bones in BONE_GROUPS.values():
        for name in bones:
            if name in skeleton.name_to_idx:
                controlled_indices[name] = skeleton.name_to_idx[name]

    # Euler angles per controlled bone (radians)
    euler_angles = {name: [0.0, 0.0, 0.0] for name in controlled_indices}

    # --- Polyscope UI ---
    ps.init()
    ps.set_up_dir("y_up")
    mesh = ps.register_surface_mesh("man", verts, faces, smooth_shade=True)

    # Register skeleton curve network
    skeleton_net = ps.register_curve_network("skeleton", bind_bone_positions, bone_edges, radius=0.002)
    skeleton_net.set_color((0.2, 0.6, 1.0))

    camera_position = (0.0, 80.0, 250.0)
    look_at_target = (0.0, 80.0, 0.0)
    ps.look_at(camera_position, look_at_target)

    # --- IK state ---
    ik_chains = []
    ik_lr_exp = [-4.0]  # log10(learning_rate); -4 -> lr=0.0001
    ik_max_iter = [20]
    ik_point_clouds = {}

    if ENABLE_IK:
        ik_chains = create_ik_chains(skeleton.name_to_idx)

        # Get bind-pose world positions for initial targets
        for chain in ik_chains:
            chain.target_position = world_np[chain.end_effector_index, :3].copy()
            pc = ps.register_point_cloud(
                f"IK: {chain.name}", np.array([[0.0, 0.0, 0.0]]), radius=0.008
            )
            pc.set_position(tuple(chain.target_position))
            pc.set_transform_gizmo_enabled(True)
            pc.set_enabled(False)
            ik_point_clouds[chain.name] = pc

    def callback():
        changed = False
        ik_changed = False

        # --- Skeleton Visibility ---
        c, show_skel = psim.Checkbox("Show Skeleton", skeleton_net.is_enabled())
        if c:
            skeleton_net.set_enabled(show_skel)
        psim.Separator()

        # --- IK Controls ---
        if ENABLE_IK:
            if psim.TreeNode("Inverse Kinematics"):
                for chain in ik_chains:
                    c, chain.enabled = psim.Checkbox(
                        f"Enable {chain.name} IK", chain.enabled
                    )
                    if c:
                        pc = ik_point_clouds[chain.name]
                        if chain.enabled:
                            # Initialize target at current end-effector position via GPU FK
                            euler_np = pack_euler_angles(euler_angles, controlled_indices, skeleton.num_bones)
                            local_transforms = compose_local_transforms_func(
                                skeleton.bind_local_transforms, euler_np, device)
                            world_transforms = compute_world_transforms_func(
                                skeleton.parent_indices, local_transforms, device)
                            wp.synchronize()
                            wt_np = world_transforms.numpy()
                            ee_pos = wt_np[chain.end_effector_index, :3].copy()
                            chain.target_position = ee_pos
                            pc.set_position(tuple(ee_pos))
                            pc.set_enabled(True)
                            ik_changed = True
                        else:
                            pc.set_enabled(False)
                            changed = True  # snap: re-render with IK-solved euler_angles via FK path

                c, ik_lr_exp[0] = psim.SliderFloat(
                    "Learning Rate (1e)", ik_lr_exp[0], v_min=-5.0, v_max=-2.0
                )
                psim.TextUnformatted(f"  lr = {10.0 ** ik_lr_exp[0]:.6f}")
                c, ik_max_iter[0] = psim.SliderInt(
                    "Max Iterations", ik_max_iter[0], v_min=1, v_max=50
                )
                psim.TreePop()

            psim.Separator()

        # --- FK Controls ---
        if psim.Button("Reset All"):
            for name in euler_angles:
                euler_angles[name] = [0.0, 0.0, 0.0]
            changed = True

        for group_name, bone_names in BONE_GROUPS.items():
            if psim.TreeNode(group_name):
                for bone_name in bone_names:
                    if bone_name not in euler_angles:
                        continue
                    if psim.TreeNode(bone_name):
                        for axis_i, axis_label in enumerate(["X", "Y", "Z"]):
                            c, euler_angles[bone_name][axis_i] = psim.SliderAngle(
                                f"{axis_label}##{bone_name}",
                                euler_angles[bone_name][axis_i],
                                v_degrees_min=-180.0,
                                v_degrees_max=180.0,
                            )
                            if c:
                                changed = True
                        psim.TreePop()
                psim.TreePop()

        # --- Detect IK target movement ---
        if ENABLE_IK:
            for chain in ik_chains:
                if chain.enabled:
                    pc = ik_point_clouds[chain.name]
                    new_pos = np.array(pc.get_position(), dtype=np.float32)
                    if not np.allclose(new_pos, chain.target_position, atol=1e-4):
                        chain.target_position = new_pos.copy()
                        ik_changed = True

        # --- Solve IK + update mesh ---
        any_ik_active = ENABLE_IK and any(c.enabled for c in ik_chains)

        if changed or ik_changed or (any_ik_active and changed):
            euler_np = pack_euler_angles(euler_angles, controlled_indices, skeleton.num_bones)

            if any_ik_active:
                for chain in ik_chains:
                    if chain.enabled:
                        solve_ik(
                            chain,
                            euler_angles,
                            pack_euler_angles(euler_angles, controlled_indices, skeleton.num_bones),
                            skeleton.parent_indices,
                            skeleton.bind_local_transforms,
                            device,
                            max_iterations=ik_max_iter[0],
                            learning_rate=10.0 ** ik_lr_exp[0],
                        )
                # Re-compose after IK modified euler_angles
                euler_np = pack_euler_angles(euler_angles, controlled_indices, skeleton.num_bones)

            local_transforms = compose_local_transforms_func(
                skeleton.bind_local_transforms, euler_np, device)
            world_transforms = compute_world_transforms_func(
                skeleton.parent_indices, local_transforms, device)
            new_verts = linear_blend_skin(
                mesh_data.material_coords, mesh_data.bone_indices, mesh_data.bone_weights,
                world_transforms, device)
            wp.synchronize()
            wt_np = world_transforms.numpy()
            skeleton_net.update_node_positions(wt_np[:, :3].copy())
            mesh.update_vertex_positions(new_verts.numpy())

    ps.set_user_callback(callback)
    ps.show()

    return 0

if __name__ == "__main__":
    main()
