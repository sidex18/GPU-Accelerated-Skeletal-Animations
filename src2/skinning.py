import warp as wp
import numpy as np

@wp.kernel
def compute_material_coords(
    vert_world: wp.array(dtype=wp.vec3),
    vert_bone_indices: wp.array(dtype=wp.int32, ndim=2),
    world_transforms: wp.array(dtype=wp.transform),
    vert_bone: wp.array(dtype=wp.vec3, ndim=2)
):
    """Transform world-space vertices into bone-local (material) coordinates
    using inverse bind transforms, for use in linear blend skinning."""
    # TODO: Implement material coordinate computation
    tid = wp.tid()
    pos = vert_world[tid]

    bone_count = vert_bone_indices.shape[1]

    for i in range(bone_count):
        bone_idx = vert_bone_indices[tid, i]
        bone_idx = vert_bone_indices[tid, i]

        if bone_idx == -1:
            vert_bone[tid, i] = wp.vec3(0.0, 0.0, 0.0)
        else:
            inv_bind = wp.transform_inverse(world_transforms[bone_idx])
            vert_bone[tid, i] = wp.transform_point(inv_bind, pos)



@wp.kernel
def compute_vertex_positions(
    vert_bone_pos: wp.array(dtype=wp.vec3, ndim=2),
    vert_bone_indices: wp.array(dtype=wp.int32, ndim=2),
    vert_bone_weights: wp.array(dtype=wp.float32, ndim=2),
    world_transforms: wp.array(dtype=wp.transform),
    out_verts: wp.array(dtype=wp.vec3),
):
    """Linear blend skinning: blend bone-transformed material coordinates
    using wp.transform_point and per-vertex bone weights."""
    # TODO: Implement linear blend skinning
    tid = wp.tid()
    final_pos = wp.vec3(0.0, 0.0, 0.0)
    bone_count = vert_bone_weights.shape[1]

    for i in range(bone_count):
        w = vert_bone_weights[tid, i]
        bone_ind = vert_bone_indices[tid, i]
        if bone_ind == -1:
            continue
        if w < 1e-6:
            continue

       
        mat_coord = vert_bone_pos[tid, i]
        animated_pos = wp.transform_point(world_transforms[bone_ind], mat_coord)
        final_pos = final_pos + w * animated_pos

    out_verts[tid] = final_pos


def compute_material_coordinates(verts: np.ndarray, bone_indices: np.ndarray,
                                world_bind_transforms: wp.array, device) -> wp.array:
    """Compute material coordinates for skinning.

    Args:
        verts: Vertex positions (num_verts, 3)
        bone_indices: Bone indices per vertex (num_verts, max_bones)
        world_bind_transforms: World bind transforms (Warp array, dtype=wp.transform)
        device: Warp device

    Returns:
        Material coordinates array (Warp, num_verts, max_bones, 3)
    """
    num_verts = len(verts)
    max_bones = bone_indices.shape[1]

    wp_verts = wp.from_numpy(verts, dtype=wp.vec3, device=device)
    wp_bone_indices = wp.from_numpy(bone_indices, dtype=wp.int32, device=device)
    wp_material_coords = wp.zeros((num_verts, max_bones), dtype=wp.vec3, device=device)

    wp.launch(
        kernel=compute_material_coords,
        dim=num_verts,
        inputs=[wp_verts, wp_bone_indices, world_bind_transforms],
        outputs=[wp_material_coords],
        device=device
    )

    return wp_material_coords


def linear_blend_skin(material_coords: wp.array, bone_indices: np.ndarray,
                     bone_weights: np.ndarray, world_transforms: wp.array,
                     device) -> wp.array:
    """Perform linear blend skinning.

    Args:
        material_coords: Material coordinates (Warp array, num_verts, max_bones, 3)
        bone_indices: Bone indices per vertex (num_verts, max_bones)
        bone_weights: Bone weights per vertex (num_verts, max_bones)
        world_transforms: World transforms (Warp array, dtype=wp.transform)
        device: Warp device

    Returns:
        Deformed vertex positions (Warp array)
    """
    num_verts = material_coords.shape[0]

    wp_bone_indices = wp.from_numpy(bone_indices, dtype=wp.int32, device=device)
    wp_bone_weights = wp.from_numpy(bone_weights, dtype=wp.float32, device=device)
    wp_deformed_verts = wp.zeros(num_verts, dtype=wp.vec3, device=device)

    wp.launch(
        kernel=compute_vertex_positions,
        dim=num_verts,
        inputs=[material_coords, wp_bone_indices, wp_bone_weights,
                world_transforms],
        outputs=[wp_deformed_verts],
        device=device
    )

    return wp_deformed_verts
