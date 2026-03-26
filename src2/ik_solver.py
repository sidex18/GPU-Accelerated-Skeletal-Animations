import numpy as np
import warp as wp
from dataclasses import dataclass, field
from forward_kinematics import compose_local_transforms_func


@dataclass
class IKChain:
    name: str
    joint_names: list[str]
    joint_indices: list[int]
    end_effector_index: int
    enabled: bool = False
    target_position: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))


def create_ik_chains(name_to_idx: dict[str, int]) -> list[IKChain]:
    """Create the 4 limb IK chains."""
    chain_defs = [
        ("Left Arm",  ["l_uparm", "l_lowarm"], "l_wrist"),
        ("Right Arm", ["r_uparm", "r_lowarm"], "r_wrist"),
        ("Left Leg",  ["l_upleg", "l_lowleg"], "l_foot"),
        ("Right Leg", ["r_upleg", "r_lowleg"], "r_foot"),
    ]
    chains = []
    for name, joint_names, ee_name in chain_defs:
        joint_indices = [name_to_idx[n] for n in joint_names]
        ee_index = name_to_idx[ee_name]
        chains.append(IKChain(
            name=name,
            joint_names=joint_names,
            joint_indices=joint_indices,
            end_effector_index=ee_index,
        ))
    return chains

JOINT_ANGLE_LIMITS: dict[str, tuple[tuple[float, float], ...]] = {
    # Legs — hip: ball-and-socket; knee: hinge (Y/Z near-locked)
    "l_upleg":  ((-2.0, 0.5), (-0.8, 0.8), (-0.5, 0.5)),
    "r_upleg":  ((-2.0, 0.5), (-0.8, 0.8), (-0.5, 0.5)),
    "l_lowleg": ((0.0, 2.5),  (-0.1, 0.1), (-0.1, 0.1)),
    "r_lowleg": ((0.0, 2.5),  (-0.1, 0.1), (-0.1, 0.1)),
    # Arms — shoulder: ball-and-socket; elbow: hinge (Y/Z near-locked)
    "l_uparm":  ((-1.5, 3.0), (-1.5, 1.5), (-1.0, 1.0)),
    "r_uparm":  ((-1.5, 3.0), (-1.5, 1.5), (-1.0, 1.0)),
    "l_lowarm": ((-2.5, 0.0), (-0.1, 0.1), (-0.1, 0.1)),
    "r_lowarm": ((-2.5, 0.0), (-0.1, 0.1), (-0.1, 0.1)),
}


def _clamp_joint_angles(euler_angles: dict, limits: dict = JOINT_ANGLE_LIMITS):
    """Clamp per-bone Euler angles in-place to their joint limit ranges."""
    for bone_name, angle_limits in limits.items():
        if bone_name not in euler_angles:
            continue
        for k in range(3):
            lo, hi = angle_limits[k]
            euler_angles[bone_name][k] = max(lo, min(hi, euler_angles[bone_name][k]))


MAX_CHAIN_DEPTH = 10


# ─────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────

def _build_chain_path(chain: IKChain, parent_indices: np.ndarray):
    """Trace the bone chain from root to end-effector and build the euler_map.

    Returns:
        path_padded: (MAX_CHAIN_DEPTH,) int32 — bone indices, root->EE, zero-padded
        euler_map_padded: (MAX_CHAIN_DEPTH,) int32 — maps path index to chain euler
                          array index (-1 if uncontrolled)
        chain_len: int — actual number of bones in the path
    """
    path = []
    idx = chain.end_effector_index
    while idx != -1:
        path.append(idx)
        idx = parent_indices[idx]
    path.reverse()
    chain_len = len(path)
    assert chain_len <= MAX_CHAIN_DEPTH, (
        f"Chain depth {chain_len} exceeds MAX_CHAIN_DEPTH {MAX_CHAIN_DEPTH}"
    )

    joint_set = {bone_idx: j for j, bone_idx in enumerate(chain.joint_indices)}
    euler_map_list = [joint_set.get(b, -1) for b in path]

    path_padded = np.zeros(MAX_CHAIN_DEPTH, dtype=np.int32)
    path_padded[:chain_len] = path
    euler_map_padded = np.full(MAX_CHAIN_DEPTH, -1, dtype=np.int32)
    euler_map_padded[:chain_len] = euler_map_list

    return path_padded, euler_map_padded, chain_len


# ─────────────────────────────────────────────────────────────────────
# Warp kernels
# ─────────────────────────────────────────────────────────────────────

@wp.kernel
def chain_fk_loss(
    chain_path:       wp.array(dtype=wp.int32),
    chain_length:     wp.array(dtype=wp.int32),
    local_transforms: wp.array(dtype=wp.transform),
    bind_local_transforms: wp.array(dtype=wp.transform),
    euler_angles:     wp.array(dtype=wp.vec3),
    euler_map:        wp.array(dtype=wp.int32),
    target_pos:       wp.array(dtype=wp.vec3),
    loss:             wp.array(dtype=float),
):
    """Differentiable forward-kinematics loss for a single IK chain.

    Walk the chain from root to end-effector, accumulating a world-space
    wp.transform. Return ||EE_pos - target||^2.
    """
    n = chain_length[0]

    # Start with identity transform and accumulate along chain
    world_xform = wp.transform_identity()

    for i in range(MAX_CHAIN_DEPTH):
        if i < n:
            bone_idx = chain_path[i]
            euler_slot = euler_map[i]

            if euler_slot >= 0:
                angles = euler_angles[euler_slot]
                qx = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), angles[0])
                qy = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angles[1])
                qz = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), angles[2])
                q_euler = wp.mul(wp.mul(qx, qy), qz)
                a_k = wp.transform(wp.vec3(0.0, 0.0, 0.0), q_euler)
                local_xform = wp.transform_multiply(bind_local_transforms[bone_idx], a_k)
            else:
                
                local_xform = local_transforms[bone_idx]

            world_xform = wp.transform_multiply(world_xform, local_xform)

 
    ee_pos = wp.transform_get_translation(world_xform)
    target = target_pos[0]

  
    diff = ee_pos - target
    loss[0] = wp.dot(diff, diff)


@wp.kernel
def step_gd(
    x: wp.array(dtype=wp.vec3),
    grad: wp.array(dtype=wp.vec3),
    learning_rate: float,
):
    """Gradient descent step: x -= lr * grad, applied per-element."""
    tid = wp.tid()
    x[tid] = x[tid] - learning_rate * grad[tid]


# ─────────────────────────────────────────────────────────────────────
# IK solver
# ─────────────────────────────────────────────────────────────────────

def solve_ik(
    chain: IKChain,
    euler_angles: dict,
    full_euler_angles: np.ndarray,
    parent_indices: np.ndarray,
    bind_local_transforms: np.ndarray,
    device,
    max_iterations: int = 20,
    convergence_threshold: float = 0.1,
    learning_rate: float = 1.0,
) -> None:
    """Solve IK for a single chain using autodiff gradient descent.

    The kernel computes a scalar squared-distance loss between the
    end-effector and target. A single backward pass yields ∂Loss/∂θ,
    and we update θ -= lr * ∂Loss/∂θ each iteration.
    """
    num_joints = len(chain.joint_indices)

    # ==================================================================
    # Build chain path (PROVIDED)
    # ==================================================================
    path_padded, euler_map_padded, chain_len = _build_chain_path(chain, parent_indices)

    wp_chain_path = wp.array(path_padded, dtype=wp.int32, device=device)
    wp_chain_length = wp.array(np.array([chain_len], dtype=np.int32), dtype=wp.int32, device=device)
    wp_euler_map = wp.array(euler_map_padded, dtype=wp.int32, device=device)
    wp_target = wp.array(chain.target_position.reshape(1, 3).astype(np.float32), dtype=wp.vec3, device=device)

    # ==================================================================
    # Setup Warp arrays (PROVIDED)
    # ==================================================================

    # Bind-pose data
    wp_bind_xforms = wp.from_numpy(bind_local_transforms, dtype=wp.transform, device=device)

    # Pre-composed local transforms for non-controlled bones in the chain
    wp_local_transforms = compose_local_transforms_func(bind_local_transforms, full_euler_angles, device)

    # Gradient-enabled arrays (reused across iterations)
    chain_angles = np.zeros((num_joints, 3), dtype=np.float32)
    for j in range(num_joints):
        chain_angles[j] = euler_angles[chain.joint_names[j]]
    wp_euler = wp.array(chain_angles, dtype=wp.vec3, device=device, requires_grad=True)
    wp_loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)

    # ==================================================================
    # Gradient-descent optimisation loop
    # ==================================================================
    for iteration in range(max_iterations):
        # 1. Zero out gradients
        wp_euler.grad.zero_()
        wp_loss.grad.zero_()

        # 2. Record forward pass with wp.Tape
        tape = wp.Tape()
        with tape:
            wp.launch(
                chain_fk_loss,
                dim=1,
                inputs=[
                    wp_chain_path,
                    wp_chain_length,
                    wp_local_transforms,
                    wp_bind_xforms,
                    wp_euler,
                    wp_euler_map,
                    wp_target,
                ],
                outputs=[wp_loss],
                device=device,
            )

        # 3. Sync and check convergence
        wp.synchronize()
        loss_val = wp_loss.numpy()[0]
        if loss_val < convergence_threshold ** 2:
            break

        # 4. Seed loss gradient and backpropagate
        wp_loss.grad.fill_(1.0)
        tape.backward()

        # 5. Gradient descent update: euler -= lr * grad
        wp.launch(
            step_gd,
            dim=num_joints,
            inputs=[wp_euler, wp_euler.grad, learning_rate],
            outputs=[],
            device=device,
        )

        # 6. Wrap to [-pi, pi] and clamp to joint limits
        wp.synchronize()
        current_angles = wp_euler.numpy().copy()
        # Wrap angles
        current_angles = (current_angles + np.pi) % (2 * np.pi) - np.pi
        # Clamp to joint limits
        for j in range(num_joints):
            bone_name = chain.joint_names[j]
            if bone_name in JOINT_ANGLE_LIMITS:
                limits = JOINT_ANGLE_LIMITS[bone_name]
                for k in range(3):
                    lo, hi = limits[k]
                    current_angles[j, k] = max(lo, min(hi, current_angles[j, k]))
        # Write clamped values back
        wp_euler = wp.array(current_angles, dtype=wp.vec3, device=device, requires_grad=True)

    # ==================================================================
    # Unpack final angles back to dict
    # ==================================================================
    wp.synchronize()
    final_angles = wp_euler.numpy()
    for j in range(num_joints):
        bone_name = chain.joint_names[j]
        for k in range(3):
            euler_angles[bone_name][k] = float(final_angles[j][k])
        for k in range(3):
            euler_angles[bone_name][k] = (
                (euler_angles[bone_name][k] + np.pi) % (2 * np.pi) - np.pi
            )

    _clamp_joint_angles(euler_angles)
