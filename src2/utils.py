import numpy as np


# For calculation efficiency, restrict dependent bones to MAX_BIND_BONES
def load_weights(filename: str, num_verts: int, name_to_idx: dict[str, int], max_bind_bones:int) -> tuple[np.ndarray, np.ndarray]:
        data = np.load(filename)
        # Initialize top-4 structure
        v_indices = np.full((num_verts, max_bind_bones), -1, dtype=np.int32)
        v_weights = np.zeros((num_verts, max_bind_bones), dtype=np.float32)

        # Temporary storage to sort all weights per vertex
        all_weights = [[] for _ in range(num_verts)]

        for bone_name in data.files:
            if bone_name not in name_to_idx:
                continue
            bone_idx = name_to_idx[bone_name]
            weights = data[bone_name]

            for v_idx, w in enumerate(weights):
                if w > 1e-5:
                    all_weights[v_idx].append((bone_idx, w))

        # Sort by weight descending, pick tops, re-normalize
        for v_idx in range(num_verts):
            sorted_influences = sorted(all_weights[v_idx], key=lambda x: x[1], reverse=True)[:max_bind_bones]

            total_w = sum(w for _, w in sorted_influences)
            for i, (b_idx, w) in enumerate(sorted_influences):
                v_indices[v_idx, i] = b_idx
                v_weights[v_idx, i] = w / total_w if total_w > 0 else 0.0

        return v_indices, v_weights
