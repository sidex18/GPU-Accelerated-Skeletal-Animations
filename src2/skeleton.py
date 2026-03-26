import numpy as np
import json
import warp as wp
from scipy.spatial.transform import Rotation


class Skeleton:
    """Data container for skeleton hierarchy and bind-pose transforms.

    Fields:
     - bone_names: List of bone names.
     - name_to_idx: Mapping from bone names to their indices.
     - parent_indices: Numpy array of parent indices for each bone.
     - bind_local_transforms: Numpy array of bind-pose local transforms (tx,ty,tz,qx,qy,qz,qw).
     - bind_local_quats: Numpy array of bind-pose local quaternions (x,y,z,w) [convenience].
     - bind_local_trans: Numpy array of bind-pose local translations [convenience].
    """
    def __init__(self, skeleton_path: str, device="cpu"):
        self.device = device
        self.bone_names, self.name_to_idx, self.parent_indices, bind_local_mats = self._load_skeleton(skeleton_path)
        self.num_bones = len(self.bone_names)

        # Derive bind-pose local transforms (translation + quaternion) as a flat array
        # Warp uses a 7-element transform layout: [tx,ty,tz, qx,qy,qz,qw]
        quats = Rotation.from_matrix(bind_local_mats[:, :3, :3]).as_quat().astype(np.float32)
        trans = bind_local_mats[:, :3, 3].astype(np.float32)
        self.bind_local_transforms = np.zeros((self.num_bones, 7), dtype=np.float32)
        self.bind_local_transforms[:, :3] = trans
        # note: Rotation.as_quat returns [x,y,z,w]
        self.bind_local_transforms[:, 3:] = quats

        # for convenience (used by IK), still expose quats/trans separately
        # self.bind_local_quats = quats
        # self.bind_local_trans = trans

    def _load_skeleton(self, filename: str) -> tuple[list[str], dict[str, int], np.ndarray, np.ndarray]:
        """Load skeleton data from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)

        bone_names = list(data.keys())
        name_to_idx = {name: i for i, name in enumerate(bone_names)}
        num_bones = len(bone_names)

        parent_indices = np.full(num_bones, -1, dtype=np.int32)
        bind_local_mats = np.zeros((num_bones, 4, 4), dtype=np.float32)

        for name, info in data.items():
            idx = name_to_idx[name]
            if info['parent']:
                parent_indices[idx] = name_to_idx[info['parent']]

            bind_local_mats[idx] = np.array(info['matrix_local'])

        return bone_names, name_to_idx, parent_indices, bind_local_mats
