import numpy as np
import warp as wp


class Mesh:
    """Data container for mesh geometry and skinning data.

    Fields:
     - verts: Numpy array of vertex positions.
     - faces: Numpy array of face indices.
     - num_verts: Number of vertices.
     - bone_indices: Numpy array of bone indices per vertex (2D).
     - bone_weights: Numpy array of bone weights per vertex (2D).
     - material_coords: Numpy array of material coordinates (computed during binding).
    """
    def __init__(self, mesh_path: str, device="cpu"):
        self.device = device
        self.verts, self.faces, self.num_verts = self._load_mesh(mesh_path)

        # Skinning data (set later)
        self.bone_indices = None
        self.bone_weights = None
        self.material_coords = None

    def _load_mesh(self, filename: str) -> tuple[np.ndarray, np.ndarray, int]:
        """Load mesh data from OBJ file."""
        verts = []
        faces = []

        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("v "):
                    # Vertex line: v x y z
                    parts = line.split()
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith("f "):
                    # Face line: f v1 v2 v3 (indices are 1-based in OBJ format)
                    parts = line.split()
                    # Handle faces with texture/normal indices
                    face = []
                    for part in parts[1:]:
                        vertex_idx = int(part.split("/")[0]) - 1
                        face.append(vertex_idx)
                    faces.append(face)

        # Convert to numpy arrays
        verts = np.array(verts, dtype=np.float32)
        # Convert from Z-UP meters to Y-UP centimeters
        verts *= 100.0  # meters -> centimeters
        verts = verts[:, [0, 2, 1]]  # reorder columns: (x,y,z) -> (x,z,y)
        verts[:, 2] *= -1  # negate new Z: (x,z,y) -> (x,z,-y)
        faces = np.array(faces, dtype=np.int32)

        return verts, faces, len(verts)

    def set_skinning_data(self, bone_indices: np.ndarray, bone_weights: np.ndarray):
        """Set the bone indices and weights for skinning."""
        self.bone_indices = bone_indices
        self.bone_weights = bone_weights
