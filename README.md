# Assignment 3: GPU-Accelerated Skeletal Animation

**Name:** (TODO)

**Student Number:** (TODO)

**CWL:** (TODO)

**Collaborators:** (TODO, or N/A)

**Resources/AI:** (TODO, or N/A)


## Overview

GPU-accelerated skeletal animation with Forward Kinematics (FK), Inverse Kinematics (IK), and Linear Blend Skinning (LBS), built on [NVIDIA Warp](https://github.com/NVIDIA/warp) with a [Polyscope](https://polyscope.run/) UI.

Entry point: `src/main.py`

## Setup

```bash
pip install warp-lang polyscope scipy numpy
```

(or from the same env in A2)

```bash
conda activate cpsc426
conda install scipy -y
```

## Running

```bash
python src/main.py              # uses preferred device (GPU if available)
python src/main.py --device cpu # force CPU execution
python src/main.py --IK         # enable IK for Part III
```

## Project Structure

```
src/
  main.py               — UI + orchestration (PROVIDED)
  cli.py                — CLI (PROVIDED)
  utils.py              — Data loaders (PROVIDED)
  skeleton.py           — Skeleton data container (PROVIDED)
  mesh.py               — Mesh data container (PROVIDED)
  forward_kinematics.py — FK kernels (STUDENT)
  skinning.py           — Skinning kernels (STUDENT)
  ik_solver.py          — IK kernels + solver loop (STUDENT, with provided setup)
data/
  skeleton_bind.json    — Bone hierarchy, bind-pose local matrices
  base_mesh.obj         — Character mesh (Z-up meters, converted to Y-up cm on load)
  vertex_weights.npz    — Per-bone vertex weights
  lod3.fbx              — Original full-body rig (reference, e.g. for Blender)
```

## What You Need to Implement

Please see the [A3 spec](./A3_spec.pdf) for detailed instructions, figures, and logic on how to implement each section of the code!

Recommended order: FK kernels -> Skinning kernels -> IK solver.