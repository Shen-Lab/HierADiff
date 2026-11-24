"""Graph construction for inference from two unbound proteins."""

import numpy as np
import torch
from prody import ANM
from torch_geometric.data import HeteroData


def build_unbound_graph(item, n_nm=20):
    """Build the model-facing graph without bound structures."""

    graph = HeteroData()
    graph["name"] = item["path"]
    graph.has_bound = False

    center = (
        item["receptor_u_xyz"]
        .float()
        .mean(dim=0, keepdim=True)
    )
    graph.center = center

    for key in ("receptor", "ligand"):
        xyz = item[f"{key}_u_xyz"].float()
        bb = item[f"{key}_u_bb_coords"].float()
        seq = item[f"{key}_u_seq"]

        n = xyz.shape[0]

        if n <= n_nm // 3 + 2:
            raise ValueError(
                f"{key}: too few residues for {n_nm} modes"
            )

        assert xyz.shape == (n, 3)
        assert bb.shape == (n, 3, 3)
        assert len(seq) == n

        node = graph[key]

        # Center both structures using the unbound receptor.
        node.pos_u = xyz - center
        node.pos = node.pos_u.clone()
        node.bb_atoms_u = bb - center.view(1, 1, 3)

        # Inference has no residue-matching mask against GT.
        node.mask = torch.ones(n, dtype=torch.bool)

        # ESM-2 preprocessing will replace these names with embeddings.
        node.x_ori = seq
        node.x = None

        # Calculate normal modes directly from unbound C-alpha atoms.
        anm = ANM(f"{item['path']}_{key}")
        anm.buildHessian(xyz.detach().cpu().numpy())
        anm.calcModes(n_nm)

        vectors = anm.getEigvecs()
        values = anm.getEigvals()

        if vectors is None or vectors.shape != (3 * n, n_nm):
            raise ValueError(
                f"{key}: unexpected normal-mode dimensions"
            )

        modes = torch.from_numpy(
            np.asarray(vectors, dtype=np.float32)
            .reshape(n, 3, n_nm)
            .copy()
        )
        eigenvalues = torch.from_numpy(
            np.asarray(values, dtype=np.float64).copy()
        )

        node.u_nm = modes
        node.eig_vals_u = eigenvalues

        # Legacy sampler names: these are UNBOUND-derived modes.
        node.b_nm = modes.clone()
        node.eig_vals_b = eigenvalues.clone()
        node.cur_nm = None

        # Preserve the original edge construction convention.
        forward = torch.stack(
            [torch.arange(n - 1), torch.arange(1, n)],
            dim=0,
        )
        node.edge_index_u = forward
        node.edge_index_b = torch.cat(
            [forward, forward.flip(0)], dim=1
        )

        graph[key, "contact", key].edge_index = None

    return graph


# AUTO_RAW_PDB_INPUT
