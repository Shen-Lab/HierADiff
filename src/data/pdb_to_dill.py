"""Convert two unbound PDBs into a DIPS-compatible DILL pair."""

import csv
import dill
import hashlib
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser

from .data_train_utils_nm import Loader


def prepare_dips_input(input_dir):
    input_dir = Path(input_dir).resolve()
    receptor = input_dir / "receptor.pdb"
    ligand = input_dir / "ligand.pdb"

    for path in (receptor, ligand):
        if not path.is_file():
            raise FileNotFoundError(path)

    digest = hashlib.sha256()
    for path in (receptor, ligand):
        digest.update(path.read_bytes())

    # New input coordinates get a different preprocessing cache.
    root = input_dir / ".hieradiff_dips" / digest.hexdigest()[:16]
    filename = f"{input_dir.name}.dill"
    dill_path = root / "raw" / filename
    manifest = root / "pairs.csv"

    if not dill_path.is_file():
        loader = Loader.__new__(Loader)
        loader.parser = PDBParser(QUIET=True)

        def convert(path):
            residues, atoms, xyz, vis = loader.parse_pdb(str(path))
            atom_names = [atom[0] for atom in atoms]
            n_ca = atom_names.count("CA")

            if not n_ca or any(
                atom_names.count(name) != n_ca
                for name in ("N", "C")
            ):
                raise ValueError(
                    f"{path}: inconsistent N/CA/C backbone atoms"
                )

            return pd.DataFrame({
                "resname": residues,
                "atom_name": atom_names,
                "element": [atom[1] for atom in atoms],
                "x": xyz[:, 0].tolist(),
                "y": xyz[:, 1].tolist(),
                "z": xyz[:, 2].tolist(),
                "chain": vis["chain"],
                "residue": vis["residue"],
            })

        receptor_df = convert(receptor)
        ligand_df = convert(ligand)

        dill_path.parent.mkdir(parents=True, exist_ok=True)

        with dill_path.open("wb") as f:
            dill.dump([None, receptor_df, ligand_df], f)

    with manifest.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "split", "unbound_only"])
        writer.writerow([f"raw/{filename}", "test", "1"])

    return root, manifest


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m data.pdb_to_dill INPUT_DIRECTORY"
        )

    root, manifest = prepare_dips_input(sys.argv[1])
    print("DIPS_ROOT:", root)
    print("DIPS_MANIFEST:", manifest)
    print("PDB_TO_DILL_SUCCESS")
