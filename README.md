# HierADiff

**Induced-fit-inspired hierarchical generative modeling for flexible protein-protein docking**

HierADiff predicts protein-protein complexes from separately supplied unbound structures by coordinating global association with local backbone accommodation.

- **Global association:** protein-level rigid-body translation and rotation modeled by diffusion.
- **Local accommodation:** residue-level backbone transformations modeled by flow matching.
- **Target-adaptive scheduling:** the timing of local refinement depends on the predicted magnitude of binding-associated conformational change.
- **Confidence-based ranking:** generated structures are ranked using predicted contact-LDDT.

> **Repository status**
>
> The repository includes trained checkpoints, model configurations, inference scripts, and an example of protein-protein docking.

## Usage

### Inference example

An example using two unbound protein structures is provided in
`examples/1HCF_unbound/`:

- Receptor: `input/receptor.pdb` (208 residues)
- Ligand: `input/ligand.pdb` (84 residues)

Generate one prediction with 40 reverse-time steps:

    bash scripts/infer_unbound.sh examples/1HCF_unbound/input 7 40

The second argument specifies the random seed (default: 7), and
the third specifies the number of reverse-time steps (default: 40).

The generated PDB and prediction scores are saved under
`examples/1HCF_unbound/input/hieradiff_predictions/epoch-141/`.

Generate 20 structures with seeds 8–27:

    bash scripts/infer_unbound_many.sh examples/1HCF_unbound/input predictions

The script generates 20 structures using 40 reverse-time steps
per structure and ranks them by predicted contact-LDDT.

The example directory includes:

- `output/seed_8.pdb` through `output/seed_27.pdb`
- `ranked_poses.csv`, containing ranks, seeds, confidence scores,
  and corresponding PDB paths

### Preprocessing

The inference pipeline performs the following steps:

1. Convert receptor and ligand PDB files to DIPS-compatible DILL
   format using `src/data/pdb_to_dill.py`.
2. Construct residue-level graphs through DIPSLoader.
3. Compute ESM2 sequence embeddings and ProDy normal-mode features.
4. Predict iRMSD using the trained predictor for adaptive scheduling.
5. Generate docking structures using the HierADiff sampler.

### Confidence ranking

For each generated structure, the docking model predicts a
contact-LDDT confidence score.

Scores are stored under the `pLDDT` key in `all_results.pkl`.
The multi-seed inference script sorts predictions by descending
confidence and writes `ranked_poses.csv` and ranked PDB files.

The standalone confidence model is provided separately from
the confidence prediction used in this inference workflow.

### Model checkpoints

The final docking model is:

`ckpts/DEBUG_db5_ft_flexFM_Rt_interSWReadoutSW_adaptive_schedule_fape_bothNMFeat_readOutInterAtt/fold_0/model_best_2484_141_6.235_6.235.pth`

The inference launcher uses `model_every_141.pth` from the
same directory, which contains identical model weights.

Other checkpoints:

- Adaptive iRMSD predictor:
  `ckpts/DEBUG_db5_iRMSD_pred_contr/`
- Standalone confidence model:
  `ckpts/DEBUG_confidence_db5_finalSteps/`
- Ablation models: Model0, Model_NMFeatures,
  and Model_adaptive_noPretrain

The matching configuration files are provided under `config/`,
and saved experiment arguments are stored in the corresponding
checkpoint directories.

### Computing environment

| Package | Version |
|---|---|
| Python | 3.9.20 |
| PyTorch | 1.12.1+cu113 |
| CUDA (PyTorch build) | 11.3 |
| PyTorch Geometric | 2.3.1 |
| e3nn | 0.5.4 |
| torch-scatter | 2.1.2 |
| torch-cluster | 1.6.0+pt112cu113 |
| torch-sparse | 0.6.18 |
| NumPy | 1.23.5 |
| SciPy | 1.10.1 |
| Biopython | 1.79 |
| ProDy | 2.4.1 |
| fair-esm | 2.0.0 |
| pandas | 2.2.3 |
| dill | 0.3.9 |
| PyYAML | 6.0.2 |

The environment specifications are provided in
`environment.yml` and `environment-linux-64.lock.txt`.

### Training and evaluation

| Purpose | Implementation |
|---|---|
| Training | `src/main_nm.py` |
| Checkpoint selection | `src/train_nm.py` |
| Docking model | `src/model/model_nm.py` |
| Score network | `src/model/diffusion_flexFM_Rt.py` |
| Noise schedule | `src/geom_utils/transform_flexFM_Rt.py` |
| Residue-level and FAPE losses | `src/model/losses_flexFM_Rt.py` |
| Docking inference | `src/main_inf_nm_1gpu.py` |
| Sampling | `src/sample_flexFM_Rt.py` |
| Confidence inference | `src/main_inf_conf_1gpu.py` |
| Confidence sampling | `src/sample_flexFM_Rt_conf.py` |
| cRMSD/iRMSD evaluation | `src/evaluation/compute_rmsd_nm.py` |

The final docking experiment uses:

`config/db5_flexFM_Rt_interSWReadoutSW_adaptive_schedule_fape_bothNMFeat_readOutInterAtt.yaml`

The final checkpoint was selected by minimum validation loss.
Training resumed from `model_last.pth` with command-line seed 9.
The saved experiment configuration records seed 803.

Complex RMSD and interface RMSD are computed using
`RMSDComputer.update_complex_rmsd` and
`RMSDComputer.update_interface_rmsd`, respectively.

## Data availability

The processed DB5.5 and DIPS-AF datasets used in HierADiff are available on [Zenodo][https://doi.org/10.5281/zenodo.24491375].

The archives include bound and unbound protein structures in DILL format, train/validation/test splits, and residue correspondence masks. Graph representations and other derived features can be generated using the preprocessing code provided in this repository.


## Abstract

Structural prediction of protein-protein complexes is important for mechanistic understanding and therapeutic discovery, yet structure-based flexible docking remains challenging when binding is accompanied by conformational change. We present **HierADiff**, an induced-fit-inspired generative framework that models docking from separately supplied unbound structures as coupled global association and local backbone accommodation. Global protein-level rigid-body motions are modeled by diffusion, whereas local residue-level transformations are modeled by flow matching with a distinct schedule. A target-adaptive local schedule is conditioned on the predicted magnitude of binding-associated conformational change, allowing the progression of backbone refinement to vary across targets. The global and local processes are coupled through a shared score-and-confidence network that integrates sequence, evolutionary, structural, and normal-mode-derived dynamics features while maintaining the required transformation properties of its outputs. To support pretraining, we curate DIPS-AF, a dataset of 38,887 protein-protein complexes with paired bound and predicted unbound monomer structures, and fine-tune the model on the training split of Docking Benchmark 5.5 (DB5.5). On the DB5.5 test set, **HierADiff** achieves lower mean and median complex RMSD and interface RMSD than DiffDock-PP, a rigid-body diffusion model, and GeoDock, a flexible iterative-refinement model. Ablation results support the contributions of the adaptive schedule and DIPS-AF pretraining. However, near-native recovery remains limited, and confidence-based ranking falls short of oracle selection. These results support hierarchical and target-adaptive coordination of global association and local backbone accommodation as a promising direction for structure-based flexible docking.

## Method overview

HierADiff represents flexible docking at two coupled resolutions:

1. **Protein level**
   - The receptor is fixed as the reference partner.
   - The ligand undergoes global translation and rotation.
   - Global rigid-body association is sampled through reverse-time diffusion.

2. **Residue level**
   - Receptor and ligand residues are represented by backbone frames.
   - Flow matching predicts local translation and rotation corrections toward the bound state.
   - The local schedule is adapted separately for the two partners using predicted interface conformational change.

The shared network combines:

- amino-acid sequence features;
- evolutionary features;
- geometric and structural features;
- normal-mode-derived flexibility features;
- equivariant translation and rotation outputs;
- an invariant contact-LDDT confidence output.

## Repository structure

```text
HierADiff/
├── ckpts/             # Model checkpoints and saved experiment arguments
├── config/            # Model and experiment configurations
├── datasets/          # Dataset definitions and preprocessing resources
├── eva_out/           # Raw evaluation outputs
├── jobs/              # Cluster job-submission scripts
├── out_jobs/          # Scheduler logs and job outputs
├── results/           # Summarized experimental results
├── src/               # Models, data processing, sampling, training, and evaluation
├── visualization/     # Visualization and figure-generation scripts
├── LICENSE-GPLv3      # GNU GPL v3 license
└── license.md         # Human-readable license information
```

