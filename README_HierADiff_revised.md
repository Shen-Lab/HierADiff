# HierADiff

**Induced-fit-inspired hierarchical generative modeling for flexible protein-protein docking**

HierADiff predicts protein-protein complexes from separately supplied unbound structures by coordinating global association with local backbone accommodation.

- **Global association:** protein-level rigid-body translation and rotation modeled by diffusion.
- **Local accommodation:** residue-level backbone transformations modeled by flow matching.
- **Target-adaptive scheduling:** the timing of local refinement depends on the predicted magnitude of binding-associated conformational change.
- **Confidence-based ranking:** generated structures are ranked using predicted contact-LDDT.

> **Repository status**
>
> This repository is currently a research-code release. The source tree, configurations, evaluation outputs, and an inference notebook are available, but the `ckpts/` directory does not currently contain the trained weights required to reproduce inference from a clean clone. The final paper checkpoint, matching configuration, and a tested end-to-end example should be added before describing the repository as fully reproducible.

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

The `.ipynb_checkpoints/` directory is generated automatically by Jupyter and should be removed from version control and added to `.gitignore`.

## Installation

The saved inference notebook reports Python 3.9.13. A release-ready version of the repository should include a pinned environment file, such as `environment.yml`, `requirements.txt`, or `pyproject.toml`.

Once an environment file is available, the expected setup will be similar to:

```bash
git clone https://github.com/Shen-Lab/HierADiff.git
cd HierADiff
conda env create -f environment.yml
conda activate hieradiff
```

Until the environment file is published, inspect imports under `src/` and the original computing environment before installing dependencies manually. Do not assume that both PyTorch and TensorFlow are required; the exact framework and versions should be documented from the paper environment.

## Checkpoints and configurations

The current inference notebook expects an experiment directory containing both the saved arguments and the trained model, for example:

```text
ckpts/DEBUG_flexFM_Rt_2piRound/
├── args.yaml
└── fold_0/
    └── model_best_47334_160_3.056_3.056.pth
```

The saved notebook output loaded this checkpoint from a user-specific scratch directory rather than the public repository. To make inference reproducible, publish:

- the exact DB5.5 fine-tuned checkpoint used for the paper;
- the matching `args.yaml` or a stable paper configuration;
- the DIPS-AF pretrained checkpoint, if required;
- any separate confidence-model checkpoint used for ranking;
- SHA-256 hashes and file sizes for downloadable weights.

## Inference

The current notebook invokes:

```bash
bash src/inf_nm.sh
```

The saved notebook output shows a 40-step sampling run. However, the notebook does not yet provide a portable end-to-end example because the required checkpoint is absent and the recorded run uses an absolute scratch path.

Before advertising inference as reproducible, document:

- the receptor and ligand PDB input format;
- preprocessing requirements;
- the paper checkpoint and configuration;
- the number of reverse-time steps;
- the number of generated structures;
- the confidence-ranking procedure;
- output filenames and directory structure;
- expected hardware and runtime.

### Recommended user-facing command

After the script interface is cleaned and tested, expose one command similar to:

```bash
bash src/inf_nm.sh \
  --config config/hieradiff_db55.yaml \
  --checkpoint ckpts/hieradiff_db55/model_best.pth \
  --receptor examples/receptor.pdb \
  --ligand examples/ligand.pdb \
  --output_dir results/example \
  --num_samples 40
```

> The command above illustrates the desired interface. Update the flags to match the actual script before presenting the command as verified.

## Minimal worked example

Add a small example to the repository:

```text
examples/
├── receptor.pdb
├── ligand.pdb
└── expected_output/
    ├── rank1_complex.pdb
    └── confidence.csv
```

A clean-clone test should generate at least one docked complex and its confidence score without requiring users to edit source files or hard-coded paths.

## Data

### DIPS-AF

DIPS-AF contains 38,887 protein-protein complexes paired with predicted unbound monomer structures and is used for pretraining.

For the archival release, provide:

- a permanent download location;
- preprocessing instructions;
- file formats and directory layout;
- license and access conditions;
- integrity hashes for downloadable archives.

### Docking Benchmark 5.5

HierADiff is fine-tuned and evaluated on DB5.5. Provide instructions for obtaining and preprocessing the benchmark without redistributing files that are governed by external terms.

## Reproducing the manuscript results

Document the following for the paper release:

- Git commit used for the manuscript;
- exact model, transform, loss, sampler, and confidence classes;
- paper configuration file;
- checkpoint identifiers;
- random seeds;
- number of reverse-time steps;
- 40 generated structures per target;
- contact-LDDT Top-1 ranking procedure;
- cRMSD and iRMSD evaluation scripts;
- hardware and timing protocol.

Create a tagged release such as `v1.0-paper` and cite the release or archived DOI in the manuscript.

## Known limitations

- Near-native recovery remains limited on DB5.5.
- Confidence-based ranking falls short of oracle selection.
- Reverse-time generated trajectories should not be interpreted as physical association pathways or binding kinetics.
- The current repository is not self-contained until the required checkpoints and matching configurations are published.

## Release checklist

- [ ] Add or link the final fine-tuned checkpoint.
- [ ] Add the matching paper configuration.
- [ ] Add any required pretrained and confidence checkpoints.
- [ ] Remove hard-coded scratch paths.
- [ ] Add a pinned environment specification.
- [ ] Add a tested receptor-ligand example.
- [ ] Test inference from a fresh clone.
- [ ] Clear large historical notebook outputs.
- [ ] Remove `.ipynb_checkpoints/` and add it to `.gitignore`.
- [ ] Distinguish paper code from legacy or experimental variants.
- [ ] Add `CITATION.cff`.
- [ ] Create a tagged paper release.

## Citation

```bibtex
@article{hieradiff,
  title   = {HierADiff: Induced-Fit-Inspired Hierarchical Adaptive Diffusion for Flexible Protein-Protein Docking},
  author  = {TODO},
  journal = {Protein Science},
  year    = {TODO},
  doi     = {TODO}
}
```

## License

This repository includes the GNU General Public License v3.0. See `LICENSE-GPLv3` and `license.md` for details.
