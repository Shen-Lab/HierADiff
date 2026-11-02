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

