# Learned Residual Merge Costs for Branch Compression in Multi-Round GHZ Purification Simulation

Code, configuration and result artifacts for the paper *Learning What Twirling Misses: Residual Merge Costs for Branch Compression in Multi-Round GHZ Purification Simulation*.

## Repository layout

```
ml_guided_pta_public_version/
├── README.md, requirements.txt, MANIFEST.tsv (sha256 + size of every file)
├── code/
│   ├── residual_merge/          # the method and every experiment of the paper
│   │   ├── config.json          # all settings, seeds, grids and the job ids that link the stages
│   │   ├── run.py               # entry 1: smoke | labels | labels_early | labels_report | train | deploy | analyze | diagnose
│   │   ├── run2.py              # entry 2: ext_* | test_* | n3_* | abl | figs | pta_sim (+ earlier variants)
│   │   ├── rm_core.py           # pairwise merge gaps (labels), budgeted agglomeration, C1 cost, adaptive simulator
│   │   ├── rm_labels.py         # label generation (exact tree for 3 rounds, coupled Monte Carlo beyond, minus PTA)
│   │   ├── rm_model.py          # pair MLP (3 x 256, SiLU), features, loss, training, LearnedCost
│   │   ├── rm_deploy.py         # running L / C1 on a task; required-atom computation; ratios
│   │   ├── rm_test.py, rm_test_report.py   # n=4 independent test set (pool, runs, registered analysis)
│   │   ├── rm_n3.py, rm_n3_report.py       # n=3 second setting
│   │   ├── rm_ext.py            # n=4 development set, extended grid
│   │   ├── rm_abl.py            # label-level ablation
│   │   ├── rm_pta.py            # PTA simulator on every resolved task
│   │   ├── rm_figs.py, rm_cmpfig.py, rm_causefig.py, rm_ptafig.py, rm_accfig.py   # figures
│   │   ├── rm_eval2.py, rm_lik.py, rm_signed.py, rm_curv.py   # earlier variants; kept because the analyses import them
│   │   └── results/             # archived artifacts (see "Archived results")
│   ├── frozen/                  # frozen simulator code this work builds on (imported read-only)
│   │   ├── ghz_scaling_gate/    # GHZ physics, noise letters, PTA letter, merging primitives (g1_core, ghz_*)
│   │   ├── qoi_adjoint/         # QoI simulator (qa_sim), PTA correction sources, metrics
│   │   ├── hard_regime/         # Monte Carlo references (hr_mc), task loading/scoring (hr_eval), fixed-K rollouts
│   │   ├── p_correction/        # preparation pools and isolation (pc_guard), runtime/context
│   │   ├── lookahead_truth/     # exact branch tree and coupled path estimator used for labels (lg_probe)
│   │   └── ...                  # the remaining folders are runtime links in the original import chain
│   └── slurm/                   # the job scripts used on the Snellius cluster (paths adapted to this tree)
├── paper_figures/               # the figures of the paper as generated (PDF/PNG) and the TikZ pipeline figure
└── logs/                        # Slurm logs go here
```

The `frozen/` folders keep their original names and layout because the code locates configs and artifacts relative to its own folder. Every `frozen/<folder>/config.json` is a frozen protocol file, and the runtimes refuse to run if it has been changed.

## Installation

```bash
conda create -n qn python=3.11 numpy scipy matplotlib pytorch -c conda-forge   # or: pip install -r requirements.txt
```

Every computation uses float64 in PyTorch. The archived runs used NVIDIA A100 GPUs (`--device cuda`). All modes also accept `--device cpu`.

Run Python from `code/residual_merge/`, or submit the Slurm scripts from the repository root. `rm_runtime.py` sets up every import path, so no package install is needed.

## Archived results (what the paper reports)

All artifacts sit in `code/residual_merge/results/<stage>_<slurm job id>[_<shard>]/`. Each directory has a

`manifest.json` that records:

- the status;
- the full configuration;
- the sha256 of every source file at run time;
- the inputs (with their hashes);
- the sha256 of every output file.

| Paper item | Artifact | Produced by |
|---|---|---|
| Frozen n=4 model L | `train_28736643/model.pt` | `run.py train` (labels `28727358` + early labels `28735174`) |
| Model trained without early labels (ablation) | `train_28734894/` | `run.py train` |
| n=4 label shards (summaries only) and noise report | `labels_28727358_*`, `labels_early_28735174_*`, `labels_report_28734447` | `run.py labels`, `labels_early`, `labels_report` |
| Sanity checks (identity, pairwise additivity, coupled vs exact) | `smoke_28722362/checks.json` | `run.py smoke` |
| n=4 development set (M3 grid; extended grid, Sec. VII) | `deploy_28737007_*`, `analyze_28737011`; `ext_28837517_*`, `exta_28837518/analysis.json` | `run.py deploy/analyze`; `run2.py ext_deploy/ext_analyze` |
| **n=4 independent test set (main result, Table 1)** | `testpool_28851716/pool.json`, `test_28851717_*/{cells,runs,mc}.json` + `mc_*.npz`, `testa_28851720/analysis.json` | `run2.py test_pool/test_run/test_analyze` |
| **n=3 second setting** | `labels_n3_28878732_*`, `labels_n3early_28878732_*`, `n3train_28878734/`, `n3eval_28878735_*`, `n3a_28878736/analysis.json` | `run2.py n3_labels/n3_train/n3_eval/n3_analyze` |
| Label-level ablation | `abl_28894051/ablation.json` (and the variant models) | `run2.py abl` |
| α calibration, break-even wall clock, forced-merge counts | `figs_28907449/{figs_data,points}.json` | `run2.py figs` |
| PTA simulator on all 217 resolved tasks | `pta_28921046/{pta_sim,cells}.json` | `run2.py pta_sim` |
| Mechanism examples (error before/after PTA correction) | `diagnose_28770883/` | `run.py diagnose` |
| Paper figures 2–7 | `cmpfig_28978070/`, `causefig_28978070/`, `ptafig_29033789/` (copies in `paper_figures/`) | `rm_cmpfig.py`, `rm_causefig.py`, `rm_ptafig.py`, `rm_accfig.py` |
| Consistency checks against earlier grids | `eval2_28780005_*`, `eval2_28793378_*`, `testsmoke_*` | read by the analyses |

The frozen side contributes these inputs:

- `frozen/qoi_learn/results/data_28437183_*/`: the n=3/4 development (monitor) tasks and their merge references.
  Only `tasks.json`, `refs/` and the manifests are included.
- `frozen/hard_regime/results/step1_28470804_*/`: the Monte Carlo references for the development and n=3 tasks, and
  the F2/M0/F2mul cells on them.
- `frozen/hard_regime/results/audit2/compare_28502410/`: the earlier PTA-alone numbers, which `pta_sim` checks it
  reproduces.
- `frozen/ghz_scaling_gate/results/g1a_28087770/evaluation/tasks_eval_items.json`: the metadata that the
  preparation pools must stay isolated from.

**Not distributed:**

- The label tensors `labels.pt` (n=4 and n=3; several GB).
- The 13 GB `data.pt` of the frozen data job.

Their manifests still list them. The pipeline below regenerates the labels.

## Reproducing

### 1. Paper numbers and figures from the archived cells (CPU, minutes)

```bash
cd code/residual_merge
python run2.py test_analyze --device cpu --minutes 30 --pool-job 28851716 --job 28851717 --out results/testa_local
python run2.py n3_analyze   --device cpu --minutes 30 --train-job 28878734 --job 28878735 --out results/n3a_local
python run2.py ext_analyze  --device cpu --minutes 30 --job 28837517 --out results/exta_local
python run2.py figs         --device cpu --minutes 30 --out results/figs_local
python run2.py pta_sim      --device cpu --minutes 60 --out results/pta_local      # runs the PTA simulator (~3 min)
python rm_cmpfig.py  --out results/cmpfig_local
python rm_causefig.py --out results/causefig_local
python rm_ptafig.py  --out results/ptafig_local
mkdir -p results/accfig_local && python rm_accfig.py results/accfig_local

# compare a regenerated file with the archived one
python ../../tools/compare_json.py results/testa_local/analysis.json results/testa_28851720/analysis.json
```

Output directories must be new subfolders of `results/`. The figure scripts and analyses take their input job ids
from `config.json` or from their defaults, which are the archived jobs.

### 2. Full pipeline (GPU cluster)

Submit from the repository root with the scripts in `code/slurm/`. Each script's header gives its exact usage.
`environment.sh` defines the Python command:

- an active conda environment named `$QN_ENV` (default `llm`), or
- `QN_PYTHON=/path/to/python`.

Set `QN_MODULES=""` to skip `module load`. Most stages read the job ids of the earlier stages from `config.json`.
After rerunning a stage, put the new job id there before you run the stages that depend on it. The stages, in order:

| # | Stage | Command (from the repository root) | config.json key(s) to update |
|---|---|---|---|
| 0 | Development MC references + F2/M0/F2mul | `sbatch --array=0-7%4 --time=HH:MM:SS --export=ALL,QN_HR_MODE=step1,QN_HR_MINUTES=<m>,QN_HR_SHARDS=8 code/slurm/hr_step1.sbatch` | `hard_regime.step1_job` |
| 1 | Sanity checks | `sbatch --export=ALL,QN_RM_MODE=smoke code/slurm/run.sbatch` | – |
| 2 | n=4 labels, rounds 3–33 and 0–2 | `sbatch --array=0-11%4 --export=ALL,QN_RM_SHARDS=12 code/slurm/labels.sbatch` (again with `QN_RM_LABEL_MODE=labels_early`) | `m2.labels_jobs` |
| 3 | Train L | `sbatch code/slurm/train.sbatch` | `s1.old_train_job`, `abl.frozen_train_job` |
| 4 | n=4 development (extended grid) + analysis | `ext.sbatch`, then `ext_cpu.sbatch` | `test.ext_deploy_job`, `test.ext_analysis_job` |
| 5 | n=4 test set: pool → runs → analysis | `test_pool.sbatch` → `test.sbatch` (12 shards) → `test_cpu.sbatch` | `figs.test_*`, `pta_sim.test_*` |
| 6 | n=3: labels → train → eval → analysis | `n3_labels.sbatch` → `n3_train.sbatch` → `n3_eval.sbatch` → `n3_cpu.sbatch` | `figs.n3_*` |
| 7 | Ablation, figures, PTA simulator | `abl.sbatch`, `figs.sbatch`, `pta_sim.sbatch`, `ptafig.sbatch`, `cmpfig.sbatch` | – |

The development set in stage 0 is read from `frozen/qoi_learn/results/data_28437183_*/tasks.json`. Regenerating
that data job (which also produced the training data of earlier, superseded models) is not part of this release.

Archived compute cost:

| Stage | Cost |
|---|---|
| Labels | about 25 GPU-min per shard |
| Training | about 3 min (n=4) and 1 min (n=3) on one A100 |
| n=4 test runs | 19.5 A100-h in total |
| n=3 labels + training + evaluation | about 6.8 A100-h |

### Determinism

- Fixed seeds cover every random choice: preparation pools, label pairs, coupled paths, Monte Carlo seeds per task,
  training, and the bootstrap.
- Within the archived A100/CUDA setup, reruns matched value by value. Examples: the n=3 runs reproduce 1344 F2/M0/F2mul
  cells of an earlier job, and `pta_sim` reproduces job `28502410`.
- On other hardware, floating-point differences can flip individual merge decisions on hard tasks. The individual
  cells (atoms, Q) can then differ, while the analysis pipeline itself stays exact.

## License

Released under the MIT License; see [LICENSE](LICENSE).
