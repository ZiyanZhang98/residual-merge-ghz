"""Residual-merge runtime (HANDOFF 5j item 22.28, M0). Compute nodes only.

The code lives in ml_guided_pta/ but imports the frozen experiment code read-only through
lookahead_truth (exact tree, path sampling, coupled estimator), which in turn sets up the paths of
lookahead_oracle, lookahead_tuning, physics_guided, hard_regime, qoi_learn, qoi_adjoint, p_correction
and the physics modules. Importing p_correction's runtime enforces the Snellius rule: Slurm compute
nodes only, user scratch in TMPDIR.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ARCH = REPO/'code'/'frozen'  # PUBLIC: originally archived_research/predictive_state_claim/architecture_head_ablation
sys.path.insert(0, str(ARCH/'lookahead_truth'))
import lg_runtime as lg  # noqa: E402  (require_compute and every frozen path)

hr = lg.hr
read, write, sha, context, references = lg.read, lg.write, lg.sha, lg.context, lg.references


def config():
    c = read(HERE/'config.json')
    if (c['version'], c['stage'], c['new_final_queries']) != (1, 'RM', 0):
        raise ValueError('RM protocol version/final scope mismatch')
    return c


def sources():
    own = {str(p.relative_to(REPO)): sha(p) for p in sorted(HERE.iterdir())
           if p.suffix in ('.py', '.json', '.sh', '.sbatch')}
    return dict(lg.sources(), **own)


def new_root(path):
    p = Path(path).resolve()
    if not p.is_relative_to(HERE/'results'):
        raise ValueError('RM outputs must be below ml_guided_pta/residual_merge/results')
    p.mkdir(parents=True, exist_ok=False)
    return p


def manifest(root, mode, status, inputs=None):
    return dict(mode=mode, status=status, config=config(), sources=sources(),
        job_id=os.environ.get('SLURM_JOB_ID', 'local'), array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
        array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'), inputs=inputs or {}, new_final_queries=0,
        files={p.name: sha(p) for p in sorted(root.iterdir()) if p.is_file() and p.name != 'manifest.json'})


def artifact(root, statuses):
    root = Path(root).resolve()
    m = read(root/'manifest.json')
    from qn_public import same_sources, present  # PUBLIC: see qn_public.py
    if m['status'] not in statuses or m['config'] != config() or not same_sources(m['sources'], sources()):
        raise ValueError('Incomplete/incompatible RM artifact: '+str(root))
    for filename, value in m['files'].items():
        if present(root/filename) and sha(root/filename) != value:
            raise ValueError('RM artifact content changed: '+str(root/filename))
    return m
