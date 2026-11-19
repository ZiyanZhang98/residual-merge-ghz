"""Oracle look-ahead check runtime (HANDOFF 5j item 22.23). Compute nodes only.

Imports lookahead_tuning (standardized look-ahead, LTSim), physics_guided (exact and
exact-prefix responses) and hard_regime (tasks, Monte Carlo truth) read-only; a separate
directory keeps every completed artifact verifiable.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = HERE.parent
sys.path.insert(0, str(ARCH/'physics_guided'))
sys.path.insert(0, str(ARCH/'lookahead_tuning'))
import lt_runtime as lt  # noqa: E402  (hard_regime/qoi_learn/qoi_adjoint/PC paths, require_compute)

hr = lt.hr
ROOT, read, write, sha, context, references = lt.ROOT, lt.read, lt.write, lt.sha, lt.context, lt.references
USED = [ARCH/'physics_guided'/'pg_sources.py', ARCH/'physics_guided'/'pg_runtime.py']


def config():
    c = read(HERE/'config.json')
    if (c['version'], c['stage'], c['new_final_queries']) != (1, 'LO', 0):
        raise ValueError('LO protocol version/final scope mismatch')
    return c


def sources():
    own = {str(p.relative_to(ROOT)): sha(p) for p in sorted(HERE.iterdir())
           if p.suffix in ('.py', '.json', '.sh', '.sbatch')}
    own.update({str(p.relative_to(ROOT)): sha(p) for p in USED})
    return dict(lt.sources(), **own)


def new_root(path):
    p = Path(path).resolve()
    if not p.is_relative_to(HERE/'results'):
        raise ValueError('LO outputs must be below lookahead_oracle/results')
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
        raise ValueError('Incomplete/incompatible LO artifact: '+str(root))
    for filename, value in m['files'].items():
        if present(root/filename) and sha(root/filename) != value:
            raise ValueError('LO artifact content changed: '+str(root/filename))
    return m
