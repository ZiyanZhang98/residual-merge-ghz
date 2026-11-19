"""Look-ahead ground-truth check runtime (HANDOFF 5j item 22.26). Compute nodes only.

Imports lookahead_oracle (and through it lookahead_tuning, physics_guided, hard_regime and the
learned models) read-only; a separate directory keeps every completed artifact verifiable.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = HERE.parent
sys.path.insert(0, str(ARCH/'lookahead_oracle'))
import lo_runtime as lo  # noqa: E402  (lookahead_tuning/physics_guided/hard_regime paths, require_compute)

hr, lt = lo.hr, lo.lt
ROOT, read, write, sha, context, references = lo.ROOT, lo.read, lo.write, lo.sha, lo.context, lo.references


def config():
    c = read(HERE/'config.json')
    if (c['version'], c['stage'], c['new_final_queries']) != (1, 'LG', 0):
        raise ValueError('LG protocol version/final scope mismatch')
    return c


def sources():
    own = {str(p.relative_to(ROOT)): sha(p) for p in sorted(HERE.iterdir())
           if p.suffix in ('.py', '.json', '.sh', '.sbatch')}
    return dict(lo.sources(), **own)


def new_root(path):
    p = Path(path).resolve()
    if not p.is_relative_to(HERE/'results'):
        raise ValueError('LG outputs must be below lookahead_truth/results')
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
        raise ValueError('Incomplete/incompatible LG artifact: '+str(root))
    for filename, value in m['files'].items():
        if present(root/filename) and sha(root/filename) != value:
            raise ValueError('LG artifact content changed: '+str(root/filename))
    return m
