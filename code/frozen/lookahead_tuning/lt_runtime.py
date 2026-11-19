"""Look-ahead tuning runtime (HANDOFF 5j item 22.22). Compute nodes only.

Imports hard_regime (tasks, Monte Carlo truth, look-ahead simulator) and the learned models
read-only; a separate directory keeps every completed artifact verifiable.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = HERE.parent
sys.path.insert(0, str(ARCH/'hard_regime'))
import hr_runtime as hr  # noqa: E402  (qoi_learn/qoi_adjoint/PC paths, require_compute)

ROOT, read, write, digest, sha, context = hr.ROOT, hr.read, hr.write, hr.digest, hr.sha, hr.context
HR_RESULTS = ARCH/'hard_regime'/'results'


def config():
    c = read(HERE/'config.json')
    if (c['version'], c['stage'], c['new_final_queries']) != (1, 'LT', 0):
        raise ValueError('LT protocol version/final scope mismatch')
    return c


def sources():
    own = {str(p.relative_to(ROOT)): sha(p) for p in sorted(HERE.iterdir())
           if p.suffix in ('.py', '.json', '.sh', '.sbatch')}
    return dict(hr.sources(), **own)


def new_root(path):
    p = Path(path).resolve()
    if not p.is_relative_to(HERE/'results'):
        raise ValueError('LT outputs must be below lookahead_tuning/results')
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
        raise ValueError('Incomplete/incompatible LT artifact: '+str(root))
    for filename, value in m['files'].items():
        if present(root/filename) and sha(root/filename) != value:
            raise ValueError('LT artifact content changed: '+str(root/filename))
    return m


def references(inputs, step1_job):
    """Monte Carlo truth and step-1 cells of hard_regime (verified, read-only)."""
    refs, npz, cells = [], {}, []
    for r in sorted(HR_RESULTS.glob('step1_%s_*' % step1_job)):
        hr.artifact(r, ['STEP1_SHARD_COMPLETE']); inputs[r.name] = sha(r/'manifest.json')
        m = read(r/'mc.json'); refs += m; cells += read(r/'cells.json')
        npz.update({x['task_id']: str(r/('mc_%s.npz' % x['task_id'])) for x in m})
    return refs, npz, cells
