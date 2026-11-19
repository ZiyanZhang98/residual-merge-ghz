"""QoI adjoint runtime (HANDOFF 5j item 22). Compute nodes only.

Frozen p_correction and physics modules are imported read-only; changing them
would invalidate completed PC artifacts, so all new logic lives here.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = HERE.parent
ROOT = ARCH.parents[1]
PC = ARCH/'p_correction'
sys.path.insert(0, str(PC))
import pc_runtime as prt  # noqa: E402  (require_compute and physics paths)

read, write, append, digest, sha = prt.read, prt.write, prt.append, prt.digest, prt.sha


def config():
    c = read(HERE/'config.json')
    if (c['version'], c['stage'], c['new_final_queries']) != (1, 'QA', 0):
        raise ValueError('QA protocol version/final scope mismatch')
    if c['K'] != [4, 8, 16, 32, 64, 128, 256]:
        raise ValueError('Change protocol version before changing the registered grid')
    return c


def sources():
    own = {str(p.relative_to(ROOT)): sha(p) for p in sorted(HERE.iterdir())
           if p.suffix in ('.py', '.json', '.sh', '.sbatch')}
    return dict(prt.sources(), **own)


def new_root(path):
    p = Path(path).resolve()
    if not p.is_relative_to(HERE/'results'):
        raise ValueError('QA outputs must be below qoi_adjoint/results')
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
    if m['status'] not in statuses or m['config'] != config():
        raise ValueError('Incomplete/incompatible QA artifact: '+str(root))
    for filename, value in m['files'].items():
        if __import__('qn_public').present(root/filename) and sha(root/filename) != value:  # PUBLIC: 
            raise ValueError('QA artifact content changed: '+str(root/filename))
    return m


context = prt.context
