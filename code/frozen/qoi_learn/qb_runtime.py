"""Step-B runtime (HANDOFF 5j item 22). Compute nodes only.

Imports step A (qoi_adjoint), PC and physics code read-only; a separate directory keeps
the completed step-A artifacts verifiable.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = HERE.parent
sys.path.insert(0, str(ARCH/'qoi_adjoint'))
import qa_runtime as qa  # noqa: E402  (PC paths, require_compute)

ROOT, read, write, append, digest, sha = qa.ROOT, qa.read, qa.write, qa.append, qa.digest, qa.sha
context = qa.context


def config():
    c = read(HERE/'config.json')
    if (c['version'], c['stage'], c['new_final_queries']) != (1, 'QB', 0):
        raise ValueError('QB protocol version/final scope mismatch')
    if c['K'] != [4, 8, 16, 32, 64, 128, 256]:
        raise ValueError('Registered K grid changed')
    return c


def sources():
    own = {str(p.relative_to(ROOT)): sha(p) for p in sorted(HERE.iterdir())
           if p.suffix in ('.py', '.json', '.sh', '.sbatch')}
    return dict(qa.sources(), **own)


def new_root(path):
    p = Path(path).resolve()
    if not p.is_relative_to(HERE/'results'):
        raise ValueError('QB outputs must be below qoi_learn/results')
    p.mkdir(parents=True, exist_ok=False)
    return p


def manifest(root, mode, status, inputs=None):
    return dict(mode=mode, status=status, config=config(), sources=sources(),
        job_id=os.environ.get('SLURM_JOB_ID', 'local'), array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
        array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'), inputs=inputs or {}, new_final_queries=0,
        files={p.name: sha(p) for p in sorted(root.iterdir()) if p.is_file() and p.name != 'manifest.json'})


DATA_CONFIG_KEYS = ('n', 'H', 'gamma', 'K', 'data', 'reference')
# Files that only train/evaluate. Changing them does not change generated data (revision 2026-10-21).
TRAINING_FILES = ('qb_train.py', 'qb_model.py', 'qb_pregate.py', 'qb_runtime.py', 'run.py',
                  'config.json', 'run.sbatch', 'decide.sbatch')


def data_artifact(root):
    """Accept a completed data shard if every data-generation source and data setting is unchanged."""
    root = Path(root).resolve()
    m = read(root/'manifest.json')
    if m['status'] != 'DATA_COMPLETE':
        raise ValueError('Incomplete data artifact: '+str(root))
    now = config()
    if any(m['config'][k] != now[k] for k in DATA_CONFIG_KEYS):
        raise ValueError('Data settings changed since: '+str(root))
    mine = str(HERE.relative_to(ROOT))+'/'
    fixed = lambda s: {k: v for k, v in s.items() if not (k.startswith(mine) and Path(k).name in TRAINING_FILES)}  # noqa: E731
    from qn_public import same_sources, present  # PUBLIC: see qn_public.py; data.pt is not distributed
    if not same_sources(fixed(m['sources']), fixed(sources())):
        raise ValueError('Data-generation sources changed since: '+str(root))
    for filename, value in m['files'].items():
        if present(root/filename) and sha(root/filename) != value:
            raise ValueError('Data artifact content changed: '+str(root/filename))
    return m


def read_verified(root, filename):
    """Read one file of an earlier artifact after checking its recorded hash."""
    root = Path(root).resolve()
    m = read(root/'manifest.json')
    if sha(root/filename) != m['files'][filename]:
        raise ValueError('Artifact file changed: '+str(root/filename))
    return read(root/filename)


def artifact(root, statuses, same_sources=True):
    root = Path(root).resolve()
    m = read(root/'manifest.json')
    if m['status'] not in statuses or m['config'] != config():
        raise ValueError('Incomplete/incompatible QB artifact: '+str(root))
    if same_sources and not __import__('qn_public').same_sources(m['sources'], sources()):  # PUBLIC: 
        raise ValueError('QB artifact produced by different sources: '+str(root))
    for filename, value in m['files'].items():
        if __import__('qn_public').present(root/filename) and sha(root/filename) != value:
            raise ValueError('QB artifact content changed: '+str(root/filename))
    return m
