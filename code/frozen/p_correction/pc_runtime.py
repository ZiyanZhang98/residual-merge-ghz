"""Independent PC runtime. Every Python entry (including analysis) is compute-only."""
import os
import sys
import json
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = HERE.parent
ROOT = ARCH.parents[1]
sys.path.insert(0, str(ARCH/'positive_block_kernel'))
from block_runtime import require_compute
require_compute()
sys.path.insert(0, str(ARCH/'response_learning'))
import l_runtime as legacy
from rr_runtime import cfg as physical_config
from g1_common import g1cfg
from g1_core import Context


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    tmp.replace(path)


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


sha = legacy.sha


def config():
    c = read(HERE/'config.json')
    if (c['version'], c['stage'], c['new_final_queries']) != (3, 'PC', 0):
        raise ValueError('PC protocol version/final scope mismatch')
    if c['K'] != [4, 8, 16, 32, 64, 128, 256] or c['train']['seeds'] != [0, 1, 2]:
        raise ValueError('Change protocol version before changing the registered grid/seeds')
    return c


def sources():
    return dict(legacy.sources(), **{str(p.relative_to(ROOT)): sha(p) for p in HERE.iterdir()
        if p.suffix in ('.py', '.json', '.sh', '.sbatch')})


def new_root(path):
    p = Path(path).resolve()
    if not p.is_relative_to(HERE/'results'):
        raise ValueError('PC outputs must be below p_correction/results')
    p.mkdir(parents=True, exist_ok=False)
    return p


def artifact(root, filename, statuses, allow_profile=False):
    root = Path(root).resolve()
    m = read(root/'manifest.json')
    if m['status'] not in statuses or m['config'] != config():
        raise ValueError('Incomplete/incompatible PC artifact')
    if m['profile'] and not allow_profile:
        raise ValueError('Profile artifact cannot feed production')
    if not __import__('qn_public').same_sources(m['sources'], sources()):  # PUBLIC: see qn_public.py
        raise ValueError('PC artifact source inventory differs')
    if sha(root/filename) != m['files'][filename]:
        raise ValueError('PC artifact content changed: '+filename)
    return root/filename, m


def historical_pc0(root):
    """Explicit migration of immutable v1 PC0 evidence, never a training artifact.

    Verify all recorded output hashes and unchanged external physical sources.
    Changed PC interpretation code is expected and recorded as a post-PC0 revision.
    This does not manufacture unavailable multiplicative response sums or timings.
    """
    root = Path(root).resolve()
    m = read(root/'manifest.json')
    c = m['config']
    current = config()
    old_expected = {k: v for k, v in current.items() if k != 'revision'}
    old_expected['version'] = 1
    if (m['status'] not in ('PC0_COMPLETE', 'PC0_SHARD_COMPLETE') or m['profile']
            or c != old_expected or m.get('new_final_queries') != 0):
        raise ValueError('Only completed unchanged-protocol v1 PC0 evidence can be reselected')
    for filename, value in m['files'].items():
        if sha(root/filename) != value:
            raise ValueError('Historical PC0 output changed: '+filename)
    for filename, value in m['sources'].items():
        path = (ROOT/filename).resolve()
        if not path.is_relative_to(HERE) and sha(path) != value:
            raise ValueError('Frozen physical source changed: '+filename)
    return m


def context(root, hard, device):
    import torch
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested GPU unavailable')
    return Context(physical_config(), g1cfg(), torch.device(device), root, hard)


def manifest(root, mode, profile, status, inputs=None):
    return dict(mode=mode, status=status, profile=profile, config=config(), sources=sources(),
        job_id=os.environ.get('SLURM_JOB_ID', 'local'), inputs=inputs or {}, new_final_queries=0,
        files={p.name: sha(p) for p in root.iterdir() if p.is_file() and p.name != 'manifest.json'})
