"""No numerical imports before the compute-node boundary."""
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

HERE = Path(__file__).resolve().parent
ARCH = HERE.parent
ROOT = ARCH.parents[1]
ARMS = ('single_future', 'block_endpoint', 'block_future')


def require_compute():
    # PUBLIC: Snellius-only guard, enabled with QN_STRICT_SLURM=1 (see qn_public.py).
    from qn_public import STRICT_SLURM
    if not STRICT_SLURM:
        return
    host = socket.gethostname().split('.')[0]
    if not os.environ.get('SLURM_JOB_ID') or not host.startswith(('tcn', 'gcn', 'fcn', 'hcn')):
        raise RuntimeError('Only Slurm compute nodes may run this code, including imports/tests/reporting.')
    scratch = os.environ.get('TMPDIR', '')
    if not scratch or scratch == '/tmp' or scratch.startswith(('/tmp/', '/var/tmp')):
        raise RuntimeError('Slurm user scratch TMPDIR is required; no system /tmp fallback.')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    require_compute()
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    require_compute()
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def cfg():
    c = read(HERE/'config.json')
    if (c['arms'] != list(ARMS) or c['seed'] != 0 or c['menu'] != 'damped'
            or c['K'] != 16 or c['m'] != 4 or c['block_length'] != 4
            or c['max_updates'] > 4000 or c['train_seconds'] > 2700
            or c['new_final_queries'] != 0):
        raise ValueError('Not the v1 three-arm screen; revise protocol explicitly before changing scope.')
    return c


def bridge():
    require_compute()
    old = ARCH/'ml_predictive_closure'
    sys.path.insert(0, str(old))
    import runtime
    import physics
    sys.path.insert(0, str(HERE))
    return runtime, physics


def sources():
    old, _ = bridge()
    result = old.sources()
    for p in HERE.iterdir():
        if p.suffix in ('.py', '.json', '.sbatch', '.sh'):
            result[str(p.relative_to(ROOT))] = sha(p)
    for p in (ARCH.parent/'steps_0_2/manifest.json',):
        result[str(p.relative_to(ROOT))] = sha(p)
    return result


def verify(expected):
    for name, value in expected.items():
        if sha(ROOT/name) != value:
            raise ValueError('Provenance mismatch: '+name)


def load_manifest(directory):
    obj = read(Path(directory)/'manifest.json')
    if obj['status'] != 'COMPLETE' or obj['config'] != cfg() or obj['new_final_queries'] != 0:
        raise ValueError('Incomplete/incompatible block dataset')
    verify(obj['sources'])
    verify(obj['inputs'])
    for name, value in obj['files'].items():
        if sha(Path(directory)/name) != value:
            raise ValueError('Dataset hash mismatch: '+name)
    return obj


def output_dir(path):
    require_compute()
    p = Path(path).resolve()
    if not p.is_relative_to(HERE/'results'):
        raise ValueError('Write new artifacts only below positive_block_kernel/results')
    p.mkdir(parents=True, exist_ok=False)
    return p


def environment():
    require_compute()
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return dict(host=socket.gethostname(), job=os.environ['SLURM_JOB_ID'],
                torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                threads=torch.get_num_threads(), python=sys.version)
