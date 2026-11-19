"""Paths, Snellius compute-node boundary, config and provenance for the GHZ scaling gate.

Reuses the strict boundary of positive_block_kernel (Slurm job on tcn/gcn/fcn/hcn, user
scratch TMPDIR). Importing this module on a login node raises immediately.
"""
from pathlib import Path
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
ARCH = HERE.parent
ROOT = ARCH.parents[1]
sys.path.insert(0, str(ARCH/'positive_block_kernel'))
from block_runtime import require_compute, read, write, sha  # noqa: E402,F401
require_compute()
sys.path.insert(0, str(HERE))

# Frozen code this experiment reuses (cross-checks, clustering algorithm, job environment).
REUSED = [
    'positive_block_kernel/block_runtime.py', 'positive_block_kernel/job_environment.sh',
    'h3_short_sweep/environment.sh', '../steps_3_5/engine.py', '../steps_3_5/config.json',
    '../steps_0_2/run.py', '../../research_review/exp2/run_exp2.py', '../../research_review/exp2/config.json',
    '../../dejmps_experiment/simulator/dejmps.py', '../../dejmps_experiment/simulator/channels.py',
    '../../dejmps_experiment/simulator/linalg.py', '../../dejmps_experiment/data/pauli_basis.py',
]


def cfg():
    c = read(HERE/'config.json')
    if c['seed'] != 0 or c['n'] != [2, 3, 4] or c['new_final_queries'] != 0 or c['stage'] != 'G0':
        raise ValueError('Frozen G0 protocol; a changed scope needs a new config version')
    return c


def sources():
    files = sorted(HERE.glob('*.py')) + [HERE/'config.json', HERE/'run.sbatch']
    files += [(ARCH/p).resolve() for p in REUSED]
    return {str(p.relative_to(ROOT)): sha(p) for p in files if p.exists()}  # PUBLIC: omitted files skipped


def stage_dir(root, name):
    path = (Path(root)/name).resolve()
    if not path.is_relative_to(HERE/'results'):
        raise ValueError('Write new artifacts only below ghz_scaling_gate/results')
    path.mkdir(parents=True, exist_ok=False)
    return path


def finish(path, files, **extra):
    write(Path(path)/'manifest.json', dict(status='COMPLETE', config=cfg(), sources=sources(),
          files={f: sha(Path(path)/f) for f in files}, new_final_queries=0, **extra))


def engine():
    """The frozen d=4 engine (steps_3_5/engine.py): weighted Lloyd clustering, DEJMPS Operation, Pauli features."""
    spec = importlib.util.spec_from_file_location('ghz_frozen_engine', ROOT/'predictive_state_claim/steps_3_5/engine.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sys.path.insert(0, str(HERE))
    return module


def setup():
    import torch
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(cfg()['seed'])
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
