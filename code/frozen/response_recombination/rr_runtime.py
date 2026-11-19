"""Compute-only runtime; reuse the existing Snellius boundary and frozen physics setup."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'bsc_gate'))
import bsc_runtime  # noqa: E402,F401; guard runs before any numerical import
from ghz_runtime import ROOT, read, write, sha, cfg, setup  # noqa: E402,F401
from g1_common import merge, clean  # noqa: E402,F401


def config(smoke=False):
    b = read(HERE/'config.json')
    if smoke:
        b = merge(b, b['smoke'])
    b['mode'] = 'smoke' if smoke else 'B0'
    if (b['version'], b['stage'], b['new_final_queries'], b['n'], b['H']) != (1, 'B0', 0, 4, 10):
        raise ValueError('Scope changed: revise the protocol version explicitly')
    if b['K_grid'] != sorted(set(b['K_grid'])) or b['replicates'] < 2:
        raise ValueError('Invalid K grid or replicate count')
    if b['supplement_max'] != 0:
        raise ValueError('No automatic supplementary sampling in protocol v1')
    if b.get('physics_backend') != 'sqrt':
        raise ValueError('B0 requires square-root physical propagation after smoke 28115797')
    ids = [a['id'] for a in b['candidates']]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate candidate IDs')
    return b


def sources():
    s = bsc_runtime.sources()
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.suffix in ('.py', '.json', '.sh', '.sbatch'):
            s[str(p.relative_to(ROOT))] = sha(p)
    return s


def input_sources(plan, b):
    """Freeze reused task metadata and selected accepted references, not just their filenames."""
    source_root = bsc_runtime.g1a_root(b)
    paths = {source_root/'gamma'/'gamma.json'}
    for split, tag in (('calibration', 'tasks_cal'), ('evaluation', 'tasks_eval')):
        paths.add(source_root/split/(tag+'_items.json'))
        paths.update((source_root/split/'rows').glob(tag+'_*.jsonl'))
    for items in plan['tasks'].values():
        paths.update(Path(i['source_reference']) for i in items if i['source_reference'])
    return {str(p.resolve()): sha(p) for p in sorted(paths)}


def new_root(path):
    path = Path(path).resolve()
    if not path.is_relative_to((HERE/'results').resolve()):
        raise ValueError('New outputs must be below response_recombination/results')
    path.mkdir(parents=True, exist_ok=False)
    for name in ('refs', 'trajectories'):
        (path/name).mkdir()
    return path
