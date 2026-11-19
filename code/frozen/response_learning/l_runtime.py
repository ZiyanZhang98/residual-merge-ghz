"""Stage L boundary and provenance; importing on a login node fails before NumPy/Torch."""
import sys
import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'response_recombination'))
import rr_runtime as rr
from rr_runtime import read, write, sha, clean, ROOT
from g1_core import guard
from block_runtime import require_compute


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    require_compute()
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def config():
    c = read(HERE/'config.json')
    if (c['version'], c['stage'], c['n'], c['H'], c['new_final_queries']) != (1, 'L', 4, 10, 0):
        raise ValueError('Stage L v1 scope mismatch')
    if c['state_q'] != 4 or c['label_depth'] != 2:
        raise ValueError('S4 party representation has q=n=4; B0 selected depth=2')
    return c


def sources():
    return dict(rr.sources(), **{str(p.relative_to(ROOT)): sha(p) for p in HERE.iterdir()
                if p.suffix in ('.py', '.json', '.sh', '.sbatch')})


def verify(mapping):
    for name, value in mapping.items():
        if sha(ROOT/name) != value:
            raise ValueError('Source/input changed: '+name)


def frozen(c):
    root = (HERE/c['calibration_root']).resolve()
    m = read(root/'manifest.json')
    if m['status'] != 'CALIBRATION_COMPLETE':
        raise ValueError('Completed B0 calibration required')
    verify(m['sources'])
    if sha(root/'selection.json') != m['selection_sha256']:
        raise ValueError('B0 selection changed')
    plan = read(root/'full_plan.json')
    if sha(root/'full_plan.json') != m['full_plan_sha256']:
        raise ValueError('B0 plan changed')
    cal, ev = plan['tasks']['calibration'], plan['tasks']['evaluation']
    cp, ep = {i['prep']['id'] for i in cal}, {i['prep']['id'] for i in ev}
    if cp & ep or not set(c['monitor_preparations']) < cp:
        raise ValueError('Preparation leakage or invalid monitor split')
    for i in cal+ev:
        guard(i['prep'])
    return root, m['config'], cal, ev


def new_root(path):
    path = Path(path).resolve()
    if not path.is_relative_to(HERE/'results'):
        raise ValueError('Outputs must be in response_learning/results')
    path.mkdir(parents=True, exist_ok=False)
    return path


def append(path, obj):
    import json
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(clean(obj), allow_nan=False)+'\n')


def load_dataset(path, c):
    import torch
    path = Path(path).resolve()
    m = read(path/'manifest.json')
    if m['status'] != 'DATA_COMPLETE' or m['config'] != c or m['profile']:
        raise ValueError('Require complete production data with identical configuration')
    verify(m['sources'])
    if sha(path/'data.pt') != m['data_sha256']:
        raise ValueError('Dataset changed')
    return torch.load(path/'data.pt', map_location='cpu', weights_only=False), m
