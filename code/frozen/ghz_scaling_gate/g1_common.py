"""G1a protocol, deadlines and provenance (HANDOFF 5h, G1精简版). Compute nodes only (via ghz_runtime)."""
import copy
import math
import os
import time
from ghz_runtime import HERE, ROOT, cfg, read, write, sha, sources

G1_FILES = ['g1_config.json', 'g1a.sbatch', 'g1a_smoke.sbatch', 'g1a_pipeline.sh']


def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def g1cfg(smoke=False):
    """The frozen G1a protocol; smoke=True applies only the 'smoke' overrides (never a gate result)."""
    g = read(HERE/'g1_config.json')
    c = cfg()
    if g['stage'] != 'G1a' or g['version'] != 2 or g['new_final_queries'] != 0 or c['n'] != [2, 3, 4]:
        raise ValueError('Frozen G1a protocol; a changed scope needs a new config version')
    if not set(g['threshold']['families']) <= set(c['spectra']['families']):
        raise ValueError('Unknown threshold spectrum family')
    g['threshold'].setdefault('F_points', c['threshold_scan_points'])
    if smoke:
        g = merge(g, g['smoke'])
    g['mode'] = 'smoke' if smoke else 'full'
    return g


def workers(g):
    return max(1, min(g['parallel']['workers'], int(os.environ.get('SLURM_CPUS_PER_TASK', '18'))-2))


def job_start():
    return float(os.environ['QN_JOB_START'])


def elapsed():
    return time.time()-job_start()


def deadline(g):
    """No new work item starts after this time (the report keeps its reserve inside the allocation)."""
    t = g['time']
    return job_start() + 3600*t['limit_hours'] - 60*t['report_reserve_minutes']


def hard_deadline(g):
    """Running tasks stop between runs after this time (recorded NOT_FINISHED_DEADLINE), so the report step
    still fits inside the allocation even if a long task started just before `deadline`."""
    t = g['time']
    return job_start() + 3600*t['limit_hours'] - 60*t['hard_stop_minutes']


def g1_sources():
    s = sources()   # every *.py of this folder (G0 and G1), config.json, run.sbatch, frozen reused files
    for f in (f for f in G1_FILES if (HERE/f).exists()):  # PUBLIC: omitted job scripts skipped
        s[str((HERE/f).relative_to(ROOT))] = sha(HERE/f)
    return s


def clean(obj):
    """JSON-safe copy: non-finite floats become None, numpy scalars become Python numbers."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if hasattr(obj, 'item') and not isinstance(obj, (str, bytes)):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def finish(path, files, g, **extra):
    write(path/'manifest.json', clean(dict(status='COMPLETE', mode=g['mode'], g0_config=cfg(), g1_config=g,
                                           sources=g1_sources(), files={f: sha(path/f) for f in files},
                                           new_final_queries=0, final_accessed=False, **extra)))
