"""Paths, Snellius compute-node boundary, configuration and provenance for the BSC gate (HANDOFF 5i).

Reuses the GHZ gate runtime (ghz_scaling_gate/ghz_runtime.py: compute-node boundary, frozen G0
config, provenance) and the G1a protocol helpers (ghz_scaling_gate/g1_common.py). Importing this
module outside a Slurm compute node raises immediately, before any numerical import.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCH = HERE.parent
GHZ = ARCH/'ghz_scaling_gate'
sys.path.insert(0, str(GHZ))
import ghz_runtime  # noqa: E402  raises outside Slurm compute nodes; G0 config and provenance
from ghz_runtime import ROOT, read, write, sha  # noqa: E402,F401
import g1_common  # noqa: E402
sys.path.insert(0, str(HERE))

FILES = ['bsc_config.json', 'bsc.sbatch', 'bsc_smoke.sbatch', 'bsc_pipeline.sh']


def bcfg(smoke=False):
    """The frozen BSC gate protocol; smoke=True applies only the 'smoke' overrides (never a gate result)."""
    b = read(HERE/'bsc_config.json')
    if b['stage'] != 'BSC-Q' or b['version'] != 1 or b['new_final_queries'] != 0 or b['n'] != 4:
        raise ValueError('Frozen BSC gate protocol; a changed scope needs a new config version')
    if smoke:
        b = g1_common.merge(b, b['smoke'])
    b['mode'] = 'smoke' if smoke else 'full'
    return b


def g1a_root(b):
    return (HERE/b['g1a_results']).resolve()


def stage_dir(root, name):
    path = (Path(root)/name).resolve()
    if not path.is_relative_to(HERE/'results'):
        raise ValueError('Write new artifacts only below bsc_gate/results')
    path.mkdir(parents=True, exist_ok=False)
    return path


def sources():
    s = g1_common.g1_sources()   # GHZ gate code (G0, G1a), configs and the frozen reused files
    for p in sorted(HERE.glob('*.py')) + [HERE/f for f in FILES if (HERE/f).exists()]:  # PUBLIC: omitted job scripts skipped
        s[str(p.relative_to(ROOT))] = sha(p)
    return s


def finish(path, files, b, **extra):
    write(Path(path)/'manifest.json', g1_common.clean(dict(
        status='COMPLETE', mode=b['mode'], bsc_config=b, g0_config=ghz_runtime.cfg(), g1a_config=g1_common.g1cfg(),
        sources=sources(), files={f: sha(Path(path)/f) for f in files}, new_final_queries=0, final_accessed=False,
        **extra)))
