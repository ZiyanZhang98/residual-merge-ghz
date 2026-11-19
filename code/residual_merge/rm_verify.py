"""Verification of the M2 training artifact for later deployment/diagnostics (HANDOFF 22.28)."""
import rm_runtime

MODEL_FILES = ('rm_core.py', 'rm_labels.py', 'rm_runtime.py', 'rm_report.py', 'rm_model.py')
MODEL_KEYS = ('labels', 'm1', 'merge', 'm2', 'tolerance')


def verified_model(root):
    """A completed training artifact whose model-defining sources (labels, features, network) and settings are
    unchanged and whose files match their hashes; later deployment/analysis code may differ."""
    from pathlib import Path
    m = rm_runtime.read(root/'manifest.json')
    if m['status'] != 'TRAIN_COMPLETE':
        raise ValueError('Incomplete training artifact: '+str(root))
    cfg = rm_runtime.config()
    if any(m['config'][k] != cfg[k] for k in MODEL_KEYS):
        raise ValueError('Model settings changed since: '+str(root))
    mine = str(rm_runtime.HERE.relative_to(rm_runtime.REPO))+'/'
    keep = lambda s: {k: v for k, v in s.items() if not k.startswith(mine) or Path(k).name in MODEL_FILES}  # noqa: E731
    if not __import__('qn_public').same_sources(keep(m['sources']), keep(rm_runtime.sources())):  # PUBLIC: see qn_public.py
        raise ValueError('Model-defining sources changed since: '+str(root))
    for filename, value in m['files'].items():
        if __import__('qn_public').present(root/filename) and rm_runtime.sha(root/filename) != value:
            raise ValueError('Training artifact content changed: '+str(root/filename))
    return m
