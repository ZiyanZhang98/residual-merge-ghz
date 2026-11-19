"""Names shared by paired additive/multiplicative P correction estimators.

This module has no numerical imports, so artifact readers can use the same
unambiguous names without constructing a model or a physical simulator.
"""

MULTIPLICATIVE_SUFFIX = '__mul'
ESTIMATORS = ('add', 'mul')


def source_name(arm):
    """Return the response provider name associated with a correction arm."""
    return arm[:-len(MULTIPLICATIVE_SUFFIX)] if arm.endswith(MULTIPLICATIVE_SUFFIX) else arm


def estimator(arm):
    """Return ``raw``, ``add`` or ``mul`` for an output arm name."""
    if arm == 'M0':
        return 'raw'
    return 'mul' if arm.endswith(MULTIPLICATIVE_SUFFIX) else 'add'


def normalize_estimators(estimators):
    """Validate a nonempty, duplicate-free estimator selection."""
    selected = tuple(estimators)
    if not selected or len(set(selected)) != len(selected) or any(x not in ESTIMATORS for x in selected):
        raise ValueError('Estimators must be a nonempty subset of ("add", "mul")')
    return selected


def arm_name(source, method):
    if source == 'M0' or source.endswith(MULTIPLICATIVE_SUFFIX):
        raise ValueError('Reserved response source name: '+source)
    if method not in ESTIMATORS:
        raise ValueError('Unknown correction estimator: '+method)
    return source if method == 'add' else source+MULTIPLICATIVE_SUFFIX


def arm_names(sources, estimators=ESTIMATORS):
    """List correction arms, excluding M0, in provider/estimator order."""
    selected = normalize_estimators(estimators)
    return [arm_name(source, method) for source in sources for method in selected]
