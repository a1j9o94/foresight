"""Experiment handlers package.

Each experiment has its own subpackage with handler implementations.
"""

from typing import Callable


def get_handlers_for_experiment(experiment_id: str) -> dict[str, Callable]:
    """Get handlers for a specific experiment.

    Args:
        experiment_id: Experiment ID (e.g., 'c1-vlm-latent-sufficiency')

    Returns:
        Dict mapping sub-experiment IDs to handler functions
    """
    # Extract experiment prefix (e.g., 'c1' from 'c1-vlm-latent-sufficiency')
    prefix = experiment_id.split("-")[0].lower()

    if prefix == "c1":
        from .c1 import get_handlers

        return get_handlers()
    elif prefix == "c2":
        from .c2 import get_handlers

        return get_handlers()
    elif prefix == "c3":
        from .c3 import get_handlers

        return get_handlers()
    elif prefix == "c4":
        from .c4 import get_handlers

        return get_handlers()
    elif prefix == "q1":
        from .q1 import get_handlers

        return get_handlers()
    elif prefix == "q2":
        from .q2 import get_handlers

        return get_handlers()
    elif prefix == "q3":
        from .q3 import get_handlers

        return get_handlers()
    elif prefix == "q4":
        from .q4 import get_handlers

        return get_handlers()
    elif prefix == "q5":
        from .q5 import get_handlers

        return get_handlers()
    else:
        return {}
