from typing import Optional

from scarlet2.observation import Observation
from scarlet2.scene import Scene


def check_observation(observation: Optional[Observation] = None):
    """Check the observation object for consistency

    Parameters
    ----------
    observation: Observation, optional
        The observation object to check. If None, no checks are performed. Default is None
    """
    if observation is None:
        return

    pass


def check_scene(scene: Optional[Scene] = None):
    """Check the scene against the various validation rules.

    Parameters
    ----------
    scene : Scene, optional
        The scene object to check, by default None
    """
    if scene is None:
        return

    pass


def check_fit(scene: Optional[Scene] = None):
    """Check the scene after fitting against the various validation rules.

    Parameters
    ----------
    scene : Scene, optional
        The scene object to check, by default None
    """
    if scene is None:
        return

    pass
