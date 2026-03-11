import os


def get_home(models_home=None):
    """Return a path to the cache directory for trained autoencoders.

    This directory is then used by :func:`save`.

    """

    if models_home is None:
        models_home = os.environ.get(
            "CF_MODELS", os.path.join("~", "charmlab", "models", "autoencoders")
        )

    models_home = os.path.expanduser(models_home)
    if not os.path.exists(models_home):
        os.makedirs(models_home)

    return models_home