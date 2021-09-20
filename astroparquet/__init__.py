try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("astroparquet")
except PackageNotFoundError:
    # package is not installed
    pass

from .astroparquet import read_astroparquet, write_astroparquet

