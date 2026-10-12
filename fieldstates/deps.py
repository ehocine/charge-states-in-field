"""Locate chargestates (paper 2, release 1.0.4 of https://github.com/ehocine/impurity-charge-states,
doi:10.5281/zenodo.23325384), which in turn locates collapse2d (paper 1, doi:10.5281/zenodo.23123320).

Search order: an importable chargestates; $CHARGESTATES_ROOT; ./external/impurity-charge-states;
~/Desktop/impurity-charge-states.
"""
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CANDIDATES = [os.environ.get("CHARGESTATES_ROOT"), os.path.join(_HERE, "external", "impurity-charge-states"),
               os.path.expanduser("~/Desktop/impurity-charge-states")]

try:
    import chargestates  # noqa: F401
except ImportError:
    for _root in _CANDIDATES:
        if _root and os.path.isdir(os.path.join(_root, "chargestates")):
            sys.path.insert(0, _root)
            break
    try:
        import chargestates  # noqa: F401,E402
    except ImportError as err:
        raise ImportError(
            "chargestates not found. Run\n"
            "  git clone --branch 1.0.4 https://github.com/ehocine/impurity-charge-states external/impurity-charge-states\n"
            "or set CHARGESTATES_ROOT to a checkout of release 1.0.4.") from err

from chargestates.deps import collapse2d  # noqa: E402,F401
