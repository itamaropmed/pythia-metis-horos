'''Deprecated alias package: `peras` was renamed to `pythia` on 2026-10-18.

`import peras` and `from peras.x.y import z` keep working: every `peras.*` module is a thin proxy whose attributes
are the objects of the matching `pythia.*` module (same functions and classes, so identity and isinstance hold).
New code should import `pythia` directly.
'''
import importlib
import importlib.abc
import importlib.util
import sys
import types
import warnings

_OLD, _NEW = "peras", "pythia"


class _Proxy(types.ModuleType):
    def __init__(self, name, real):
        super().__init__(name)
        object.__setattr__(self, "_real", real)
        self.__doc__ = real.__doc__

    def __getattr__(self, key):
        return getattr(object.__getattribute__(self, "_real"), key)

    def __dir__(self):
        return dir(object.__getattribute__(self, "_real"))


class _AliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name.startswith(_OLD + "."):
            real = importlib.import_module(_NEW + name[len(_OLD):])
            return importlib.util.spec_from_loader(name, self, is_package=hasattr(real, "__path__"))
        return None

    def create_module(self, spec):
        return _Proxy(spec.name, importlib.import_module(_NEW + spec.name[len(_OLD):]))

    def exec_module(self, module):
        pass


if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _AliasFinder())

warnings.warn(f"'{_OLD}' was renamed to '{_NEW}'; please update imports.", DeprecationWarning, stacklevel=2)
_real = importlib.import_module(_NEW)
__all__ = list(getattr(_real, "__all__", []))
globals().update({k: getattr(_real, k) for k in __all__ if hasattr(_real, k)})


def __getattr__(key):
    return getattr(_real, key)
