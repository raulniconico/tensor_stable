from .nn import *
from .tensor import *
from .btree import *
from .util import *


__version__ = "0.0.1"


# ############## phantom functions ##############
# def set_core():
#     raise ImportError
#
#
# ############## Nuwa core management ##############
# class Nuwa(types.ModuleType):
#     _implemented_func = [""]
#
#     _permitted_core = ["numpy_core"]
#     _default_core = "numpy_core"
#     _loaded_cores = dict()
#     _core = None
#     _THREAD_LOCAL_DATA = threading.local()
#     _ENV_DEFAULT_VAR = "NUWA_CORE"
#
#     @classmethod
#     def use_static_dispatch(cls):
#         """Switch to static dispatching: backend functions no longer will be dynamically dispatched.
#
#         This means that if you import a function after calling this function
#         and change the backend, those imported functions will still point to the old backend
#         """
#         # Define class methods and attributes that dynamically dispatch to the backend
#         for name in cls._functions:
#             setattr(cls, name, staticmethod(getattr(cls.current_backend(), name)))
#         for name in cls._attributes:
#             setattr(cls, name, getattr(cls.current_backend(), name))
#
#     @classmethod
#     def load_core(cls, core_name):
#         assert core_name in cls._permitted_core, \
#             f"{core_name!r} not permitted, permitted cores: {cls.available_core_names}"
#
#         if core_name not in Core._available_cores:
#             importlib.import_module(f"nuwa.core.{core_name}")
#
#         core = Core._available_cores[core_name]()
#         cls._loaded_cores[core_name] = core
#         return core
#
#     @classmethod
#     def set_core(cls, core, local_threadsafe=False):
#
#         if not isinstance(core, Core):
#             # Backend is a string
#             if core not in cls._loaded_cores:
#                 core = cls.load_core(core)
#             else:
#                 core = cls._loaded_cores[core]
#
#         cls._THREAD_LOCAL_DATA.core = core
#         if not local_threadsafe:
#             cls._default_core = core.core_name
#             cls._core = core
#
#     @classmethod
#     def initialize_core(cls):
#
#         core_name = os.environ.get(cls._ENV_DEFAULT_VAR, cls._default_core)
#         assert core_name in cls._permitted_core, f"core_name must in {cls._permitted_core}"
#         cls._default_core = core_name
#         cls.set_core(core_name)
#
#
# Nuwa.initialize_core()
# # Nuwa.use_static_dispatch()
#
# sys.modules[__name__].__class__ = Nuwa

