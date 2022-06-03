import ctypes
import os
from typing import Any


def load_lib(path: str, dll_name: str, *argv: Any) -> ctypes:
    dll = os.path.join(
        path,
        (argv[0] if argv else dll_name) + ".dll")
    dll = ctypes.CDLL(dll)
    return dll
