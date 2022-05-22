import ctypes
import os
import sys


def load_lib(path, dll_name, *argv):
    dll = os.path.join(
        path,
        (argv[0] if argv else dll_name) + ".dll")
    dll = ctypes.CDLL(dll)
    return dll
