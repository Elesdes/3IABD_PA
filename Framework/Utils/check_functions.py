import ctypes
from typing import List, Tuple, Any


def set_var(x: List[int], y: List[int]) -> Tuple[int, int, int, Any]:
    rowsXLen = len(x)
    colsXLen = len(x[0])
    rowsWLen = colsXLen + 1
    if isinstance(y[0], int):
        arr_type_y = ctypes.c_int32 * len(y)
    else:
        arr_type_y = ctypes.c_float * len(y)
    return rowsXLen, colsXLen, rowsWLen, arr_type_y