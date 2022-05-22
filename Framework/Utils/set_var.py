import ctypes


def set_var(x, y):
    rowsXLen = len(x)
    colsXLen = len(x[0])
    rowsWLen = colsXLen + 1
    if isinstance(y[0], int):
        arr_type_y = ctypes.c_int32 * len(y)
    else:
        arr_type_y = ctypes.c_float * len(y)
    return rowsXLen, colsXLen, rowsWLen, arr_type_y