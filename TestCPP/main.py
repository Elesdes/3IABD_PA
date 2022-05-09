import ctypes

import numpy as np


def test(my_dll):
    iter = 10000
    x = [[1, 0], [0, 1], [0, 0], [1, 1]]
    y = [1, 1, -1, -1]
    if isinstance(x[0][0],int):
        C_TYPE = ctypes.c_int32
        POINTER_C_TYPE = ctypes.POINTER(C_TYPE)
        POINTER_POINTER_C_TYPE = ctypes.POINTER(POINTER_C_TYPE)
    else:
        C_TYPE = ctypes.c_float
        POINTER_C_TYPE = ctypes.POINTER(C_TYPE)
        POINTER_POINTER_C_TYPE = ctypes.POINTER(POINTER_C_TYPE)
    ITLARR = C_TYPE * len(x[0])
    PITLARR = POINTER_C_TYPE * len(x)
    ptr = PITLARR()
    for i in range(len(x)):
        ptr[i] = ITLARR()
        for j in range(len(x[0])):
            ptr[i][j] = x[i][j]

    rowsXLen = len(x)
    colsXLen = len(x[0])
    rowsYLen = len(y)
    rowsWLen = colsXLen + 1
    arr_type_y = ctypes.c_int32 * len(y)

    my_dll.initModelWeights.argtypes = [ctypes.c_int32, ctypes.c_int32]
    my_dll.initModelWeights.restype = ctypes.POINTER(ctypes.c_float)

    my_dll.destroyFloatArray.argtypes = [ctypes.POINTER(ctypes.c_float)]
    my_dll.destroyFloatArray.restype = None

    my_dll.printFloatArray.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
    my_dll.printFloatArray.restype = ctypes.c_float

    my_dll.trainLinearInt.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), arr_type_y, ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    my_dll.trainLinearInt.restype = ctypes.POINTER(ctypes.c_float)

    my_dll.trainLinearFloat.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), arr_type_y, ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    my_dll.trainLinearFloat.restype = ctypes.POINTER(ctypes.c_float)

    my_dll.predictLinearModelClassificationInt.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
    my_dll.predictLinearModelClassificationInt.restype = ctypes.c_int32

    my_dll.predictLinearModelClassificationFloat.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
    my_dll.predictLinearModelClassificationFloat.restype = ctypes.c_int32

    if isinstance(x[0][0],int):
        w = my_dll.initModelWeights(colsXLen, rowsWLen)
        w = my_dll.trainLinearInt(ptr, arr_type_y(*y), w, rowsXLen, rowsWLen, iter)
        for i in range(rowsXLen):
            print(my_dll.predictLinearModelClassificationInt(w, ptr[i], rowsWLen))
        my_dll.destroyFloatArray(w)
    else:
        w = my_dll.initModelWeights(colsXLen, rowsWLen)
        w = my_dll.trainLinearFloat(ptr, arr_type_y(*y), w, rowsXLen, rowsWLen, iter)
        for i in range(rowsXLen):
            print(my_dll.predictLinearModelClassificationFloat(w, ptr[i], rowsWLen), " : result", y[i])
        my_dll.destroyFloatArray(w)



if __name__ == '__main__':
    my_cpp_dll = ctypes.CDLL(
        "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/LinearLib/cmake-build-debug/Library.dll")

    print("In Cpp :")
    test(my_cpp_dll)

