import sys

import numpy
import numpy as np
import ctypes as ct
import os


def set_var_svm(x, y):
    colsXLen = len(x)
    rowsXLen = len(x[0])
    rowsWLen = rowsXLen + 1
    rowsYLen = len(y)
    if y.ndim < 2:
        colsYLen = 1
    else:
        colsYLen = len(y[0])
    return rowsXLen, colsXLen, rowsWLen, rowsYLen, colsYLen


def prepare_dll_SVM(my_dll):
    #my_dll.run.argtypes = [ct.POINTER(ct.POINTER(ct.c_double)), ct.POINTER(ct.c_int32), ct.c_int32, ct.c_int32, ct.c_int32]
    #my_dll.run.restype = ct.c_int32
    my_dll.freeArr.argtypes = [ct.POINTER(ct.c_double)]
    my_dll.freeArr.restype = None

    my_dll.initSVMWeight.argtypes = [ct.c_int32]
    my_dll.initSVMWeight.restype = ct.POINTER(ct.c_double)

    my_dll.initSVMWeightDerive.argtypes = [ct.c_int32]
    my_dll.initSVMWeightDerive.restype = ct.POINTER(ct.c_double)

    my_dll.getHingeLoss.argtypes = [ct.POINTER(ct.c_double), ct.c_int32, ct.POINTER(ct.c_double), ct.c_int32, ct.c_int32]
    my_dll.getHingeLoss.restype = ct.c_double

    my_dll.getSVMCost.argtypes = [ct.POINTER(ct.POINTER(ct.c_double)), ct.POINTER(ct.c_int32), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32]
    my_dll.getSVMCost.restype = ct.c_double

    my_dll.resultSVM.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_int32]
    my_dll.resultSVM.restype = ct.c_int32

    my_dll.trainSVM.argtypes = [ct.POINTER(ct.POINTER(ct.c_double)), ct.POINTER(ct.c_int32), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_double, ct.c_double]
    my_dll.trainSVM.restype = ct.POINTER(ct.c_double)

    my_dll.saveSVM.argtypes = [ct.POINTER(ct.c_double), ct.c_char_p, ct.c_int32, ct.c_double]
    my_dll.saveSVM.restype = None

    my_dll.loadSVM.argtypes = [ct.c_char_p]
    my_dll.loadSVM.restype = ct.POINTER(ct.c_double)

    return my_dll

def load_lib_svm(*argv):
    #dll_name = os.path.join("C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/Projet/Framework/LinearModel/cmake-build-debug", (argv[0] if argv else DLL_BASE_NAME) + ".dll")

    return "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/SVMLib/cmake-build-debug/SVMlib.dll"


def test():
    print("Python {0:s} {1:03d}bit on {2:s}\n".format(" ".join(item.strip() for item in sys.version.split("\n")),
                                                      64 if sys.maxsize > 0x100000000 else 32, sys.platform))
    dll = load_lib_svm(*sys.argv[1:])
    dll_test = ct.CDLL(dll)
    func_test = dll_test.test
    func_test.restype = ct.c_int

    res = func_test()
    print("{0:s} returned {1:d}".format(func_test.__name__, res))
    print("\nSVM lib successfully loaded")


if __name__ == '__main__':
    print("In SVM :")
    my_dll = load_lib_svm(*sys.argv[1:])
    my_dll = ct.CDLL(my_dll)
    test()
    file_name_given = "SVMSimple"

    my_dll = prepare_dll_SVM(my_dll)

    X = np.array([
        [1.0, 1.0],
        [2.0, 3.0],
        [3.0, 3.0]
    ])
    Y = np.array([
        1,
        -1,
        -1
    ])
    l_rate = 0.0005
    threshold = 0.001

    if isinstance(X[0][0], int) or isinstance(X[0][0], numpy.int32):
        C_TYPE_X = ct.c_int32
    else:
        C_TYPE_X = ct.c_double
    POINTER_C_TYPE_X = ct.POINTER(C_TYPE_X)
    ITLARR_X = C_TYPE_X * len(X[0])
    PITLARR_X = POINTER_C_TYPE_X * len(X)
    ptr_x = PITLARR_X()
    for i in range(len(X)):
        ptr_x[i] = ITLARR_X()
        for j in range(len(X[0])):
            ptr_x[i][j] = X[i][j]

    if isinstance(Y[0], int) or isinstance(Y[0], numpy.int32):
        C_TYPE_Y = ct.c_int32
    else:
        C_TYPE_Y = ct.c_float
    POINTER_C_TYPE_Y = ct.POINTER(C_TYPE_Y)
    ITLARR_Y = C_TYPE_Y * len(Y)
    ptr_y = ITLARR_Y()
    for i in range(len(Y)):
        ptr_y[i] = Y[i]


    #my_dll.run(ptr_x, ptr_y, len(X), len(X[0]), len(Y))
    rowsXLen, colsXLen, rowsWLen, rowsYLen, colsYLen = set_var_svm(X, Y)
    print("===TRAINING===")

    w = my_dll.initSVMWeight(rowsWLen)
    derive = my_dll.initSVMWeightDerive(rowsWLen)
    w = my_dll.trainSVM(ptr_x, ptr_y, w, derive, colsXLen, rowsXLen, rowsYLen, rowsWLen, l_rate, threshold)
    verify = 0
    result = 0
    for i in range(colsXLen):
        result = my_dll.resultSVM(ptr_x[i], w, rowsXLen)
        if result == Y[i]:
            verify += 1
    print("Result : ", verify, "/", len(Y), " | ", verify / len(Y) * 100, "%")
    str_file = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/TestLib/save/svm/" + file_name_given + ".txt"
    str_file = str_file.encode('UTF-8')
    my_dll.saveSVM(w, str_file, rowsWLen, verify / len(Y) * 100)
    my_dll.freeArr(w)
    my_dll.freeArr(derive)

    w = my_dll.loadSVM(str_file)
    str_file = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/TestLib/save/svm/" + file_name_given + "LOADED.txt"
    str_file = str_file.encode('UTF-8')
    my_dll.saveSVM(w, str_file, rowsWLen, verify / len(Y) * 100)
    my_dll.freeArr(w)