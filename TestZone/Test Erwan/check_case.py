import sys
import ctypes as ct
import os

import numpy
from PIL import Image
import numpy as np

DLL_BASE_NAME = "LinearModel"


import ctypes
import os
from PIL import Image
import numpy as np
from tqdm import tqdm


def prepare_dll_mlp(my_dll, npl, is_not_loaded):
    my_dll.destroyMlpModel.argtypes = [ctypes.c_void_p]
    my_dll.destroyMlpModel.restype = None

    if(is_not_loaded):
        arr_type_npl = ctypes.c_int32 * len(npl)
        my_dll.destroyDoubleArray3D.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))), arr_type_npl, ctypes.c_int32]
        my_dll.destroyDoubleArray3D.restype = None
        my_dll.initiateMLP.argtypes = [arr_type_npl, ctypes.c_int32]
        my_dll.initiateMLP.restype = ctypes.c_void_p

    my_dll.destroyDoubleArray2D.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int32]
    my_dll.destroyDoubleArray2D.restype = None

    my_dll.destroyIntArray2D.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
    my_dll.destroyIntArray2D.restype = None

    my_dll.destroyDoubleArray1D.argtypes = [ctypes.POINTER(ctypes.c_double)]
    my_dll.destroyDoubleArray1D.restype = None

    my_dll.predictMLPInt.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
    my_dll.predictMLPInt.restype = ctypes.c_double
    my_dll.trainMLPInt.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32,
                                   ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_float,
                                   ctypes.c_int32, ctypes.c_int32]
    my_dll.trainMLPInt.restype = None

    my_dll.predictMLPFloat.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
    my_dll.predictMLPFloat.restype = ctypes.c_double

    my_dll.predictMLPFloatMultipleOutputs.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32]
    my_dll.predictMLPFloatMultipleOutputs.restype = ctypes.POINTER(ctypes.c_double)

    my_dll.trainMLPFloat.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int32,
                                     ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_float,
                                     ctypes.c_int32, ctypes.c_int32]
    my_dll.trainMLPFloat.restype = None

    my_dll.trainMLPFloatMultipleOutputs.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                                                    ctypes.c_int32,
                                                    ctypes.c_int32, ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32, ctypes.c_int32, ctypes.c_float,
                                                    ctypes.c_int32, ctypes.c_int32]
    my_dll.trainMLPFloatMultipleOutputs.restype = None



    my_dll.saveModelMLP.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_double]
    my_dll.saveModelMLP.restype = None

    my_dll.useLoadedMLP.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_char_p, ctypes.c_int32]
    my_dll.useLoadedMLP.restype = ctypes.c_double

    my_dll.returnLenModel.argtypes = [ctypes.c_void_p]
    my_dll.returnLenModel.restype = ctypes.c_int32

    my_dll.readArray.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int32]
    my_dll.readArray.restype = ctypes.c_double

    my_dll.returnDoubleArray.argtypes = [ctypes.c_int32]
    my_dll.returnDoubleArray.restype = ctypes.POINTER(ctypes.c_double)

    return my_dll


def set_var_mlp(x, y, npl):
    rowsXLen = len(x)
    colsXLen = len(x[0])
    rowsYLen = len(y)
    if y.ndim < 2:
        colsYLen = 1
    else:
        colsYLen = len(y[0])
    arr_type_y = ctypes.c_int32 * len(y)
    len_npl = len(npl)
    return rowsXLen, colsXLen, rowsYLen, colsYLen, arr_type_y, len_npl


def training_mlp(my_dll, max_pourcentage, x, y, npl, is_classification):
    pourcentage = 0
    final_result = 0
    iter = 4000000
    learning_rate = 0.0001

    if isinstance(x[0][0], int) or isinstance(x[0][0], numpy.int32):
        C_TYPE = ctypes.c_int32
    else:
        C_TYPE = ctypes.c_float
    POINTER_C_TYPE = ctypes.POINTER(C_TYPE)
    POINTER_POINTER_C_TYPE = ctypes.POINTER(POINTER_C_TYPE)
    ITLARR = C_TYPE * len(x[0])
    PITLARR = POINTER_C_TYPE * len(x)
    ptr_x = PITLARR()

    for i in range(len(x)):
        ptr_x[i] = ITLARR()
        for j in range(len(x[0])):
            ptr_x[i][j] = x[i][j]

    #Set Y
    if y.ndim < 2:
        ITLARRY = ctypes.c_long * len(y)
        ptr_y = ITLARRY()
        for i in range(len(y)):
            ptr_y[i] = int(y[i])
    else:
        ITLARRY = ctypes.c_long * len(y[0])
        PITLARRY = ctypes.POINTER(ctypes.c_long) * len(y)
        ptr_y = PITLARRY()
        for i in range(len(y)):
            ptr_y[i] = ITLARRY()
            for j in range(len(y[0])):
                ptr_y[i][j] = y[i][j]

    rowsXLen, colsXLen, rowsYLen, colsYlen, arr_type_y, len_npl = set_var_mlp(x, y, npl)
    arr_type_npl = ctypes.c_int32 * len(npl)

    if isinstance(x[0][0], int) or isinstance(x[0][0], numpy.int32):
        my_dll = prepare_dll_mlp(my_dll, npl, 0)
    else:
        my_dll = prepare_dll_mlp(my_dll, npl, 0)

    #print("===TRAINING===")
    MLP = my_dll.initiateMLP(arr_type_npl(*npl), len_npl)
    if isinstance(x[0][0], int) or isinstance(x[0][0], numpy.int32):
        my_dll.trainMLPInt(MLP, ptr_x, rowsXLen, colsXLen, ptr_y, rowsYLen, learning_rate, is_classification, iter)
        for i in range(rowsXLen):
            result = my_dll.predictMLPInt(MLP, ptr_x[i], 1)
            if (result > 0 and y[i] > 0) or (result < 0 and y[i] < 0):
                final_result += 1
            #print("Data : ", y[i], " | Result: ", result)
    else:
        if y.ndim < 2:
            my_dll.trainMLPFloat(MLP, ptr_x, rowsXLen, colsXLen, ptr_y, rowsYLen, learning_rate, is_classification, iter)
        else:
            my_dll.trainMLPFloatMultipleOutputs(MLP, ptr_x, rowsXLen, colsXLen, ptr_y, rowsYLen, colsYlen, learning_rate, is_classification,
                                                iter)
        if y.ndim < 2:
            for i in range(rowsXLen):
                result = my_dll.predictMLPFloat(MLP, ptr_x[i], 1)
                if (result > 0 and y[i] > 0) or (result < 0 and y[i] < 0):
                    final_result += 1
        else:
            for i in range(rowsXLen):
                result = my_dll.predictMLPFloatMultipleOutputs(MLP, ptr_x[i], 1, len(y[0]))
                # TODO: A voir /!\ range(3)
                verify = 0
                res_list = list()
                for iter_test in range(3):
                    res_list.append(my_dll.readArray(result, iter_test))
                #print("y[",i,"] : ",y[i], " | Result : ",test_list)
                for case_num, res_y in enumerate(y[i]):
                    if (res_y > 0.5 and my_dll.readArray(result, case_num) > 0.5) or (res_y < 0.5 and my_dll.readArray(result, case_num) < 0.5):
                        verify = verify + 1
                #print(verify)
                #print(len(y[0]))
                if verify == len(y[0]):
                    final_result += 1

                my_dll.destroyDoubleArray1D(result)
    print("Result : ", final_result, "/", len(y), " | ", final_result / len(y) * 100, "%")
    pourcentage = final_result / len(y) * 100
    if pourcentage > max_pourcentage:
        str_file = "./save/mlp/best.txt"
        str_file = str_file.encode('UTF-8')
        my_dll.saveModelMLP(MLP, str_file, len_npl, pourcentage)
        my_dll.destroyMlpModel(MLP)
        return pourcentage
    """
    my_dll.destroyDoubleArray2D()
    my_dll.destroyDoubleArray2D()
    my_dll.destroyDoubleArray3D()
    """
    my_dll.destroyMlpModel(MLP)
    return max_pourcentage


def prepare_dll_linear(my_dll):
    my_dll.initModelWeights.argtypes = [ct.c_int32, ct.c_int32]
    my_dll.initModelWeights.restype = ct.POINTER(ct.c_float)

    my_dll.printFloatArray.argtypes = [ct.POINTER(ct.c_float), ct.c_int32]
    my_dll.printFloatArray.restype = ct.c_float

    my_dll.destroyFloatArray.argtypes = [ct.POINTER(ct.c_float)]
    my_dll.destroyFloatArray.restype = None

    my_dll.trainLinearInt.argtypes = [ct.POINTER(ct.POINTER(ct.c_int32)), ctypes.POINTER(ctypes.c_int32),
                                      ct.POINTER(ct.c_float), ct.c_int32, ct.c_int32, ct.c_int32, ct.c_float]
    my_dll.trainLinearInt.restype = ct.POINTER(ct.c_float)

    my_dll.trainLinearFloat.argtypes = [ct.POINTER(ct.POINTER(ct.c_float)), ctypes.POINTER(ctypes.c_int32),
                                        ct.POINTER(ct.c_float), ct.c_int32, ct.c_int32, ct.c_int32, ct.c_float]
    my_dll.trainLinearFloat.restype = ct.POINTER(ct.c_float)

    my_dll.predictLinearModelClassificationInt.argtypes = [ct.POINTER(ct.c_float),
                                                           ct.POINTER(ct.c_int32), ct.c_int32]
    my_dll.predictLinearModelClassificationInt.restype = ct.c_int32

    my_dll.predictLinearModelClassificationFloat.argtypes = [ct.POINTER(ct.c_float),
                                                             ct.POINTER(ct.c_float), ct.c_int32]
    my_dll.predictLinearModelClassificationFloat.restype = ct.c_int32

    my_dll.saveModelLinear.argtypes = [ct.POINTER(ct.c_float), ct.c_char_p, ct.c_int32]
    my_dll.saveModelLinear.restype = None

    my_dll.loadModelLinear.argtypes = [ct.c_char_p]
    my_dll.loadModelLinear.restype = ct.POINTER(ct.c_float)

    my_dll.printIntArray.argtypes = [ct.POINTER(ct.c_int32), ct.c_int32]
    my_dll.printIntArray.restype = None

    return my_dll


def set_var_linear(x):
    rowsXLen = len(x)
    colsXLen = len(x[0])
    rowsWLen = colsXLen + 1
    return rowsXLen, colsXLen, rowsWLen


def training_linear(my_dll, x, y):
    # doing_resizer_and_gray()
    final_result = 0
    iter = 1000000
    learning_rate = 0.001

    #Set X
    if isinstance(x[0][0], int) or isinstance(x[0][0], numpy.int32):
        C_TYPE = ct.c_int32
    else:
        C_TYPE = ct.c_float
    POINTER_C_TYPE = ct.POINTER(C_TYPE)
    POINTER_POINTER_C_TYPE = ct.POINTER(POINTER_C_TYPE)
    ITLARR = C_TYPE * len(x[0])
    PITLARR = POINTER_C_TYPE * len(x)
    ptr = PITLARR()
    for i in range(len(x)):
        ptr[i] = ITLARR()
        for j in range(len(x[0])):
            ptr[i][j] = x[i][j]

    #Set Y
    if isinstance(y[0], int) or isinstance(y[0], numpy.int32):
        arr_type_y = ct.c_int32
    else:
        arr_type_y = ct.c_float
    ITLARRY = arr_type_y * len(y)
    ptr_y = ITLARRY()
    for i in range(len(y)):
        ptr_y[i] = y[i]

    rowsXLen, colsXLen, rowsWLen = set_var_linear(x)
    my_dll = prepare_dll_linear(my_dll)

    print("===TRAINING===")
    w = my_dll.initModelWeights(colsXLen, rowsWLen)

    if isinstance(x[0][0], int) or isinstance(x[0][0], numpy.int32):
        w = my_dll.trainLinearInt(ptr, ptr_y, w, rowsXLen, rowsWLen, iter, learning_rate)
        for i in range(rowsXLen):
            result = my_dll.predictLinearModelClassificationInt(w, ptr[i], rowsWLen)
            if (result == y[i]):
                final_result += 1
            #print("Data: ", result, " | Result: ", y[i])
    else:
        w = my_dll.trainLinearFloat(ptr, ptr_y, w, rowsXLen, rowsWLen, iter, learning_rate)
        for i in range(rowsXLen):
            result = my_dll.predictLinearModelClassificationFloat(w, ptr[i], rowsWLen)
            if (result == y[i]):
                final_result += 1
            #print("Data: ", result, " | Result: ", y[i])
    print("Result : ", final_result, "/", len(y), " | ", final_result / len(y) * 100, "%")
    """
    for i in range(len(x)):
        ptr[i] = ITLARR()
        for j in range(len(x[0])):
            ptr[i][j] = x[i][j]

    rowsXLen, colsXLen, rowsWLen, arr_type_y = set_var_linear(x, y)
    my_dll = prepare_dll_linear(my_dll, arr_type_y)
    """
    print("===TESTING===")
    final_result = 0
    if isinstance(x[0][0], int) or isinstance(x[0][0], numpy.int32):
        for i in range(rowsXLen):
            result = my_dll.predictLinearModelClassificationInt(w, ptr[i], rowsWLen)
            if (result == y[i]):
                final_result += 1
            #print("Data: ", result, " | Result: ", y[i])
    else:
        for i in range(rowsXLen):
            result = my_dll.predictLinearModelClassificationFloat(w, ptr[i], rowsWLen)
            if (result == y[i]):
                final_result += 1
            #print("Data: ", result, " | Result: ", y[i])
    print("Result : ", final_result, "/", len(y), " | ", final_result / len(y) * 100, "%")
    my_dll.saveModelLinear(w, b"./save/test.txt", rowsWLen)
    my_dll.destroyFloatArray(w)


def load_lib_linear(*argv):
    #dll_name = os.path.join("C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/Projet/Framework/LinearModel/cmake-build-debug", (argv[0] if argv else DLL_BASE_NAME) + ".dll")

    return "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/LinearLib/cmake-build-debug/Library.dll"


def load_lib_mlp(*argv):
    #dll_name = os.path.join("C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/Projet/Framework/LinearModel/cmake-build-debug", (argv[0] if argv else DLL_BASE_NAME) + ".dll")

    return "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/MLPLib/cmake-build-debug/MLPLib.dll"


def test():
    print("Python {0:s} {1:03d}bit on {2:s}\n".format(" ".join(item.strip() for item in sys.version.split("\n")),
                                                      64 if sys.maxsize > 0x100000000 else 32, sys.platform))
    dll = load_lib_linear(*sys.argv[1:])
    dll_test = ctypes.CDLL(dll)
    func_test = dll_test.test
    func_test.restype = ct.c_int

    res = func_test()
    print("{0:s} returned {1:d}".format(func_test.__name__, res))
    print("\nLinearModel lib successfully loaded")


if __name__ == '__main__':
    print("In Cpp :")
    var = load_lib_linear(*sys.argv[1:])
    var = ctypes.CDLL(var)
    test()

    x = np.array([
        [1, 1],
        [2, 3],
        [3, 3]
    ])
    y = np.array([
        1,
        -1,
        -1
    ])
    print("Linear Simple - Linear model: ")
    training_linear(var, x, y)
    x = np.concatenate(
        [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])
    y = y.astype(int).flatten()
    print("Linear Multiple - Linear model: ")
    training_linear(var, x, y)
    x = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    y = np.array([1, 1, -1, -1])
    print("XOR - Linear model: ")
    training_linear(var, x, y)
    x = np.random.random((500, 2)) * 2.0 - 1.0
    y = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in x])
    print("Cross - Linear model: ")
    training_linear(var, x, y)
    x = np.random.random((500, 2)) * 2.0 - 1.0
    y = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
                  [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
                  [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
                  [0, 0, 0] for p in x])
    x = x[[not np.all(arr == [0, 0, 0]) for arr in y]]
    y = y[[not np.all(arr == [0, 0, 0]) for arr in y]]
    print("Multi Linear 3 classes - Linear model: ")
    print("FAILED")
    #training_linear(var, x, y)
    x = np.random.random((1000, 2)) * 2.0 - 1.0
    y = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [0, 1, 0] if abs(
        p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [0, 0, 1] for p in x])
    print("Multi Cross - Linear model: ")
    print("FAILED")
    #training_linear(var, x, y)
    x = np.array([
        [1],
        [2]
    ])
    y = np.array([
        2,
        3
    ])
    print("Régression - Linear model: ")
    training_linear(var, x, y)
    x = np.array([
        [1],
        [2],
        [3]
    ])
    y = np.array([
        2,
        3,
        2.5
    ])
    print("Non Linear Simple 2D - Linear model: ")
    print("FAILED")
    #training_linear(var, x, y)
    x = np.array([
        [1, 1],
        [2, 2],
        [3, 1]
    ])
    y = np.array([
        2,
        3,
        2.5
    ])
    print("Linear Simple 3D - Linear model: ")
    print("FAILED")
    #training_linear(var, x, y)
    x = np.array([
        [1, 1],
        [2, 2],
        [3, 3]
    ])
    y = np.array([
        1,
        2,
        3
    ])
    print("Linear Tricky 3D - Linear model: ")
    training_linear(var, x, y)
    x = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ])
    y = np.array([
        2,
        1,
        -2,
        -1
    ])
    print("Non Linear Simple 3D - Linear model: ")
    training_linear(var, x, y)

    var = load_lib_mlp(*sys.argv[1:])
    var = ctypes.CDLL(var)
    max_pourcentage = 0
    npl = [2, 2, 1]
    x = np.array([
        [4, 4],
        [2, 3],
        [3, 3]
    ])
    y = np.array([
        1,
        -1,
        -1
    ])
    print("Linear Simple - Mlp: ")
    training_mlp(var, max_pourcentage, x, y, npl, 1)
    x = np.concatenate(
        [np.random.random((50, 2)) * 0.9 + np.array([1, 1]), np.random.random((50, 2)) * 0.9 + np.array([2, 2])])
    y = np.concatenate([np.ones((50, 1)), np.ones((50, 1)) * -1.0])
    y = y.astype(int).flatten()
    print("Linear Multiple - Mlp: ")
    training_mlp(var, max_pourcentage, x, y, npl, 1)
    x = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    y = np.array([1, 1, -1, -1])
    y = y.astype(int)
    print("XOR - Mlp: ")
    training_mlp(var, max_pourcentage, x, y, npl, 1)
    x = np.random.random((500, 2)) * 2.0 - 1.0
    y = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in x])
    npl = [2, 4, 1]
    print("Cross - Mlp: ")
    training_mlp(var, max_pourcentage, x, y, npl, 1)
    x = np.random.random((500, 2)) * 2.0 - 1.0
    y = np.array([[1, 0, 0] if -p[0] - p[1] - 0.5 > 0 and p[1] < 0 and p[0] - p[1] - 0.5 < 0 else
                  [0, 1, 0] if -p[0] - p[1] - 0.5 < 0 and p[1] > 0 and p[0] - p[1] - 0.5 < 0 else
                  [0, 0, 1] if -p[0] - p[1] - 0.5 < 0 and p[1] < 0 and p[0] - p[1] - 0.5 > 0 else
                  [0, 0, 0] for p in x])
    x = x[[not np.all(arr == [0, 0, 0]) for arr in y]]
    y = y[[not np.all(arr == [0, 0, 0]) for arr in y]]
    npl = [2, 3]
    print("Multi Linear 3 classes - Mlp: ")
    #print("FAILED")
    training_mlp(var, max_pourcentage, x, y, npl, 1)
    x = np.random.random((1000, 2)) * 2.0 - 1.0
    y = np.array([[1, 0, 0] if abs(p[0] % 0.5) <= 0.25 and abs(p[1] % 0.5) > 0.25 else [0, 1, 0] if abs(
        p[0] % 0.5) > 0.25 and abs(p[1] % 0.5) <= 0.25 else [0, 0, 1] for p in x])
    npl = [2, 8, 8, 3]
    print("Multi Cross - Mlp: ")
    #print("FAILED")
    training_mlp(var, max_pourcentage, x, y, npl, 1)
    x = np.array([
        [1],
        [2]
    ])
    y = np.array([
        2,
        3
    ])
    print("Régression - Mlp: ")
    npl = [1, 1]
    training_mlp(var, max_pourcentage, x, y, npl, 0)
    x = np.array([
        [1],
        [2],
        [3]
    ])
    y = np.array([
        2,
        3,
        2.5
    ])
    print("Non Linear Simple 2D - Mlp: ")
    #print("FAILED")
    training_mlp(var, max_pourcentage, x, y, npl, 0)
    x = np.array([
        [1, 1],
        [2, 2],
        [3, 1]
    ])
    y = np.array([
        2,
        3,
        2.5
    ])
    print("Linear Simple 3D - Mlp: ")
    #print("FAILED")
    training_mlp(var, max_pourcentage, x, y, npl, 0)
    x = np.array([
        [1, 1],
        [2, 2],
        [3, 3]
    ])
    y = np.array([
        1,
        2,
        3
    ])
    print("Linear Tricky 3D - Mlp: ")
    training_mlp(var, max_pourcentage, x, y, npl, 0)
    x = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0],
    ])
    y = np.array([
        2,
        1,
        -2,
        -1
    ])
    npl = [2, 1]
    print("Non Linear Simple 3D - Mlp: ")
    training_mlp(var, max_pourcentage, x, y, npl, 0)
