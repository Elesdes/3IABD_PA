import ctypes
import os
from PIL import Image
import numpy as np


def resizer_and_gray(directory_source_name, directory_target_name,is_gray):
    newsize = (32, 32)
    for filename in os.listdir(directory_source_name):
        f = os.path.join(directory_source_name, filename)
        if os.path.isfile(f):
            img = Image.open(f)
            if is_gray:
                img = img.convert('L')
            img = img.resize(newsize)
            img = img.save(directory_target_name + filename)
            img_array = np.asarray(img)


def doing_resizer_and_gray():
    resizer_and_gray('img/Tour Eiffel/Training/', "img/Tour Eiffel/Resized_Training/", 0)
    resizer_and_gray('img/Tour Eiffel/Testing/', "img/Tour Eiffel/Resized_Testing/", 0)
    resizer_and_gray('img/Tour Eiffel/Training/', "img/Tour Eiffel/Resized_Gray_Training/", 1)
    resizer_and_gray('img/Tour Eiffel/Testing/', "img/Tour Eiffel/Resized_Gray_Testing/", 1)

    resizer_and_gray('img/Other/Training/', "img/Other/Resized_Training/", 0)
    resizer_and_gray('img/Other/Testing/', "img/Other/Resized_Testing/", 0)
    resizer_and_gray('img/Other/Training/', "img/Other/Resized_Gray_Training/", 1)
    resizer_and_gray('img/Other/Testing/', "img/Other/Resized_Gray_Testing/", 1)


def fill_x_and_y(x_dir, y_dir):
    x = []
    y = []
    for filename in os.listdir(x_dir):
        f = os.path.join(x_dir, filename)
        if os.path.isfile(f):
            img = Image.open(f)
            img = np.array(img)
            img = np.ravel(img)
            x.append(img)
            y.append(1)
    for filename in os.listdir(y_dir):
        f = os.path.join(y_dir, filename)
        if os.path.isfile(f):
            img = Image.open(f)
            img = np.array(img)
            img = np.ravel(img)
            x.append(img)
            y.append(-1)
    x = np.array(x)
    return x, y


def prepare_dll(my_dll, npl, is_not_loaded):
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

    my_dll.predictMLP.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
    my_dll.predictMLP.restype = ctypes.c_double

    my_dll.trainMLP.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_float, ctypes.c_int32,ctypes.c_int32]
    my_dll.trainMLP.restype = None

    my_dll.saveModelMLP.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32]
    my_dll.saveModelMLP.restype = None

    my_dll.useLoadedMLP.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_char_p, ctypes.c_int32]
    my_dll.useLoadedMLP.restype = ctypes.c_double

    my_dll.returnLenModel.argtypes = [ctypes.c_void_p]
    my_dll.returnLenModel.restype = ctypes.c_int32

    return my_dll


def set_var(x, y):
    rowsXLen = len(x)
    colsXLen = len(x[0])
    rowsYLen = len(y)
    arr_type_y = ctypes.c_int32 * len(y)
    return rowsXLen, colsXLen, rowsYLen, arr_type_y


def training(my_dll):
    #doing_resizer_and_gray()
    x, y = fill_x_and_y("img/Tour Eiffel/Resized_Gray_Training/", "img/Other/Resized_Gray_Training/")
    final_result = 0
    iter = 10000
    learning_rate = 0.01
    if isinstance(x[0][0],int):
        C_TYPE = ctypes.c_int32
    else:
        C_TYPE = ctypes.c_long
    POINTER_C_TYPE = ctypes.POINTER(C_TYPE)
    POINTER_POINTER_C_TYPE = ctypes.POINTER(POINTER_C_TYPE)
    ITLARR = C_TYPE * len(x[0])
    PITLARR = POINTER_C_TYPE * len(x)
    ptr_x = PITLARR()
    ptr_y = ITLARR()
    for i in range(len(x)):
        ptr_x[i] = ITLARR()
        for j in range(len(x[0])):
            ptr_x[i][j] = x[i][j]
    for i in range(len(y)):
        ptr_y[i] = y[i]

    rowsXLen, colsXLen, rowsYLen, arr_type_y = set_var(x, y)
    # A CHANGER
    npl = [2, 5, 2, 1]
    len_npl = len(npl)
    arr_type_npl = ctypes.c_int32 * len(npl)
    my_dll = prepare_dll(my_dll, npl, 1)

    print("===TRAINING===")
    MLP = my_dll.initiateMLP(arr_type_npl(*npl), len_npl)
    my_dll.trainMLP(MLP, ptr_x, rowsXLen, colsXLen, ptr_y, rowsYLen, learning_rate, 1, iter)
    for i in range(rowsXLen):
        print(my_dll.predictMLP(MLP, ptr_x[i], 1))
    my_dll.saveModelMLP(MLP, b"./save/save.txt", len_npl)
    """
    my_dll.destroyDoubleArray2D()
    my_dll.destroyDoubleArray2D()
    my_dll.destroyDoubleArray3D()
    """
    my_dll.destroyMlpModel(MLP)


def loaded(my_dll):
    x, y = fill_x_and_y("img/Tour Eiffel/Resized_Gray_Testing/", "img/Other/Resized_Gray_Testing/")
    if isinstance(x[0][0], int):
        C_TYPE = ctypes.c_int32
    else:
        C_TYPE = ctypes.c_long
    POINTER_C_TYPE = ctypes.POINTER(C_TYPE)
    POINTER_POINTER_C_TYPE = ctypes.POINTER(POINTER_C_TYPE)
    ITLARR = C_TYPE * len(x[0])
    PITLARR = POINTER_C_TYPE * len(x)
    ptr_x = PITLARR()
    ptr_y = ITLARR()
    for i in range(len(x)):
        ptr_x[i] = ITLARR()
        for j in range(len(x[0])):
            ptr_x[i][j] = x[i][j]
    for i in range(len(y)):
        ptr_y[i] = y[i]

    rowsXLen, colsXLen, rowsYLen, arr_type_y = set_var(x, y)
    # A CHANGER
    my_dll.loadModelMLP.argtypes = [ctypes.c_char_p]
    my_dll.loadModelMLP.restype = ctypes.c_void_p
    my_dll.returnModel.argtypes = [ctypes.c_void_p]
    my_dll.returnModel.restype = ctypes.POINTER(ctypes.c_int32)

    MLP = my_dll.loadModelMLP(b"./save/save.txt")
    npl = my_dll.returnModel(MLP)
    my_dll = prepare_dll(my_dll, npl, 0)
    arr_type_npl = ctypes.c_int32 * my_dll.returnLenModel(MLP)

    my_dll.destroyDoubleArray3D.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
                                            arr_type_npl, ctypes.c_int32]
    my_dll.destroyDoubleArray3D.restype = None
    my_dll.initiateMLP.argtypes = [arr_type_npl, ctypes.c_int32]
    my_dll.initiateMLP.restype = ctypes.c_void_p

    print("===TESTING===")
    for i in range(rowsXLen):
        print(my_dll.predictMLP(MLP, ptr_x[i], 1))
    my_dll.destroyMlpModel(MLP)


if __name__ == '__main__':
    my_cpp_dll = ctypes.CDLL(
        "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/MLPLib/cmake-build-debug/MLPLib.dll")

    print("In Cpp :")
    # training(my_cpp_dll)
    loaded(my_cpp_dll)
