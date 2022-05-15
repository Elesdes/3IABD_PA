import ctypes
import os
from PIL import Image
import numpy as np
from tqdm import tqdm


def to_binary(a):
    l, m=[], []
    for i in a:
        l.append(ord(i))
    for i in l:
        m.append(int(bin(i)[2:]))
    return m


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


def prepare_dll(my_dll, npl, is_not_loaded, is_int):
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

    if is_int:
        my_dll.predictMLPInt.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
        my_dll.predictMLPInt.restype = ctypes.c_double
        my_dll.trainMLPInt.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32,
                                    ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_float,
                                    ctypes.c_int32, ctypes.c_int32]
        my_dll.trainMLPInt.restype = None
    else:
        my_dll.predictMLPFloat.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
        my_dll.predictMLPFloat.restype = ctypes.c_double
        my_dll.trainMLPFloat.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int32,
                                    ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_float,
                                    ctypes.c_int32, ctypes.c_int32]
        my_dll.trainMLPFloat.restype = None


    my_dll.saveModelMLP.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_double]
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


def training(my_dll, max_pourcentage):
    #doing_resizer_and_gray()
    final_result = 0
    pourcentage = 0
    # x, y = fill_x_and_y("img/Tour Eiffel/Resized_Gray_Training/", "img/Other/Resized_Gray_Training/")
    x = np.random.random((500, 2)) * 2.0 - 1.0
    y = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in x])
    final_result = 0
    iter = 1000000
    learning_rate = 0.002
    if isinstance(x[0][0], int):
        C_TYPE = ctypes.c_int32
    else:
        C_TYPE = ctypes.c_float
    POINTER_C_TYPE = ctypes.POINTER(C_TYPE)
    POINTER_POINTER_C_TYPE = ctypes.POINTER(POINTER_C_TYPE)
    ITLARR = C_TYPE * len(x[0])
    ITLARRY = ctypes.c_long * len(y)
    PITLARR = POINTER_C_TYPE * len(x)
    ptr_x = PITLARR()
    ptr_y = ITLARRY()

    for i in range(len(x)):
        ptr_x[i] = ITLARR()
        for j in range(len(x[0])):
            ptr_x[i][j] = x[i][j]
    for i in range(len(y)):
        ptr_y[i] = y[i]

    rowsXLen, colsXLen, rowsYLen, arr_type_y = set_var(x, y)
    # A CHANGER
    #npl = [2, 5, 2, 1]
    npl = [2, 4, 2, 1]
    len_npl = len(npl)
    arr_type_npl = ctypes.c_int32 * len(npl)
    if isinstance(x[0][0], int):
        my_dll = prepare_dll(my_dll, npl, 0, 1)
    else:
        my_dll = prepare_dll(my_dll, npl, 0, 0)

    #print("===TRAINING===")
    MLP = my_dll.initiateMLP(arr_type_npl(*npl), len_npl)
    if isinstance(x[0][0], int):
        my_dll.trainMLPInt(MLP, ptr_x, rowsXLen, colsXLen, ptr_y, rowsYLen, learning_rate, 1, iter)
        for i in range(rowsXLen):
            result = my_dll.predictMLPInt(MLP, ptr_x[i], 1)
            if (result > 0 and y[i] > 0) or (result < 0 and y[i] < 0):
                final_result += 1
            #print("Data : ", y[i], " | Result: ", result)
    else:
        my_dll.trainMLPFloat(MLP, ptr_x, rowsXLen, colsXLen, ptr_y, rowsYLen, learning_rate, 1, iter)
        for i in range(rowsXLen):
            result = my_dll.predictMLPFloat(MLP, ptr_x[i], 1)
            if (result > 0 and y[i] > 0) or (result < 0 and y[i] < 0):
                final_result += 1
            #print("Data : ", y[i], " | Result: ", result)
    #print("Result : ", final_result, "/", len(y), " | ", final_result / len(y) * 100, "%")
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


def loaded(my_dll):
    final_result = 0
    # x, y = fill_x_and_y("img/Tour Eiffel/Resized_Gray_Testing/", "img/Other/Resized_Gray_Testing/")
    x = np.random.random((500, 2)) * 2.0 - 1.0
    y = np.array([1 if abs(p[0]) <= 0.3 or abs(p[1]) <= 0.3 else -1 for p in x])
    if isinstance(x[0][0], int):
        C_TYPE = ctypes.c_int32
    else:
        C_TYPE = ctypes.c_float
    POINTER_C_TYPE = ctypes.POINTER(C_TYPE)
    POINTER_POINTER_C_TYPE = ctypes.POINTER(POINTER_C_TYPE)
    ITLARR = C_TYPE * len(x[0])
    ITLARRY = ctypes.c_long * len(y)
    PITLARR = POINTER_C_TYPE * len(x)
    ptr_x = PITLARR()
    ptr_y = ITLARRY()
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

    MLP = my_dll.loadModelMLP(b"./save/mlp/best.txt")
    npl = my_dll.returnModel(MLP)
    if isinstance(x[0][0], int):
        my_dll = prepare_dll(my_dll, npl, 0, 1)
    else:
        my_dll = prepare_dll(my_dll, npl, 0, 0)
    arr_type_npl = ctypes.c_int32 * my_dll.returnLenModel(MLP)

    my_dll.destroyDoubleArray3D.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
                                            arr_type_npl, ctypes.c_int32]
    my_dll.destroyDoubleArray3D.restype = None
    my_dll.initiateMLP.argtypes = [arr_type_npl, ctypes.c_int32]
    my_dll.initiateMLP.restype = ctypes.c_void_p

    print("===TESTING===")
    if isinstance(x[0][0], int):
        for i in range(rowsXLen):
            result = my_dll.predictMLPInt(MLP, ptr_x[i], 1)
            if (result > 0 and y[i] > 0) or (result < 0 and y[i] < 0):
                final_result += 1
            print("Data : ", y[i], " | Result: ", my_dll.predictMLPInt(MLP, ptr_x[i], 1))
    else:
        for i in range(rowsXLen):
            result = my_dll.predictMLPFloat(MLP, ptr_x[i], 1)
            if (result > 0 and y[i] > 0) or (result < 0 and y[i] < 0):
                final_result += 1
            print("Data : ", y[i], " | Result: ", my_dll.predictMLPFloat(MLP, ptr_x[i], 1))
    print("Result : ", final_result, "/", len(y), " | ", final_result / len(y) * 100, "%")
    my_dll.destroyMlpModel(MLP)


if __name__ == '__main__':
    max_pourcentage = 0
    my_cpp_dll = ctypes.CDLL(
        "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/MLPLib/cmake-build-debug/MLPLib.dll")

    print("In Cpp :")
    #for i in tqdm(range(0, 10)):
        #max_pourcentage = training(my_cpp_dll, max_pourcentage)
    loaded(my_cpp_dll)
