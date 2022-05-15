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


def prepare_dll(my_dll, arr_type_y):
    my_dll.initModelWeights.argtypes = [ctypes.c_int32, ctypes.c_int32]
    my_dll.initModelWeights.restype = ctypes.POINTER(ctypes.c_float)

    my_dll.printFloatArray.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
    my_dll.printFloatArray.restype = ctypes.c_float

    my_dll.destroyFloatArray.argtypes = [ctypes.POINTER(ctypes.c_float)]
    my_dll.destroyFloatArray.restype = None

    my_dll.trainLinearInt.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), arr_type_y,
                                      ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    my_dll.trainLinearInt.restype = ctypes.POINTER(ctypes.c_float)

    my_dll.trainLinearFloat.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), arr_type_y,
                                        ctypes.POINTER(ctypes.c_float), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    my_dll.trainLinearFloat.restype = ctypes.POINTER(ctypes.c_float)

    my_dll.predictLinearModelClassificationInt.argtypes = [ctypes.POINTER(ctypes.c_float),
                                                           ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
    my_dll.predictLinearModelClassificationInt.restype = ctypes.c_int32

    my_dll.predictLinearModelClassificationFloat.argtypes = [ctypes.POINTER(ctypes.c_float),
                                                             ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
    my_dll.predictLinearModelClassificationFloat.restype = ctypes.c_int32

    my_dll.saveModelLinear.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_char_p, ctypes.c_int32, ctypes.c_double]
    my_dll.saveModelLinear.restype = None

    my_dll.loadModelLinear.argtypes = [ctypes.c_char_p]
    my_dll.loadModelLinear.restype = ctypes.POINTER(ctypes.c_float)

    return my_dll


def set_var(x, y):
    rowsXLen = len(x)
    colsXLen = len(x[0])
    rowsWLen = colsXLen + 1
    arr_type_y = ctypes.c_int32 * len(y)
    return rowsXLen, colsXLen, rowsWLen, arr_type_y


def training(my_dll, max_pourcentage):
    #doing_resizer_and_gray()
    x, y = fill_x_and_y("img/Tour Eiffel/Resized_Gray_Training/", "img/Other/Resized_Gray_Training/")
    final_result = 0
    iter = 10000
    if isinstance(x[0][0],int):
        C_TYPE = ctypes.c_int32
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

    rowsXLen, colsXLen, rowsWLen, arr_type_y = set_var(x, y)
    my_dll = prepare_dll(my_dll, arr_type_y)

    print("===TRAINING===")
    w = my_dll.initModelWeights(colsXLen, rowsWLen)
    if isinstance(x[0][0], int):
        w = my_dll.trainLinearInt(ptr, arr_type_y(*y), w, rowsXLen, rowsWLen, iter)
        for i in range(rowsXLen):
            result = my_dll.predictLinearModelClassificationInt(w, ptr[i], rowsWLen)
            if(result == y[i]):
                final_result += 1
            print("Data: ", result, " | Result: ", y[i])
    else:
        w = my_dll.trainLinearFloat(ptr, arr_type_y(*y), w, rowsXLen, rowsWLen, iter)
        for i in range(rowsXLen):
            result = my_dll.predictLinearModelClassificationFloat(w, ptr[i], rowsWLen)
            if (result == y[i]):
                final_result += 1
            print("Data: ", result, " | Result: ", y[i])
    print("Result : ", final_result, "/", len(y), " | ", final_result / len(y) * 100, "%")
    x, y = fill_x_and_y("img/Tour Eiffel/Resized_Gray_Testing/", "img/Other/Resized_Gray_Testing/")
    for i in range(len(x)):
        ptr[i] = ITLARR()
        for j in range(len(x[0])):
            ptr[i][j] = x[i][j]

    rowsXLen, colsXLen, rowsWLen, arr_type_y = set_var(x, y)
    my_dll = prepare_dll(my_dll, arr_type_y)

    print("===TESTING===")
    final_result = 0
    if isinstance(x[0][0], int):
        for i in range(rowsXLen):
            result = my_dll.predictLinearModelClassificationFloat(w, ptr[i], rowsWLen)
            if (result == y[i]):
                final_result += 1
            print("Data: ", result, " | Result: ", y[i])
    else:
        for i in range(rowsXLen):
            result = my_dll.predictLinearModelClassificationFloat(w, ptr[i], rowsWLen)
            if (result == y[i]):
                final_result += 1
            print("Data: ", result, " | Result: ", y[i])
    print("Result : ", final_result, "/", len(y), " | ", final_result/len(y)*100, "%")
    pourcentage = final_result / len(y) * 100
    if pourcentage > max_pourcentage:
        str_file = "./save/linear/best.txt"
        str_file = str_file.encode('UTF-8')
        my_dll.saveModelLinear(w, str_file, rowsWLen, pourcentage)
        my_dll.destroyFloatArray(w)
        return pourcentage
    my_dll.destroyFloatArray(w)
    return max_pourcentage


def loaded(my_dll):
    x, y = fill_x_and_y("img/Tour Eiffel/Resized_Gray_Testing/", "img/Other/Resized_Gray_Testing/")
    if isinstance(x[0][0],int):
        C_TYPE = ctypes.c_int32
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
    for i in range(len(x)):
        ptr[i] = ITLARR()
        for j in range(len(x[0])):
            ptr[i][j] = x[i][j]

    rowsXLen, colsXLen, rowsWLen, arr_type_y = set_var(x, y)
    my_dll = prepare_dll(my_dll, arr_type_y)
    w = my_dll.loadModelLinear(b"./save/linear/best.txt")

    print("===TESTING===")
    final_result = 0
    if isinstance(x[0][0], int):
        for i in range(rowsXLen):
            result = my_dll.predictLinearModelClassificationFloat(w, ptr[i], rowsWLen)
            if (result == y[i]):
                final_result += 1
            print("Data: ", result, " | Result: ", y[i])
    else:
        for i in range(rowsXLen):
            result = my_dll.predictLinearModelClassificationFloat(w, ptr[i], rowsWLen)
            if (result == y[i]):
                final_result += 1
            print("Data: ", result, " | Result: ", y[i])
    print("Result : ", final_result, "/", len(y), " | ", final_result / len(y) * 100, "%")
    my_dll.destroyFloatArray(w)

"""
if __name__ == '__main__':
    max_pourcentage = 0
    my_cpp_dll = ctypes.CDLL(
        "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/LinearLib/cmake-build-debug/Library.dll")

    print("In Cpp :")
    #max_pourcentage = training(my_cpp_dll, max_pourcentage)
    loaded(my_cpp_dll)"""
