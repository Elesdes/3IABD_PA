import sys
import ctypes as ct
import os
from PIL import Image
import numpy as np

DLL_BASE_NAME = "libLinearModel"


def resizer_and_gray(directory_source_name, directory_target_name, is_gray):
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
    my_dll.initModelWeights.argtypes = [ct.c_int32, ct.c_int32]
    my_dll.initModelWeights.restype = ct.POINTER(ct.c_float)

    my_dll.printFloatArray.argtypes = [ct.POINTER(ct.c_float), ct.c_int32]
    my_dll.printFloatArray.restype = ct.c_float

    my_dll.destroyFloatArray.argtypes = [ct.POINTER(ct.c_float)]
    my_dll.destroyFloatArray.restype = None

    my_dll.trainLinearInt.argtypes = [ct.POINTER(ct.POINTER(ct.c_int32)), arr_type_y,
                                      ct.POINTER(ct.c_float), ct.c_int32, ct.c_int32, ct.c_int32]
    my_dll.trainLinearInt.restype = ct.POINTER(ct.c_float)

    my_dll.trainLinearFloat.argtypes = [ct.POINTER(ct.POINTER(ct.c_float)), arr_type_y,
                                        ct.POINTER(ct.c_float), ct.c_int32, ct.c_int32, ct.c_int32]
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

    return my_dll


def set_var(x, y):
    rowsXLen = len(x)
    colsXLen = len(x[0])
    rowsWLen = colsXLen + 1
    arr_type_y = ct.c_int32 * len(y)
    return rowsXLen, colsXLen, rowsWLen, arr_type_y


def training(my_dll):
    # doing_resizer_and_gray()
    x, y = fill_x_and_y("img/Tour Eiffel/Resized_Gray_Training/", "img/Other/Resized_Gray_Training/")
    final_result = 0
    iter = 10000
    if isinstance(x[0][0], int):
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

    rowsXLen, colsXLen, rowsWLen, arr_type_y = set_var(x, y)
    my_dll = prepare_dll(my_dll, arr_type_y)

    print("===TRAINING===")
    w = my_dll.initModelWeights(colsXLen, rowsWLen)
    if isinstance(x[0][0], int):
        w = my_dll.trainLinearInt(ptr, arr_type_y(*y), w, rowsXLen, rowsWLen, iter)
        for i in range(rowsXLen):
            result = my_dll.predictLinearModelClassificationInt(w, ptr[i], rowsWLen)
            if (result == y[i]):
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
    print("Result : ", final_result, "/", len(y), " | ", final_result / len(y) * 100, "%")
    my_dll.saveModelLinear(w, b"./save/test.txt", rowsWLen)
    my_dll.destroyFloatArray(w)


def loaded(my_dll):
    x, y = fill_x_and_y("img/Tour Eiffel/Resized_Gray_Testing/", "img/Other/Resized_Gray_Testing/")
    if isinstance(x[0][0], int):
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
    for i in range(len(x)):
        ptr[i] = ITLARR()
        for j in range(len(x[0])):
            ptr[i][j] = x[i][j]

    rowsXLen, colsXLen, rowsWLen, arr_type_y = set_var(x, y)
    my_dll = prepare_dll(my_dll, arr_type_y)
    w = my_dll.loadModelLinear(b"./save/test.txt")

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


def load_lib(*argv):
    dll_name = os.path.join(
        "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/Framework/LinearModel/cmake-build-debug",
        (argv[0] if argv else DLL_BASE_NAME) + ".dll")
    return dll_name


def test():
    print("Python {0:s} {1:03d}bit on {2:s}\n".format(" ".join(item.strip() for item in sys.version.split("\n")),
                                                      64 if sys.maxsize > 0x100000000 else 32, sys.platform))
    dll = load_lib(*sys.argv[1:])
    dll_test = ct.CDLL(dll)
    func_test = dll_test.test
    func_test.restype = ct.c_int

    res = func_test()
    print("{0:s} returned {1:d}".format(func_test.__name__, res))
    print("\nLinearModel lib successfully loaded")


"""if __name__ == '__main__':
    print("In Cpp :")
    var = load_lib(*sys.argv[1:])
    # training(var)
    # loaded(var)"""
