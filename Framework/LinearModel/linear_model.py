import ctypes

from typing import List, Any

from Framework.Utils import image_opener as io
from Framework.Utils import set_var as sv

PATH_TE_TEST = "G:\Mon Drive\Projet Annuel 3IABD\Data Lake\Tour Eiffel\img_test"
PATH_TE_TRAIN = "G:\Mon Drive\Projet Annuel 3IABD\Data Lake\Tour Eiffel\img_train"
PATH_ARC_TEST = "G:\Mon Drive\Projet Annuel 3IABD\Data Lake\Arc de triomphe\img_test"
PATH_ARC_TRAIN = "G:\Mon Drive\Projet Annuel 3IABD\Data Lake\Arc de triomphe\img_train"


def prepare_dll(my_dll: ctypes.cdll, arr_type_y: List[Any]) -> ctypes.cdll:
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


def training(my_dll: ctypes.cdll, n_iter: int, max_pourcentage: float) -> float:
    # doing_resizer_and_gray()
    x, y = io.fill_x_and_y(PATH_TE_TRAIN, PATH_ARC_TRAIN)
    final_result = 0
    # iter = 100000
    if isinstance(x[0][0], int):
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

    rowsXLen, colsXLen, rowsWLen, arr_type_y = sv.set_var(x, y)
    my_dll = prepare_dll(my_dll, arr_type_y)

    # print("===TRAINING===")
    w = my_dll.initModelWeights(colsXLen, rowsWLen)
    if isinstance(x[0][0], int):
        w = my_dll.trainLinearInt(ptr, arr_type_y(*y), w, rowsXLen, rowsWLen, n_iter)
        for i in range(rowsXLen):
            result = my_dll.predictLinearModelClassificationInt(w, ptr[i], rowsWLen)
            if (result == y[i]):
                final_result += 1
            # print("Data: ", result, " | Result: ", y[i])
    else:
        w = my_dll.trainLinearFloat(ptr, arr_type_y(*y), w, rowsXLen, rowsWLen, n_iter)
        for i in range(rowsXLen):
            result = my_dll.predictLinearModelClassificationFloat(w, ptr[i], rowsWLen)
            if (result == y[i]):
                final_result += 1
            # print("Data: ", result, " | Result: ", y[i])
    print("Result : ", final_result, "/", len(y), " | ", final_result / len(y) * 100, "%")
    x, y = io.fill_x_and_y(PATH_TE_TEST, PATH_ARC_TEST)
    for i in range(len(x)):
        ptr[i] = ITLARR()
        for j in range(len(x[0])):
            ptr[i][j] = x[i][j]

    rowsXLen, colsXLen, rowsWLen, arr_type_y = sv.set_var(x, y)
    my_dll = prepare_dll(my_dll, arr_type_y)

    # print("===TESTING===")
    final_result = 0
    if isinstance(x[0][0], int):
        for i in range(rowsXLen):
            result = my_dll.predictLinearModelClassificationInt(w, ptr[i], rowsWLen)
            if (result == y[i]):
                final_result += 1
            # print("Data: ", result, " | Result: ", y[i])
    else:
        for i in range(rowsXLen):
            result = my_dll.predictLinearModelClassificationFloat(w, ptr[i], rowsWLen)
            if (result == y[i]):
                final_result += 1
            # print("Data: ", result, " | Result: ", y[i])
    print("Result : ", final_result, "/", len(y), " | ", final_result / len(y) * 100, "%")
    pourcentage = final_result / len(y) * 100
    if pourcentage > max_pourcentage:
        str_file = "./save/best.txt"
        str_file = str_file.encode('UTF-8')
        my_dll.saveModelLinear(w, str_file, rowsWLen, pourcentage)
        my_dll.destroyFloatArray(w)
        return pourcentage
    my_dll.destroyFloatArray(w)
    return max_pourcentage


def loaded(my_dll: ctypes.cdll) -> None:
    x, y = io.fill_x_and_y(PATH_TE_TEST, PATH_ARC_TEST)
    if isinstance(x[0][0], int):
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

    rowsXLen, colsXLen, rowsWLen, arr_type_y = sv.set_var(x, y)
    my_dll = prepare_dll(my_dll, arr_type_y)
    w = my_dll.loadModelLinear(b"./save/best.txt")

    # print("===TESTING===")
    final_result = 0
    if isinstance(x[0][0], int):
        for i in range(rowsXLen):
            result = my_dll.predictLinearModelClassificationFloat(w, ptr[i], rowsWLen)
            if (result == y[i]):
                final_result += 1
            # print("Data: ", result, " | Result: ", y[i])
    else:
        for i in range(rowsXLen):
            result = my_dll.predictLinearModelClassificationFloat(w, ptr[i], rowsWLen)
            if (result == y[i]):
                final_result += 1
            # print("Data: ", result, " | Result: ", y[i])
    print("Result : ", final_result, "/", len(y), " | ", final_result / len(y) * 100, "%")
    my_dll.destroyFloatArray(w)


"""if __name__ == '__main__':
    # var = load_lib(*sys.argv[1:])
    training(load_lib(*sys.argv[1:]), 0)
    # test()
    # loaded(var)"""
