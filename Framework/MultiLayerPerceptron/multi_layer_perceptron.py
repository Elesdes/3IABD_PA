import ctypes
from typing import List, Any

from Framework.Utils import image_opener as io
from Framework.Utils import set_var as sv

PATH_TE_TEST = "G:\Mon Drive\Projet Annuel 3IABD\Data Lake\Tour Eiffel\img_test"
PATH_TE_TRAIN = "G:\Mon Drive\Projet Annuel 3IABD\Data Lake\Tour Eiffel\img_train"
PATH_ARC_TEST = "G:\Mon Drive\Projet Annuel 3IABD\Data Lake\Arc de triomphe\img_test"
PATH_ARC_TRAIN = "G:\Mon Drive\Projet Annuel 3IABD\Data Lake\Arc de triomphe\img_train"


def prepare_dll(my_dll: ctypes.cdll, npl: List[Any], is_not_loaded: bool) -> ctypes.cdll:
    my_dll.destroyMlpModel.argtypes = [ctypes.c_void_p]
    my_dll.destroyMlpModel.restype = None

    if (is_not_loaded):
        arr_type_npl = ctypes.c_int32 * len(npl)
        my_dll.destroyDoubleArray3D.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
                                                arr_type_npl, ctypes.c_int32]
        my_dll.destroyDoubleArray3D.restype = None
        my_dll.initiateMLP.argtypes = [arr_type_npl, ctypes.c_int32]
        my_dll.initiateMLP.restype = ctypes.c_void_p

    my_dll.destroyDoubleArray2D.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_double)), ctypes.c_int32]
    my_dll.destroyDoubleArray2D.restype = None

    my_dll.destroyIntArray2D.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32]
    my_dll.destroyIntArray2D.restype = None

    my_dll.predictMLP.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
    my_dll.predictMLP.restype = ctypes.c_double

    my_dll.trainMLP.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32,
                                ctypes.c_int32, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_float,
                                ctypes.c_int32, ctypes.c_int32]
    my_dll.trainMLP.restype = None

    my_dll.saveModelMLP.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32]
    my_dll.saveModelMLP.restype = None

    my_dll.useLoadedMLP.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_char_p, ctypes.c_int32]
    my_dll.useLoadedMLP.restype = ctypes.c_double

    my_dll.returnLenModel.argtypes = [ctypes.c_void_p]
    my_dll.returnLenModel.restype = ctypes.c_int32

    return my_dll


def training(my_dll: ctypes.cdll) -> None:
    # doing_resizer_and_gray()
    x, y = io.fill_x_and_y(PATH_TE_TRAIN, PATH_ARC_TRAIN)
    final_result = 0
    iter = 1
    learning_rate = 0.01
    if isinstance(x[0][0], int):
        C_TYPE = ctypes.c_int32
    else:
        C_TYPE = ctypes.c_long
    POINTER_C_TYPE = ctypes.POINTER(C_TYPE)
    POINTER_POINTER_C_TYPE = ctypes.POINTER(POINTER_C_TYPE)
    ITLARR = C_TYPE * len(x)
    PITLARR = POINTER_C_TYPE * len(x)
    ptr_x = PITLARR()
    ptr_y = ITLARR()
    print(len(x))
    print(len(y))
    for i in range(len(x)):
        ptr_x[i] = ITLARR()
        for j in range(len(x[0])):
            ptr_x[i][j] = x[i][j]
    for i in range(len(y)):
        ptr_y[i] = y[i]

    rowsXLen, colsXLen, rowsYLen, arr_type_y = sv.set_var(x, y)
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


def loaded(my_dll: ctypes.cdll) -> None:
    x, y = io.fill_x_and_y(PATH_TE_TEST, PATH_ARC_TEST)
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

    rowsXLen, colsXLen, rowsYLen, arr_type_y = sv.set_var(x, y)
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


"""if __name__ == '__main__':
    print("In Cpp :")
    var = load_lib(*sys.argv[1:])
    # training(var)
    # loaded(var)"""
