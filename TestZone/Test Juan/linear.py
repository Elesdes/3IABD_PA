import ctypes
import sys

from typing import List, Any

from TestUtils import link_to_dll, image_opener

PATH_LIB_LINEAR = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/Framework/LinearModel/cmake-build-debug"
DLL_NAME_LINEAR = "libLinearModel"

PATH_TE_TEST = "G:\Mon Drive\Projet Annuel 3IABD\Data Lake\Tour Eiffel\img_test"
PATH_TE_TRAIN = "G:\Mon Drive\Projet Annuel 3IABD\Data Lake\Tour Eiffel\img_train"
PATH_ARC_TEST = "G:\Mon Drive\Projet Annuel 3IABD\Data Lake\Arc de triomphe\img_test"
PATH_ARC_TRAIN = "G:\Mon Drive\Projet Annuel 3IABD\Data Lake\Arc de triomphe\img_train"


def prepare_dll(linear_lib: ctypes.cdll, arr_type_y: List[Any]) -> ctypes.cdll:
    linear_lib.initModelWeights.argtypes = [ctypes.c_int32,
                                            ctypes.c_int32]
    linear_lib.initModelWeights.restype = ctypes.POINTER(ctypes.c_float)

    linear_lib.printFloatArray.argtypes = [ctypes.POINTER(ctypes.c_float),
                                           ctypes.c_int32]
    linear_lib.printFloatArray.restype = ctypes.c_float

    linear_lib.destroyFloatArray.argtypes = [ctypes.POINTER(ctypes.c_float)]
    linear_lib.destroyFloatArray.restype = None

    linear_lib.trainLinearInt.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_int32)),
                                          arr_type_y,
                                          ctypes.POINTER(ctypes.c_float),
                                          ctypes.c_int32,
                                          ctypes.c_int32,
                                          ctypes.c_int32]
    linear_lib.trainLinearInt.restype = ctypes.POINTER(ctypes.c_float)

    linear_lib.trainLinearFloat.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
                                            arr_type_y,
                                            ctypes.POINTER(ctypes.c_float),
                                            ctypes.c_int32,
                                            ctypes.c_int32,
                                            ctypes.c_int32]
    linear_lib.trainLinearFloat.restype = ctypes.POINTER(ctypes.c_float)

    linear_lib.predictLinearModelClassificationInt.argtypes = [ctypes.POINTER(ctypes.c_float),
                                                               ctypes.POINTER(ctypes.c_int32),
                                                               ctypes.c_int32]
    linear_lib.predictLinearModelClassificationInt.restype = ctypes.c_int32

    linear_lib.predictLinearModelClassificationFloat.argtypes = [ctypes.POINTER(ctypes.c_float),
                                                                 ctypes.POINTER(ctypes.c_float),
                                                                 ctypes.c_int32]
    linear_lib.predictLinearModelClassificationFloat.restype = ctypes.c_int32

    linear_lib.saveModelLinear.argtypes = [ctypes.POINTER(ctypes.c_float),
                                           ctypes.c_char_p,
                                           ctypes.c_int32,
                                           ctypes.c_double]
    linear_lib.saveModelLinear.restype = None

    linear_lib.loadModelLinear.argtypes = [ctypes.c_char_p]
    linear_lib.loadModelLinear.restype = ctypes.POINTER(ctypes.c_float)

    return linear_lib


def training(linear_lib: ctypes.cdll, n_iter: int, max_percentage: float) -> float:
    x, y = image_opener.fill_x_y(PATH_TE_TRAIN, PATH_ARC_TRAIN)
    C_TYPE = ctypes.c_int32 if isinstance(x[0][0], int) else ctypes.c_float
    print(C_TYPE)
    POINTER_C_TYPE = ctypes.POINTER(C_TYPE)
    ITLARR = C_TYPE * len(x[0])
    PITLARR = POINTER_C_TYPE * len(x)
    ptr = PITLARR



if __name__ == '__main__':
    training(link_to_dll.load_lib(PATH_LIB_LINEAR, DLL_NAME_LINEAR, *sys.argv[1:]), 100, 0)
