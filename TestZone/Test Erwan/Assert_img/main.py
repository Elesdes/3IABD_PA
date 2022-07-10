import ctypes as ct
import os
import sys

import numpy as np
from PIL import Image

LINEAR_LIB = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/LinearLib/cmake-build-debug/Library.dll"
MLP_LIB = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/MLPLib/cmake-build-debug/MLPLib.dll"
SVM_LIB = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/SVMLib/cmake-build-debug/SVMLib.dll"

LINEAR_SAVE = ""
MLP_SAVE = ""
SVM_SAVE = ""


def prepare_dll_linear(my_dll):
    my_dll.destroyFloatArray.argtypes = [ct.POINTER(ct.c_float)]
    my_dll.destroyFloatArray.restype = None

    my_dll.predictLinearModelClassificationInt.argtypes = [ct.POINTER(ct.c_float), ct.POINTER(ct.c_int32), ct.c_int32]
    my_dll.predictLinearModelClassificationInt.restype = ct.c_int32

    my_dll.loadModelLinear.argtypes = [ct.c_char_p]
    my_dll.loadModelLinear.restype = ct.POINTER(ct.c_float)

    return my_dll


def prepare_dll_mlp(my_dll):
    my_dll.destroyMlpModel.argtypes = [ct.c_void_p]
    my_dll.destroyMlpModel.restype = None

    my_dll.predictMLPFloatMultipleOutputs.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_int32, ct.c_int32]
    my_dll.predictMLPFloatMultipleOutputs.restype = ct.POINTER(ct.c_double)

    my_dll.loadModelMLP.argtypes = [ct.c_char_p]
    my_dll.loadModelMLP.restype = ct.c_void_p

    return my_dll


def prepare_dll_SVM(my_dll):
    my_dll.freeArr.argtypes = [ct.POINTER(ct.c_double)]
    my_dll.freeArr.restype = None

    my_dll.resultSVM.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_int32]
    my_dll.resultSVM.restype = ct.c_int32

    my_dll.loadSVM.argtypes = [ct.c_char_p]
    my_dll.loadSVM.restype = ct.POINTER(ct.c_double)

    return my_dll


def check_if_duplicates(x):
    if x.count(1) > 1:
        return True
    return False


def assert_class(result):
    # Retour à changer
    if result[0] == 1:
        return "Tour Eiffel"
    if result[1] == 1:
        return "Arc de triomphe"
    if result[2] == 1:
        return "Louvre"
    if result[3] == 1:
        return "Pantheon"


def load_file_list(dir):
    file_list = list()
    for filename in os.listdir(dir):
        file_list.append(os.path.join(dir, filename))
    return file_list


def assert_linear(my_dll, x):
    # Set X
    C_TYPE = ct.c_int32
    POINTER_C_TYPE = ct.POINTER(C_TYPE)
    ITLARR = C_TYPE * len(x)
    ptr = ITLARR()
    for i in range(len(x)):
        ptr[i] = x[i]

    w = list()
    result = list()
    file_list = load_file_list(LINEAR_SAVE)
    for file in file_list:
        # /!\ Allocated ptr list, needs to be free/destroy
        w.append(my_dll.loadModelLinear(file))
    for each_w in w:
        result.append(my_dll.predictLinearModelClassificationInt(each_w, ptr, len(x) + 1))
    if not check_if_duplicates(result):
        print("A changer pour dire que c'est une erreur cette image")
    else:
        print(assert_class(result))
    # Destroy
    for i in range(len(file_list)):
        my_dll.destroyFloatArray(w[i])


def assert_MLP(my_dll, x):
    # Set X
    C_TYPE = ct.c_int32
    POINTER_C_TYPE = ct.POINTER(C_TYPE)
    ITLARR = C_TYPE * len(x)
    ptr = ITLARR()
    for i in range(len(x)):
        ptr[i] = x[i]

    w = list()
    result = list()
    file_list = load_file_list(MLP_SAVE)
    for file in file_list:
        # /!\ Allocated ptr list, needs to be free/destroy
        w.append(my_dll.loadModelMLP(file))
    for each_w in w:
        result.append(my_dll.predictMLPFloatMultipleOutputs(each_w, ptr, 1, 4)) # 4 because there are 4 outputs
    if not check_if_duplicates(result):
        print("A changer pour dire que c'est une erreur cette image")
    else:
        print(assert_class(result))
    # Destroy
    for i in range(len(file_list)):
        my_dll.destroyFloatArray(w[i])


def assert_SVM(my_dll, x):
    # Set X
    # Peut provoquer une erreur à cause de la conversion implicite de x->int en x->double
    C_TYPE = ct.c_double
    POINTER_C_TYPE = ct.POINTER(C_TYPE)
    ITLARR = C_TYPE * len(x)
    ptr = ITLARR()
    for i in range(len(x)):
        ptr[i] = x[i]

    w = list()
    result = list()
    file_list = load_file_list(SVM_SAVE)
    for file in file_list:
        # /!\ Allocated ptr list, needs to be free/destroy
        w.append(my_dll.loadSVM(file))
    for each_w in w:
        result.append(my_dll.resultSVM(ptr, each_w, len(x)))
    if not check_if_duplicates(result):
        print("A changer pour dire que c'est une erreur cette image")
    else:
        print(assert_class(result))
    # Destroy
    for i in range(len(file_list)):
        my_dll.freeArr(w[i])


if __name__ == '__main__':
    # Ce fichier, c'est le fichier récupéré du site qui est donc un tableau de pixel
    file = [255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 0, 0, 255]

    # Si le linear est bon
    my_dll = ct.CDLL(LINEAR_LIB)
    assert_linear(my_dll, file)

    # Si le MLP est bon
    my_dll = ct.CDLL(MLP_LIB)
    assert_MLP(my_dll, file)

    # Si le SVM est bon
    my_dll = ct.CDLL(SVM_LIB)
    assert_SVM(my_dll, file)