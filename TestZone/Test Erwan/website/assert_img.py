import ctypes as ct
import os
import sys
import statistics

import numpy as np
from PIL import Image

LINEAR_LIB = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/LinearLib/cmake-build-debug/Library.dll"
MLP_LIB = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/MLPLib/cmake-build-debug/MLPLib.dll"
SVM_LIB = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/SVMLib/cmake-build-debug/SVMLib.dll"

LINEAR_SAVE = ""
MLP_SAVE = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/Exe_all_lib/save/mlp/"
SVM_SAVE = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/Exe_all_lib/save/svm/DS32_classique-Test_max_1001-LA0.005-TH0.01/num0/"


def prepare_dll_linear(my_dll):
    my_dll.initModelWeights.argtypes = [ct.c_int32]
    my_dll.initModelWeights.restype = ct.POINTER(ct.c_float)

    my_dll.printFloatArray.argtypes = [ct.POINTER(ct.c_float), ct.c_int32]
    my_dll.printFloatArray.restype = ct.c_float

    my_dll.destroyFloatArray.argtypes = [ct.POINTER(ct.c_float)]
    my_dll.destroyFloatArray.restype = None

    my_dll.trainLinearInt.argtypes = [ct.POINTER(ct.POINTER(ct.c_int32)), ct.POINTER(ct.c_int32),
                                      ct.POINTER(ct.c_float), ct.c_int32, ct.c_int32, ct.c_int32, ct.c_float]
    my_dll.trainLinearInt.restype = ct.POINTER(ct.c_float)

    my_dll.predictLinearModelClassificationInt.argtypes = [ct.POINTER(ct.c_float), ct.POINTER(ct.c_int32), ct.c_int32]
    my_dll.predictLinearModelClassificationInt.restype = ct.c_int32

    my_dll.loadModelLinear.argtypes = [ct.c_char_p]
    my_dll.loadModelLinear.restype = ct.POINTER(ct.c_float)

    my_dll.printIntArray.argtypes = [ct.POINTER(ct.c_int32), ct.c_int32]
    my_dll.printIntArray.restype = None

    return my_dll


def prepare_dll_mlp(my_dll):
    my_dll.destroyMlpModel.argtypes = [ct.c_void_p]
    my_dll.destroyMlpModel.restype = None

    my_dll.destroyDoubleArray1D.argtypes = [ct.POINTER(ct.c_double)]
    my_dll.destroyDoubleArray1D.restype = None

    my_dll.predictMLPFloatMultipleOutputs.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_int32, ct.c_int32]
    my_dll.predictMLPFloatMultipleOutputs.restype = ct.POINTER(ct.c_double)

    my_dll.trainMLPFloatMultipleOutputs.argtypes = [ct.c_void_p, ct.POINTER(ct.POINTER(ct.c_float)), ct.c_int32,
                                                    ct.c_int32, ct.POINTER(ct.POINTER(ct.c_int32)), ct.c_int32,
                                                    ct.c_int32, ct.c_float, ct.c_int32, ct.c_int32]
    my_dll.trainMLPFloatMultipleOutputs.restype = None

    my_dll.loadModelMLP.argtypes = [ct.c_char_p]
    my_dll.loadModelMLP.restype = ct.c_void_p

    my_dll.readArray.argtypes = [ct.POINTER(ct.c_double), ct.c_int32]
    my_dll.readArray.restype = ct.c_double

    return my_dll


def prepare_dll_SVM(my_dll):
    my_dll.freeArr.argtypes = [ct.POINTER(ct.c_double)]
    my_dll.freeArr.restype = None

    my_dll.initSVMWeight.argtypes = [ct.c_int32]
    my_dll.initSVMWeight.restype = ct.POINTER(ct.c_double)

    my_dll.initSVMWeightDerive.argtypes = [ct.c_int32]
    my_dll.initSVMWeightDerive.restype = ct.POINTER(ct.c_double)

    my_dll.getHingeLoss.argtypes = [ct.POINTER(ct.c_double), ct.c_int32, ct.POINTER(ct.c_double), ct.c_int32,
                                    ct.c_int32]
    my_dll.getHingeLoss.restype = ct.c_double

    my_dll.getSVMCost.argtypes = [ct.POINTER(ct.POINTER(ct.c_double)), ct.POINTER(ct.c_int32), ct.POINTER(ct.c_double),
                                  ct.POINTER(ct.c_double), ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32]
    my_dll.getSVMCost.restype = ct.c_double

    my_dll.resultSVM.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_int32]
    my_dll.resultSVM.restype = ct.c_int32

    my_dll.trainSVM.argtypes = [ct.POINTER(ct.POINTER(ct.c_double)), ct.POINTER(ct.c_int32), ct.POINTER(ct.c_double),
                                ct.POINTER(ct.c_double), ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_double,
                                ct.c_double, ct.c_int32]
    my_dll.trainSVM.restype = ct.POINTER(ct.c_double)

    my_dll.saveSVM.argtypes = [ct.POINTER(ct.c_double), ct.c_char_p, ct.c_int32, ct.c_double]
    my_dll.saveSVM.restype = None

    my_dll.loadSVM.argtypes = [ct.c_char_p]
    my_dll.loadSVM.restype = ct.POINTER(ct.c_double)

    return my_dll


def linear_assert(x):
    # doing_resizer_and_gray()
    code_res = 0

    # Set X
    C_TYPE = ct.c_int32
    POINTER_C_TYPE = ct.POINTER(C_TYPE)
    ITLARR = C_TYPE * len(x)
    PITLARR = POINTER_C_TYPE * 1
    ptr = PITLARR()
    for i in range(1):
        ptr[i] = ITLARR()
        for j in range(len(x)):
            ptr[i][j] = x[j]

    # Set var
    rowsXLen = 1
    colsXLen = len(x)
    rowsWLen = colsXLen + 1
    my_dll = ct.CDLL(LINEAR_LIB)
    my_dll = prepare_dll_linear(my_dll)
    number_of_file = 0

    # Beginning
    w = list()
    for filename in os.listdir(LINEAR_SAVE):
        f = os.path.join(LINEAR_SAVE, filename)
        # /!\ Allocated ptr list, needs to be free/destroy
        file_binary = f.encode('ascii')
        w.append(my_dll.loadModelLinear(file_binary))
        number_of_file += 1

    result = list()
    verify = 0

    for iter_res in range(number_of_file):
        result.append(my_dll.predictLinearModelClassificationInt(w[iter_res], ptr[0], rowsWLen))
    for case_num in range(number_of_file):
        if (result[case_num] == -1):
            verify += 1
    if (verify == number_of_file - 1):
        for case_num in range(number_of_file):
            if (result[case_num] == 1):
                code_res = case_num
    else:
        code_res = -1
    # Save and destroy
    for i in range(len(w)):
        # May be change in the near future
        my_dll.destroyFloatArray(w[i])
    return code_res


def MLP_assert(x):
    # Set X
    C_TYPE = ct.c_float
    POINTER_C_TYPE = ct.POINTER(C_TYPE)
    POINTER_POINTER_C_TYPE = ct.POINTER(POINTER_C_TYPE)
    ITLARR = C_TYPE * len(x)
    PITLARR = POINTER_C_TYPE * 1
    ptr_x = PITLARR()
    for i in range(1):
        ptr_x[i] = ITLARR()
        for j in range(len(x)):
            ptr_x[i][j] = x[j]

    # Set var
    rowsXLen = 1
    colsXLen = len(x)
    my_dll = ct.CDLL(MLP_LIB)
    my_dll = prepare_dll_mlp(my_dll)
    number_of_file = 0
    for filename in os.listdir(SVM_SAVE):
        number_of_file += 1

    # Beginning
    for filename in os.listdir(MLP_SAVE):
        f = os.path.join(MLP_SAVE, filename)
        # /!\ Allocated ptr list, needs to be free/destroy
        file_binary = f.encode('ascii')
        MLP = my_dll.loadModelMLP(file_binary)

    result = my_dll.predictMLPFloatMultipleOutputs(MLP, ptr_x[0], 1, number_of_file)
    verify = 0
    code_res = -1
    res_list = list()
    for iter_test in range(number_of_file):
        res_list.append(my_dll.readArray(result, iter_test))
    print(res_list)
    for i in res_list:
        if i < 0:
            verify += 1
    if verify == number_of_file - 1:
        for case_num in range(number_of_file):
            if (result[case_num] > 0):
                code_res = case_num
    else:
        code_res = -1
    my_dll.destroyDoubleArray1D(result)
    my_dll.destroyMlpModel(MLP)

    return code_res


def SVM_assert(x):
    # Set X
    C_TYPE_X = ct.c_double
    POINTER_C_TYPE_X = ct.POINTER(C_TYPE_X)
    ITLARR_X = C_TYPE_X * len(x)
    PITLARR_X = POINTER_C_TYPE_X * 1
    ptr_x = PITLARR_X()
    for i in range(1):
        ptr_x[i] = ITLARR_X()
        for j in range(len(x)):
            ptr_x[i][j] = x[j]

    # Set var
    colsXLen = 1
    rowsXLen = len(x)
    rowsWLen = rowsXLen + 1
    my_dll = ct.CDLL(SVM_LIB)
    my_dll = prepare_dll_SVM(my_dll)

    w = list()
    final_result = 0
    number_of_file = 0
    for filename in os.listdir(SVM_SAVE):
        f = os.path.join(SVM_SAVE, filename)
        # /!\ Allocated ptr list, needs to be free/destroy
        file_binary = f.encode('ascii')
        w.append(my_dll.loadSVM(file_binary))
        number_of_file += 1

    result = list()
    verify = 0
    code_res = -1

    for iter_res in range(number_of_file):
        result.append(my_dll.resultSVM(ptr_x[0], w[iter_res], rowsXLen))
    for case_num in range(number_of_file):
        if (result[case_num] == -1):
            verify += 1
    if (verify == number_of_file - 1):
        for case_num in range(number_of_file):
            if (result[case_num] == 1):
                code_res = case_num
    else:
        code_res = -1

    # Save and destroy
    for i in range(len(w)):
        # May be change in the near future
        my_dll.freeArr(w[i])

    return code_res


# TODO: a changer par le meilleur algo
def assert_img(img):
    return -1


def linear_was_chosen(img):
    # TODO: a decommenter pour la prod
    # return linear_assert(img)
    return 0


def MLP_was_chosen(img):
    return MLP_assert(img)


# TODO: a changer par JUAN
def RBF_was_chosen(img):
    return 2


def SVM_was_chosen(img):
    return SVM_assert(img)
