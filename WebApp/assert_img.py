import ctypes as ct
import os
from typing import Any

import numpy as np

# FRAMEWORK_LIB = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/WebApp/cmake-build-debug/libWebApp.dll"
FRAMEWORK_LIB = "libtesting.so"

LINEAR_SAVE = "Save/Linear"
MLP_SAVE = "Save/MLP"
RBF_SAVE = "Save/RBF/Classification/test.txt"
SVM_SAVE = "Save/SVM"


# Initialize dll for C++ / Python Interop
def init_dll(lib: ct.CDLL) -> ct.CDLL:
    # Linear
    lib.predictLinearModelClassificationInt.argtypes = [ct.POINTER(ct.c_float), ct.POINTER(ct.c_int32), ct.c_int32]
    lib.predictLinearModelClassificationInt.restype = ct.c_int32

    lib.loadModelLinear.argtypes = [ct.c_char_p]
    lib.loadModelLinear.restype = ct.POINTER(ct.c_float)

    lib.destroyFloatArray.argtypes = [ct.POINTER(ct.c_float)]
    lib.destroyFloatArray.restype = None
    # MLP
    lib.destroyMlpModel.argtypes = [ct.c_void_p]
    lib.destroyMlpModel.restype = None

    lib.destroyDoubleArray1D.argtypes = [ct.POINTER(ct.c_double)]
    lib.destroyDoubleArray1D.restype = None

    lib.predictMLPFloatMultipleOutputs.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_int32, ct.c_int32]
    lib.predictMLPFloatMultipleOutputs.restype = ct.POINTER(ct.c_double)

    lib.loadModelMLP.argtypes = [ct.c_char_p]
    lib.loadModelMLP.restype = ct.c_void_p

    lib.readArray.argtypes = [ct.POINTER(ct.c_double), ct.c_int32]
    lib.readArray.restype = ct.c_double
    # RBF
    lib.loadModelRBF.argtypes = [ct.POINTER(ct.c_char), ct.c_int32]
    lib.loadModelRBF.restype = ct.POINTER(ct.POINTER(ct.c_float))

    lib.newRBFWeights.argtypes = [ct.POINTER(ct.POINTER(ct.c_float)), ct.c_int32, ct.c_int32,
                                  ct.POINTER(ct.POINTER(ct.c_float)), ct.c_int32, ct.c_int32,
                                  ct.c_int32, ct.c_int32]
    lib.newRBFWeights.restype = ct.POINTER(ct.POINTER(ct.c_float))
    # SVM
    lib.freeArr.argtypes = [ct.POINTER(ct.c_double)]
    lib.freeArr.restype = None

    lib.loadSVM.argtypes = [ct.c_char_p]
    lib.loadSVM.restype = ct.POINTER(ct.c_double)

    lib.resultSVM.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_int32]
    lib.resultSVM.restype = ct.c_int32

    return lib


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
    my_dll = ct.CDLL(FRAMEWORK_LIB)
    my_dll = init_dll(my_dll)
    number_of_file = 0

    # Beginning
    w = list()
    for filename in os.listdir(LINEAR_SAVE):
        f = os.path.join(LINEAR_SAVE, filename)
        # /!/ Allocated ptr list, needs to be free/destroy
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
    my_dll = ct.CDLL(FRAMEWORK_LIB)
    my_dll = init_dll(my_dll)

    # Beginning
    # /!/ Allocated ptr list, needs to be free/destroy
    file_binary = MLP_SAVE.encode('ascii')
    MLP = my_dll.loadModelMLP(file_binary)

    result = my_dll.predictMLPFloatMultipleOutputs(MLP, ptr_x[0], 1, 1)
    verify = 0
    code_res = -1
    res_list = list()
    for iter_test in range(1):
        res_list.append(my_dll.readArray(result, iter_test))
    for i in res_list:
        if i == -1:
            verify += 1
    if verify == 0:
        for case_num in range(1):
            if (result[case_num] == 1):
                code_res = case_num
    else:
        code_res = -1

    print("oui")
    my_dll.destroyDoubleArray1D(result)
    print("non")
    my_dll.destroyMlpModel(MLP)
    print("PA")

    return code_res


# RBF

# To convert to a double pointer
def convert_to_pointer(c_type: ct, array_input: Any, num_rows: int, num_cols_in_rows: int) -> Any:
    POINTER_C_TYPE = ct.POINTER(c_type)
    ITLARR = c_type * num_cols_in_rows
    PITLARR = POINTER_C_TYPE * num_rows
    ptr = PITLARR()
    for i in range(num_rows):
        ptr[i] = ITLARR()
        for j in range(num_cols_in_rows):
            ptr[i][j] = array_input[i][j]
    return ptr


def convert_array(array_input: Any) -> Any:
    array_output = convert_to_pointer(ct.c_float, array_input, len(array_input), len(array_input[0]))
    array_output = ct.cast(array_output, ct.POINTER(ct.POINTER(ct.c_float)))
    return array_output


def RBF_assert(x):
    rbf_lib = ct.CDLL(FRAMEWORK_LIB)
    rbf = init_dll(rbf_lib)

    new_x = []
    new_x.append(x.astype(int))
    new_x = np.array(new_x)

    valid = 0
    binary_file = RBF_SAVE.encode('ascii')
    centers_testing = convert_array(new_x)
    weights = rbf.loadModelRBF(binary_file, 4)

    outputs = [[1, -1, -1, -1],
               [-1, 1, -1, -1],
               [-1, -1, 1, -1],
               [-1, -1, -1, 1]]

    res = rbf.newRBFWeights(weights, 1, 4,
                            centers_testing, len(new_x), len(new_x[0]),
                            2, 1)

    for i in range(0, len(outputs)):
        for j in range(0, 4):
            if int(res[0][j]) == outputs[i][j]:
                valid += 1
                if valid == 4:
                    return i;
            valid = 0
    if valid < 4:
        return -1


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
    my_dll = ct.CDLL(FRAMEWORK_LIB)
    my_dll = init_dll(my_dll)

    w = list()
    final_result = 0
    number_of_file = 0
    for filename in os.listdir(SVM_SAVE):
        f = os.path.join(SVM_SAVE, filename)
        # /!/ Allocated ptr list, needs to be free/destroy
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


def assert_img(img):
    return linear_assert(img)


def linear_was_chosen(img):
    return linear_assert(img)


def MLP_was_chosen(img):
    return MLP_assert(img)


def RBF_was_chosen(img):
    return RBF_assert(img)


def SVM_was_chosen(img):
    return SVM_assert(img)
