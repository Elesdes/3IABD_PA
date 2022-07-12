import ctypes as ct
import os
import sys

import numpy as np
from PIL import Image

LINEAR_LIB = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/LinearLib/cmake-build-debug/Library.dll"
MLP_LIB = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/MLPLib/cmake-build-debug/MLPLib.dll"
SVM_LIB = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/SVMLib/cmake-build-debug/SVMLib.dll"

EIFFEL_TOWER_TRAINING = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/Exe_all_lib/datalake/eiffel/training/"
EIFFEL_TOWER_TESTING = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/Exe_all_lib/datalake/eiffel/testing/"
TRIUMPHAL_ARC_TRAINING = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/Exe_all_lib/datalake/triumphal/training/"
TRIUMPHAL_ARC_TESTING = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/Exe_all_lib/datalake/triumphal/testing/"
LOUVRE_TRAINING = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/Exe_all_lib/datalake/louvre/training/"
LOUVRE_TESTING = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/Exe_all_lib/datalake/louvre/testing/"
PANTHEON_TRAINING = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/Exe_all_lib/datalake/pantheon/training/"
PANTHEON_TESTING = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/Exe_all_lib/datalake/pantheon/testing/"

LINEAR_SAVE = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/Exe_all_lib/save/linear/"
MLP_SAVE = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/Exe_all_lib/save/mlp/"
SVM_SAVE = "C:/Users/erwan/Desktop/ESGI/S6/Projet Annuel/Exe_all_lib/save/svm/"

def add_img(x, y, file_path, value):
    for filename in os.listdir(file_path):
        f = os.path.join(file_path, filename)
        if os.path.isfile(f):
            img = Image.open(f)
            img = np.array(img)
            img = np.ravel(img)
            x.append(img)
            y.append(value)


def prepare_dll_linear(my_dll):
    my_dll.initModelWeights.argtypes = [ct.c_int32]
    my_dll.initModelWeights.restype = ct.POINTER(ct.c_float)

    my_dll.printFloatArray.argtypes = [ct.POINTER(ct.c_float), ct.c_int32]
    my_dll.printFloatArray.restype = ct.c_float

    my_dll.destroyFloatArray.argtypes = [ct.POINTER(ct.c_float)]
    my_dll.destroyFloatArray.restype = None

    my_dll.trainLinearInt.argtypes = [ct.POINTER(ct.POINTER(ct.c_int32)), ct.POINTER(ct.c_int32), ct.POINTER(ct.c_float), ct.c_int32, ct.c_int32, ct.c_int32, ct.c_float]
    my_dll.trainLinearInt.restype = ct.POINTER(ct.c_float)

    my_dll.predictLinearModelClassificationInt.argtypes = [ct.POINTER(ct.c_float), ct.POINTER(ct.c_int32), ct.c_int32]
    my_dll.predictLinearModelClassificationInt.restype = ct.c_int32

    my_dll.saveModelLinear.argtypes = [ct.POINTER(ct.c_float), ct.c_char_p, ct.c_int32, ct.c_double]
    my_dll.saveModelLinear.restype = None

    my_dll.printIntArray.argtypes = [ct.POINTER(ct.c_int32), ct.c_int32]
    my_dll.printIntArray.restype = None

    return my_dll


def set_var_linear(x, y):
    rowsXLen = len(x)
    colsXLen = len(x[0])
    rowsWLen = colsXLen + 1
    rowsYLen = len(y)
    colsYLen = len(y[0])
    return rowsXLen, colsXLen, rowsWLen, rowsYLen, colsYLen


def prepare_dll_mlp(my_dll, npl):
    my_dll.destroyMlpModel.argtypes = [ct.c_void_p]
    my_dll.destroyMlpModel.restype = None

    arr_type_npl = ct.c_int32 * len(npl)
    my_dll.initiateMLP.argtypes = [arr_type_npl, ct.c_int32]
    my_dll.initiateMLP.restype = ct.c_void_p

    my_dll.destroyDoubleArray1D.argtypes = [ct.POINTER(ct.c_double)]
    my_dll.destroyDoubleArray1D.restype = None

    my_dll.predictMLPFloatMultipleOutputs.argtypes = [ct.c_void_p, ct.POINTER(ct.c_float), ct.c_int32, ct.c_int32]
    my_dll.predictMLPFloatMultipleOutputs.restype = ct.POINTER(ct.c_double)

    my_dll.trainMLPFloatMultipleOutputs.argtypes = [ct.c_void_p, ct.POINTER(ct.POINTER(ct.c_float)), ct.c_int32, ct.c_int32, ct.POINTER(ct.POINTER(ct.c_int32)), ct.c_int32, ct.c_int32, ct.c_float, ct.c_int32, ct.c_int32]
    my_dll.trainMLPFloatMultipleOutputs.restype = None

    my_dll.saveModelMLP.argtypes = [ct.c_void_p, ct.c_char_p, ct.c_int32, ct.c_double]
    my_dll.saveModelMLP.restype = None

    my_dll.readArray.argtypes = [ct.POINTER(ct.c_double), ct.c_int32]
    my_dll.readArray.restype = ct.c_double

    return my_dll


def set_var_mlp(x, y, npl):
    rowsXLen = len(x)
    colsXLen = len(x[0])
    rowsYLen = len(y)
    colsYLen = len(y[0])
    arr_type_y = ct.c_int32 * len(y)
    len_npl = len(npl)
    return rowsXLen, colsXLen, rowsYLen, colsYLen, arr_type_y, len_npl


def prepare_dll_SVM(my_dll):
    my_dll.freeArr.argtypes = [ct.POINTER(ct.c_double)]
    my_dll.freeArr.restype = None

    my_dll.initSVMWeight.argtypes = [ct.c_int32]
    my_dll.initSVMWeight.restype = ct.POINTER(ct.c_double)

    my_dll.initSVMWeightDerive.argtypes = [ct.c_int32]
    my_dll.initSVMWeightDerive.restype = ct.POINTER(ct.c_double)

    my_dll.getHingeLoss.argtypes = [ct.POINTER(ct.c_double), ct.c_int32, ct.POINTER(ct.c_double), ct.c_int32, ct.c_int32]
    my_dll.getHingeLoss.restype = ct.c_double

    my_dll.getSVMCost.argtypes = [ct.POINTER(ct.POINTER(ct.c_double)), ct.POINTER(ct.c_int32), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32]
    my_dll.getSVMCost.restype = ct.c_double

    my_dll.resultSVM.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_int32]
    my_dll.resultSVM.restype = ct.c_int32

    my_dll.trainSVM.argtypes = [ct.POINTER(ct.POINTER(ct.c_double)), ct.POINTER(ct.c_int32), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_int32, ct.c_int32, ct.c_int32, ct.c_int32, ct.c_double, ct.c_double, ct.c_int32]
    my_dll.trainSVM.restype = ct.POINTER(ct.c_double)

    my_dll.saveSVM.argtypes = [ct.POINTER(ct.c_double), ct.c_char_p, ct.c_int32, ct.c_double]
    my_dll.saveSVM.restype = None

    my_dll.loadSVM.argtypes = [ct.c_char_p]
    my_dll.loadSVM.restype = ct.POINTER(ct.c_double)

    return my_dll


def set_var_svm(x, y):
    colsXLen = len(x)
    rowsXLen = len(x[0])
    rowsWLen = rowsXLen + 1
    rowsYLen = len(y)
    colsYLen = len(y[0])
    return rowsXLen, colsXLen, rowsWLen, rowsYLen, colsYLen


def training_linear(my_dll, x_training, y_training, x_testing, y_testing, iter, learning_rate):
    # doing_resizer_and_gray()
    final_result = 0

    # Set X
    C_TYPE = ct.c_int32
    POINTER_C_TYPE = ct.POINTER(C_TYPE)
    ITLARR = C_TYPE * len(x_training[0])
    PITLARR = POINTER_C_TYPE * len(x_training)
    ptr = PITLARR()
    for i in range(len(x_training)):
        ptr[i] = ITLARR()
        for j in range(len(x_training[0])):
            ptr[i][j] = x_training[i][j]
    # Set Y
    ITLARRY = ct.c_int32 * len(y_training)
    PITLARRY = ct.POINTER(ct.c_int32) * len(y_training[0])
    ptr_y = PITLARRY()
    for i in range(len(y_training[0])):
        ptr_y[i] = ITLARRY()
        for j in range(len(y_training)):
            ptr_y[i][j] = y_training[j][i]
    # Set var
    rowsXLen, colsXLen, rowsWLen, rowsYLen, colsYLen = set_var_linear(x_training, y_training)
    my_dll = prepare_dll_linear(my_dll)

    # Beginning
    print("--- Training Linear ---")
    w = list()
    for i in range(len(y_training[0])):
        # /!\ Allocated ptr list, needs to be free/destroy
        w.append(my_dll.initModelWeights(rowsWLen))
        w[i] = my_dll.trainLinearInt(ptr, ptr_y[i], w[i], rowsXLen, rowsWLen, iter, learning_rate)
    for i in range(rowsXLen):
        result = list()
        verify = 0
        for iter_res in range(len(y_training[0])):
            result.append(my_dll.predictLinearModelClassificationInt(w[iter_res], ptr[i], rowsWLen))
        for case_num in range(len(y_training[0])):
            if (result[case_num] == y_training[i][case_num]):
                verify += 1
        if (verify == len(y_training[i])):
            final_result += 1
    print("Result : ", final_result, "/", len(y_training), " | ", final_result / len(y_training) * 100, "%")

    # Set X
    C_TYPE = ct.c_int32
    POINTER_C_TYPE = ct.POINTER(C_TYPE)
    ITLARR = C_TYPE * len(x_testing[0])
    PITLARR = POINTER_C_TYPE * len(x_testing)
    ptr = PITLARR()
    for i in range(len(x_testing)):
        ptr[i] = ITLARR()
        for j in range(len(x_testing[0])):
            ptr[i][j] = x_testing[i][j]
    # Set Y
    ITLARRY = ct.c_int32 * len(y_testing)
    PITLARRY = ct.POINTER(ct.c_int32) * len(y_testing[0])
    ptr_y = PITLARRY()
    for i in range(len(y_testing[0])):
        ptr_y[i] = ITLARRY()
        for j in range(len(y_testing)):
            ptr_y[i][j] = y_testing[j][i]
    # Set var
    rowsXLen, colsXLen, rowsWLen, rowsYLen, colsYLen = set_var_linear(x_testing, y_testing)

    print("--- Testing Linear ---")
    final_result = 0
    for i in range(rowsXLen):
        result = list()
        verify = 0
        for iter_res in range(len(y_testing[0])):
            result.append(my_dll.predictLinearModelClassificationInt(w[iter_res], ptr[i], rowsWLen))
        for case_num in range(len(y_testing[0])):
            if (result[case_num] == y_testing[i][case_num]):
                verify += 1
        if (verify == len(y_testing[i])):
            final_result += 1
    print("Result : ", final_result, "/", len(y_testing), " | ", final_result / len(y_testing) * 100, "%")

    # Save and destroy
    for i in range(len(w)):
        # May be change in the near future
        str_file = LINEAR_SAVE + "Linear_" + str(final_result/len(y_testing)) + "_" + str(i) + ".txt"
        str_file = str_file.encode('UTF-8')
        my_dll.saveModelLinear(w[i], str_file, rowsWLen, final_result / len(y_testing) * 100)
        my_dll.destroyFloatArray(w[i])


def training_mlp(my_dll, x_training, y_training, x_testing, y_testing, npl, iter, learning_rate):
    # Set X
    C_TYPE = ct.c_float
    POINTER_C_TYPE = ct.POINTER(C_TYPE)
    POINTER_POINTER_C_TYPE = ct.POINTER(POINTER_C_TYPE)
    ITLARR = C_TYPE * len(x_training[0])
    PITLARR = POINTER_C_TYPE * len(x_training)
    ptr_x = PITLARR()
    for i in range(len(x_training)):
        ptr_x[i] = ITLARR()
        for j in range(len(x_training[0])):
            ptr_x[i][j] = x_training[i][j]

    # Set Y
    ITLARRY = ct.c_long * len(y_training[0])
    PITLARRY = ct.POINTER(ct.c_long) * len(y_training)
    ptr_y = PITLARRY()
    for i in range(len(y_training)):
        ptr_y[i] = ITLARRY()
        for j in range(len(y_training[0])):
            ptr_y[i][j] = y_training[i][j]

    # Set var
    rowsXLen, colsXLen, rowsYLen, colsYlen, arr_type_y, len_npl = set_var_mlp(x_training, y_training, npl)
    arr_type_npl = ct.c_int32 * len(npl)
    my_dll = prepare_dll_mlp(my_dll, npl)

    # Beginning
    print("--- Training MLP ---")
    MLP = my_dll.initiateMLP(arr_type_npl(*npl), len_npl)
    my_dll.trainMLPFloatMultipleOutputs(MLP, ptr_x, rowsXLen, colsXLen, ptr_y, rowsYLen, colsYlen, learning_rate, 1, iter)
    final_result = 0
    for i in range(rowsXLen):
        result = my_dll.predictMLPFloatMultipleOutputs(MLP, ptr_x[i], 1, len(y_training[0]))
        verify = 0
        res_list = list()
        for iter_test in range(len(y_training[0])):
            res_list.append(my_dll.readArray(result, iter_test))
        for case_num, res_y in enumerate(y_training[i]):
            if (res_y > 0 and my_dll.readArray(result, case_num) > 0) or (res_y < 0 and my_dll.readArray(result, case_num) < 0):
                verify = verify + 1
        if verify == len(y_training[0]):
            final_result += 1

        my_dll.destroyDoubleArray1D(result)
    print("Result : ", final_result, "/", len(y_training), " | ", final_result / len(y_training) * 100, "%")
    pourcentage = final_result / len(y_training) * 100

    # Set X
    C_TYPE = ct.c_float
    POINTER_C_TYPE = ct.POINTER(C_TYPE)
    POINTER_POINTER_C_TYPE = ct.POINTER(POINTER_C_TYPE)
    ITLARR = C_TYPE * len(x_testing[0])
    PITLARR = POINTER_C_TYPE * len(x_testing)
    ptr_x = PITLARR()
    for i in range(len(x_testing)):
        ptr_x[i] = ITLARR()
        for j in range(len(x_testing[0])):
            ptr_x[i][j] = x_testing[i][j]
    # Set Y
    ITLARRY = ct.c_long * len(y_testing[0])
    PITLARRY = ct.POINTER(ct.c_long) * len(y_testing)
    ptr_y = PITLARRY()
    for i in range(len(y_testing)):
        ptr_y[i] = ITLARRY()
        for j in range(len(y_testing[0])):
            ptr_y[i][j] = y_testing[i][j]

    print("--- Testing MLP ---")
    rowsXLen, colsXLen, rowsYLen, colsYlen, arr_type_y, len_npl = set_var_mlp(x_testing, y_testing, npl)
    final_result = 0
    for i in range(rowsXLen):
        result = my_dll.predictMLPFloatMultipleOutputs(MLP, ptr_x[i], 1, len(y_testing[0]))
        verify = 0
        res_list = list()
        for iter_test in range(len(y_testing[0])):
            res_list.append(my_dll.readArray(result, iter_test))
        for case_num, res_y in enumerate(y_testing[i]):
            if (res_y > 0 and my_dll.readArray(result, case_num) > 0) or (
                    res_y < 0 and my_dll.readArray(result, case_num) < 0):
                verify = verify + 1
        if verify == len(y_testing[0]):
            final_result += 1

        my_dll.destroyDoubleArray1D(result)
    print("Result : ", final_result, "/", len(y_testing), " | ", final_result / len(y_testing) * 100, "%")
    pourcentage = final_result / len(y_testing) * 100

    # May be change in the near future
    str_file = MLP_SAVE + "MLP_" + str(final_result / len(y_testing)) + ".txt"
    str_file = str_file.encode('UTF-8')
    my_dll.saveModelMLP(MLP, str_file, len_npl, pourcentage)
    my_dll.destroyMlpModel(MLP)


def training_SVM(my_dll, x_training, y_training, x_testing, y_testing, learning_rate, threshold, limit):
    # Set X
    C_TYPE_X = ct.c_double
    POINTER_C_TYPE_X = ct.POINTER(C_TYPE_X)
    ITLARR_X = C_TYPE_X * len(x_training[0])
    PITLARR_X = POINTER_C_TYPE_X * len(x_training)
    ptr_x = PITLARR_X()
    for i in range(len(x_training)):
        ptr_x[i] = ITLARR_X()
        for j in range(len(x_training[0])):
            ptr_x[i][j] = x_training[i][j]

    # Set Y
    ITLARRY = ct.c_int32 * len(y_training)
    PITLARRY = ct.POINTER(ct.c_int32) * len(y_training[0])
    ptr_y = PITLARRY()
    for i in range(len(y_training[0])):
        ptr_y[i] = ITLARRY()
        for j in range(len(y_training)):
            ptr_y[i][j] = y_training[j][i]

    # Set var
    rowsXLen, colsXLen, rowsWLen, rowsYLen, colsYLen = set_var_svm(x_training, y_training)
    print("--- Training SVM ---")
    w = list()
    derive = list()
    final_result = 0
    for i in range(len(y_training[0])):
        # /!\ Allocated ptr list, needs to be free/destroy
        w.append(my_dll.initSVMWeight(rowsWLen))
        derive.append(my_dll.initSVMWeightDerive(rowsWLen))
        w[i] = my_dll.trainSVM(ptr_x, ptr_y[i], w[i], derive[i], colsXLen, rowsXLen, rowsYLen, rowsWLen, ct.c_double(learning_rate), ct.c_double(threshold), limit)
    for i in range(rowsYLen):
        result = list()
        verify = 0
        for iter_res in range(len(y_training[0])):
            result.append(my_dll.resultSVM(ptr_x[i], w[iter_res], rowsXLen))
        for case_num in range(len(y_training[0])):
            if (result[case_num] == y_training[i][case_num]):
                verify += 1
        if (verify == len(y_training[i])):
            final_result += 1
    print("Result : ", final_result, "/", len(y_training), " | ", final_result / len(y_training) * 100, "%")

    # Set X
    C_TYPE_X = ct.c_double
    POINTER_C_TYPE_X = ct.POINTER(C_TYPE_X)
    ITLARR_X = C_TYPE_X * len(x_testing[0])
    PITLARR_X = POINTER_C_TYPE_X * len(x_testing)
    ptr_x = PITLARR_X()
    for i in range(len(x_testing)):
        ptr_x[i] = ITLARR_X()
        for j in range(len(x_testing[0])):
            ptr_x[i][j] = x_testing[i][j]

    # Set Y
    ITLARRY = ct.c_int32 * len(y_testing)
    PITLARRY = ct.POINTER(ct.c_int32) * len(y_testing[0])
    ptr_y = PITLARRY()
    for i in range(len(y_testing[0])):
        ptr_y[i] = ITLARRY()
        for j in range(len(y_testing)):
            ptr_y[i][j] = y_testing[j][i]

    print("--- Testing SVM ---")
    rowsXLen, colsXLen, rowsWLen, rowsYLen, colsYLen = set_var_svm(x_testing, y_testing)
    verify = 0
    final_result = 0
    for i in range(rowsYLen):
        result = list()
        verify = 0
        for iter_res in range(len(y_testing[0])):
            result.append(my_dll.resultSVM(ptr_x[i], w[iter_res], rowsXLen))
        for case_num in range(len(y_testing[0])):
            if (result[case_num] == y_testing[i][case_num]):
                verify += 1
        if (verify == len(y_testing[i])):
            final_result += 1
    print("Result : ", final_result, "/", len(y_testing), " | ", final_result / len(y_testing) * 100, "%")

    # Save and destroy
    for i in range(len(w)):
        # May be change in the near future
        str_file = SVM_SAVE + "SVM_" + str(final_result / len(y_testing)) + "_" + str(i) + ".txt"
        str_file = str_file.encode('UTF-8')
        my_dll.saveSVM(w[i], str_file, rowsWLen, ct.c_double((verify / len(y_testing) * 100)))
        my_dll.freeArr(w[i])
        my_dll.freeArr(derive[i])


if __name__ == '__main__':
    print("--- Starting now ---")
    # Retrieve img
    x_training = list()
    y_training = list()
    x_testing = list()
    y_testing = list()
    add_img(x_training, y_training, EIFFEL_TOWER_TRAINING, [1, -1, -1, -1])
    add_img(x_training, y_training, TRIUMPHAL_ARC_TRAINING, [-1, 1, -1, -1])
    add_img(x_training, y_training, LOUVRE_TRAINING, [-1, -1, 1, -1])
    add_img(x_training, y_training, PANTHEON_TRAINING, [-1, -1, -1, 1])
    add_img(x_testing, y_testing, EIFFEL_TOWER_TESTING, [1, -1, -1, -1])
    add_img(x_testing, y_testing, TRIUMPHAL_ARC_TESTING, [-1, 1, -1, -1])
    add_img(x_testing, y_testing, LOUVRE_TESTING, [-1, -1, 1, -1])
    add_img(x_testing, y_testing, PANTHEON_TESTING, [-1, -1, -1, 1])
    
    x_training = np.array(x_training)
    y_training = np.array(y_training)
    x_testing = np.array(x_testing)
    y_testing = np.array(y_testing)

    # Linear
    my_dll = ct.CDLL(LINEAR_LIB)
    iter = 100000
    learning_rate = 0.01
    training_linear(my_dll, x_training, y_training, x_testing, y_testing, iter, learning_rate)
    
    # MLP
    my_dll = ct.CDLL(MLP_LIB)
    iter = 200000
    learning_rate = 0.001
    npl = [len(x_training[0]), 5, 3, 4]
    x_training = x_training.astype(float)
    x_testing = x_testing.astype(float)
    training_mlp(my_dll, x_training, y_training, x_testing, y_testing, npl, iter, learning_rate)

    # SVM
    my_dll = ct.CDLL(SVM_LIB)
    learning_rate = 0.0005
    threshold = 0.001
    limit = 100000
    training_SVM(my_dll, x_training, y_training, x_testing, y_testing, learning_rate, threshold, limit)