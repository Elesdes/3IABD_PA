import sys
import ctypes

from Utils import link_to_dll as ltd

from LinearModel import linear_model as lm
from MultiLayerPerceptron import multi_layer_perceptron as mlp
from RadialBasisFunction import radial_basis_function as rbf

PATH_LIB_LINEAR = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/Framework/LinearModel/cmake-build-debug"
DLL_NAME_LINEAR = "libLinearModel"
PATH_LIB_MLP = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/Framework/MultiLayerPerceptron/cmake-build-debug"
DLL_NAME_MLP = "libMultiLayerPerceptron"

"""def test(dll):
    print("Python {0:s} {1:03d}bit on {2:s}\n".format(" ".join(item.strip() for item in sys.version.split("\n")),
                                                      64 if sys.maxsize > 0x100000000 else 32, sys.platform))
    dll_test = ctypes.CDLL(dll)
    func_test = dll_test.test
    func_test.restype = ctypes.c_int

    res = func_test()
    print("{0:s} returned {1:d}".format(func_test.__name__, res))
    print("\nLinearModel lib successfully loaded")
    return dll_test"""

if __name__ == '__main__':
    # Linear
    # linear_test = test(ltd.load_lib(PATH_LIB_LINEAR, DLL_NAME_LINEAR, *sys.argv[1:]))
    linear_training = lm.training(ltd.load_lib(PATH_LIB_LINEAR, DLL_NAME_LINEAR, *sys.argv[1:]), 10000, 0)
    # linear_model = lm.loaded(lm.load_lib(*sys.argv[1:]))

    # MLP
    # mlp_test = mlp.test()
    # mlp_training = mlp.training(ltd.load_lib(PATH_LIB_MLP, DLL_NAME_MLP, *sys.argv[1:]))
    # mlp = mlp.loaded(lm.load_lib(*sys.argv[1:]))

    # MLP with genetic algorithm for optimal results
    # mlp_training = mlp.training(mlp.load_lib(*sys.argv[1:]), 10000, 0.01, [2, 5, 2, 1])
