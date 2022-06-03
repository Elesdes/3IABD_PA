import sys

from Framework.Utils import link_to_dll as ltd

from LinearModel import linear_model as lm
from MultiLayerPerceptron import multi_layer_perceptron as mlp

PATH_LIB_LINEAR = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/Framework/LinearModel/cmake-build-debug"
DLL_NAME_LINEAR = "libLinearModel"
PATH_LIB_MLP = "C:/Users/juanm/OneDrive/Bureau/ESGI - Projets/3IABD/Projet Annuel/Framework/MultiLayerPerceptron/cmake-build-debug"
DLL_NAME_MLP = "libMultiLayerPerceptron"

if __name__ == '__main__':
    # Linear
    # linear_test = ltd.test()
    lm.training(ltd.load_lib(PATH_LIB_LINEAR, DLL_NAME_LINEAR, *sys.argv[1:]), 10000, 0)
    # linear_model = lm.loaded(ltd.load_lib(PATH_LIB_LINEAR, DLL_NAME_LINEAR, *sys.argv[1:]))

    # MLP
    # mlp_test = mlp.test()
    # mlp.training(ltd.load_lib(PATH_LIB_MLP, DLL_NAME_MLP, *sys.argv[1:]))
    # mlp = mlp.loaded(lm.load_lib(*sys.argv[1:]))

    # MLP with genetic algorithm for optimal results
    # mlp_training = mlp.training(mlp.load_lib(*sys.argv[1:]), 10000, 0.01, [2, 5, 2, 1])
