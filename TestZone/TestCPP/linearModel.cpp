//
// Created by juanm on 20/03/2022.
//

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <random>
#include <iomanip>

using std::cout;
using std::endl;
using std::setprecision;

constexpr int FLOAT_MIN = -1;
constexpr int FLOAT_MAX = 1;

struct MLP {
    int32_t L;
    int32_t *d;

    double ***W;
    double **X;
    double **deltas;
};

void init_modelWeights(int32_t rowsXLen, int32_t colsXLen, float *w) {
    using std::cout;
    using std::endl;
    using std::setprecision;

    std::random_device rd;
    std::default_random_engine e;
    std::uniform_real_distribution<> dis(-1, 1);

    // Init Linear Model
    for (int n = 0; n < colsXLen + 1; n++) {
        w[n] = float(dis(e));
    }
}

void destroy_mlp_model(struct MLP *model) {
    delete (model);
}

void destroyDoubleArray3D(double ***array, int *npl, int lenOfNPL) {
    for (int firstIter = 0; firstIter < lenOfNPL; firstIter++) {
        for (int secondIter = 0; secondIter < npl[firstIter - 1] + 1; secondIter++) {
            delete[] array[firstIter][secondIter];//deletes an inner array of integer;
        }
        delete[] array[firstIter];
    }
    delete[] array;
}

void destroyDoubleArray2D(double **array, int lenOfNPL) {
    for (int firstIter = 0; firstIter < lenOfNPL; firstIter++) {
        delete[] array[firstIter];
    }
    delete[] array;
}

MLP *initiateMLP(int *npl, int lenOfD) {
    MLP *mlp = new MLP();
    using std::cout;
    using std::endl;
    using std::setprecision;

    std::random_device rd;
    std::default_random_engine e;
    std::uniform_real_distribution<> dis(-1, 1);

    double ***tabW = new double **[lenOfD];
    double **tabX = new double *[lenOfD];
    double **tabDeltas = new double *[lenOfD];

    for (int i = 0; i < lenOfD; i++) {
        if (i == 0) {
            tabW[i] = new double *[0];
        } else {
            tabW[i] = new double *[npl[i - 1] + 1];
            for (int j = 0; j < npl[i - 1] + 1; j++) {
                tabW[i][j] = new double[npl[i] + 1];
                for (int k = 0; k < npl[i] + 1; k++) {
                    if (k == 0) {
                        tabW[i][j][k] = 0;
                    } else {
                        tabW[i][j][k] = float(dis(e));
                    }
                }
            }
        }
    }

    for (int i = 0; i < lenOfD; i++) {
        tabX[i] = new double[npl[i] + 1];
        tabDeltas[i] = new double[npl[i] + 1];
        for (int j = 0; j < npl[i] + 1; j++) {
            if (j == 0) {
                tabX[i][j] = 1.0;
            } else {
                tabX[i][j] = 0.0;
            }
            tabDeltas[i][j] = 0.0;
        }
    }

    /*
    for(int i=0; i<lenOfD; i++){
        for(int j=0; j<npl[i-1]+1; j++){
            for(int k=0; k<npl[i]+1; k++){
                printf("[%d][%d][%d] : %f\n", i, j, k, tabW[i][j][k]);
            }
        }
    }
    printf("TabX\n");
    for(int i=0; i<lenOfD; i++){
        for(int j=0; j<npl[i]+1; j++){
                printf("[%d][%d] : %f\n", i, j, tabX[i][j]);
        }
    }
    printf("TabDeltas\n");
    for(int i=0; i<lenOfD; i++){
        for(int j=0; j<npl[i]+1; j++){
            printf("[%d][%d] : %f\n", i, j, tabDeltas[i][j]);
        }
    }
    */

    mlp->L = lenOfD - 1;
    mlp->d = npl;
    mlp->W = tabW;
    mlp->X = tabX;
    mlp->deltas = tabDeltas;


    destroyDoubleArray2D(tabX, lenOfD);
    destroyDoubleArray2D(tabDeltas, lenOfD);
    destroyDoubleArray3D(tabW, npl, lenOfD);
    return mlp;
}

double predictMLP(MLP *mlp, int *sample_inputs, int is_classification) {
    double total = 0.0;
    for (int i = 0; i < mlp->d[0]; i++) {
        mlp->X[0][i + 1] = sample_inputs[i];
    }

    for (int i = 1; i < mlp->L + 1; i++) {
        for (int j = 1; j < mlp->d[i] + 1; j++) {
            total = 0.0;
            for (int k = 0; k < mlp->d[i - 1] + 1; k++) {
                total += mlp->W[i][k][j] * mlp->X[i - 1][k];
            }
            if (i < mlp->L || is_classification) {
                total = tan(total);
            }
            mlp->X[i][j] = total;
        }
    }
    // mlp->X[mlp->L][1:] en python donc à faire gaffe!
    return mlp->X[mlp->L][1];
}

int predict_linear_model_classification(float *modelWeights, int *inputs) {
    float res = 0.0f;
    // np.sum(model_weights[1:] * inputs)
    for (int32_t i = 0; i < (sizeof(modelWeights) / sizeof(*modelWeights)); i += 1) {
        res += modelWeights[i + 1] * inputs[i];
    }
    // total_sum = 1 * model_weights[0] + np.sum(model_weights[1:] * inputs)
    float totalSum = 1 * modelWeights[0] + res;
    // return 1 if total_sum >= 0 else -1
    if (totalSum >= 0) {
        return 1;
    }
    return -1;
}

float fit() {

}

int32_t radial_basis_function_network(int* x, int xLen) {
    for(int i = 0; i < xLen * 2; i += 1) {
        printf("%d\n", x[i]);
    }
}

void linear_model(int **x, int *y, int32_t xLen) {
    std::random_device rd;
    std::default_random_engine e;
    std::uniform_real_distribution<> dis(-1, 1);

    float w[3];

    int32_t wLen = (sizeof(w) / sizeof(*w));

    // Init Linear Model
    for (int32_t n = 0; n < 3; ++n) {
        w[n] = dis(e);
        printf("%f\n", w[n]);
    }

    for (int32_t i = 0; i < xLen; i += 1) {
        printf("%d\n", predict_linear_model_classification(w, x[i]));
    }

    // Training
    for (int32_t i = 0; i < 10000; i += 1) {
        int32_t k = rand() % xLen;
        int32_t gxk = predict_linear_model_classification(w, x[k]);
        int32_t yk = y[k];
        int32_t diff = yk - gxk;
        w[0] = w[0] + 0.01 * diff * 1;
        // W[1:] = W[1:] + 0.01 * diff * X[k]
        for (int32_t j = 1; j < wLen; j += 1) {
            w[j] = w[j] + 0.01 * diff * x[k][j - 1];
        }
        //printf("%f %f %f\n", w[0], w[1], w[2]);
    }

    // On cherche Y
    for (int32_t i = 0; i < xLen; i += 1) {
        printf("%d\n", predict_linear_model_classification(w, x[i]));
        printf("non");
    }
}

int main() {
    // Init for MLP
    int model[] = {2, 5, 2, 1};

    // Init arrays and length
    int arrayX[3][2] = {{1, 1},
                        {2, 3},
                        {3, 3}};
    int *x = (int *) malloc(3 * sizeof(int32_t));
    x = *arrayX;

    int arrayY[3] = {1, -1, -1};
    int *y = (int *) malloc(3 * sizeof(int32_t));
    y = arrayY;

    int32_t xLen = (sizeof(arrayX) / sizeof(*arrayX));
    //int32_t yLen = (sizeof(y) / sizeof(*y));

    /*
    // Call linear model
    printf("Linear Model :\n");
    linear_model(&x, y, xLen);
    */

    // Call RBF
    printf("RBF :\n");
    radial_basis_function_network(x, xLen);

    /*
    // Call MLP
    printf("MLP :\n");
    MLP *mlp = initiateMLP(model, 3); // len(model) à donner
    destroy_mlp_model(mlp);
    */

    // Desallocation
    free(x);
    free(y);

    return 0;
}
