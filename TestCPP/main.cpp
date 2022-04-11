//
// Created by juanm on 20/03/2022.
//

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <iostream>
#include <random>
#include <iomanip>


void init_modelWeights(int32_t rowsXLen, int32_t colsXLen, float* w) {
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


int predict_linear_model_classification(float *modelWeights, int *inputs) {
    float res = 0.0f;
    for (int32_t i = 0; i < (sizeof(modelWeights) / sizeof(*modelWeights)); i += 1) {
        res += modelWeights[i + 1] * inputs[i];
    }
    float totalSum = 1 * modelWeights[0] + res;
    if (totalSum >= 0) {
        return 1;
    }
    return -1;
}


int run() {
    // Init arrays and length
    // CES VARIABLES SERONT A DONNER DANS LA LIB
    int32_t rowsXLen = 3;
    int32_t colsXLen = 2;
    int32_t x[3][2] = {{1, 1},
                   {2, 3},
                   {3, 3}};

    int32_t y[3] = {1, -1, -1};
    float w[3];

    int32_t iter = 10000;
    //
    // Init
    init_modelWeights(rowsXLen, colsXLen, w);
    int32_t wLen = colsXLen + 1;
    // Train /!\ Impossible de mettre sous fonction car on a une corruption de mémoire!
    for (int32_t i = 0; i < iter; i += 1) {
        int32_t k = rand() % rowsXLen;
        int32_t gxk = predict_linear_model_classification(w, x[k]);
        int32_t yk = y[k];
        int32_t diff = yk - gxk;
        w[0] = w[0] + 0.01 * diff * 1;
        // W[1:] = W[1:] + 0.01 * diff * X[k]
        for (int32_t j = 1; j < wLen; j += 1) {
            w[j] = w[j] + 0.01 * diff * x[k][j - 1];
        }
    }

    // Predict
    for (int32_t i = 0; i < rowsXLen; i += 1) {
        printf("%d\n", predict_linear_model_classification(w, x[i]));
    }
    return EXIT_SUCCESS;
}
/*
int main() {
    run();

    return EXIT_SUCCESS;
}*/