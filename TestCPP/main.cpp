//
// Created by juanm on 20/03/2022.
//

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <iostream>
#include <random>
#include <iomanip>

void destroyIntArray2D(int** array, int lenOfNPL) {
    for(int firstIter = 0; firstIter < lenOfNPL; firstIter++){
        delete[] array[firstIter];
    }
    delete[] array;
}


void initModelWeights(int colsXLen, float* w) {
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

int predictLinearModelClassification(float *modelWeights, int *inputs, int rowsWLen) {
    float res = 0.0f;
    // /!\ Possible que cela engendre des soucis plus tard
    // for (int i = 0; i < (sizeof(modelWeights) / sizeof(*modelWeights)); i += 1) { à garder au cas où
    for (int i = 0; i < rowsWLen-1; i += 1) {
        res += modelWeights[i + 1] * inputs[i];
    }
    float totalSum = 1 * modelWeights[0] + res;
    if (totalSum >= 0) {
        return 1;
    }
    return -1;
}


void trainLinear(int** x, int* y, float* w, int rowsXLen, int rowsWLen, int iter){
    for (int i = 0; i < iter; i += 1) {
        int k = rand() % rowsXLen;
        int gxk = predictLinearModelClassification(w, x[k], rowsWLen);
        int yk = y[k];
        int diff = yk - gxk;
        w[0] = w[0] + 0.01 * diff * 1;
        // W[1:] = W[1:] + 0.01 * diff * X[k]
        for (int j = 1; j < rowsWLen; j += 1) {
            w[j] = w[j] + 0.01 * diff * x[k][j - 1];
        }
    }
}


int run() {
    // Init arrays and length
    // CES VARIABLES SERONT A DONNER DANS LA LIB
    int rowsXLen = 3;
    int colsXLen = 2;
    int rowsYLen = 3;
    int rowsWLen = colsXLen + 1;
    int iter = 10000;
    int x[3][2] = {{1, 1},
                   {2, 3},
                   {3, 3}};
    int y[3][1] = {{1},
                   {-1},
                   {-1}};
    float w[3];

    int **xMalloc = new int *[rowsXLen];
    int *yMalloc = new int[rowsYLen];
    float *wMalloc = new float[rowsWLen];

    for (int i = 0; i < rowsXLen; i++) {
        xMalloc[i] = new int[colsXLen];
    }

    for (int i = 0; i < rowsXLen; i++) {
        for (int j = 0; j < colsXLen; j++) {
            xMalloc[i][j] = x[i][j];
        }
    }
    for (int i = 0; i < rowsYLen; i++) {
        yMalloc[i] = y[i][0];
    }
    for (int i = 0; i < rowsWLen; i++) {
        wMalloc[i] = w[i];
    }


    // Init
    initModelWeights(colsXLen, wMalloc);
    // Train
    trainLinear(xMalloc,yMalloc,wMalloc,rowsXLen,rowsWLen,iter);
    // Predict
    for (int i = 0; i < rowsXLen; i += 1) {
        printf("%d\n", predictLinearModelClassification(wMalloc, xMalloc[i], rowsWLen));
    }

    destroyIntArray2D(xMalloc, rowsXLen);
    delete[] wMalloc;
    delete[] yMalloc;
    return EXIT_SUCCESS;
}

int main() {
    run();

    return EXIT_SUCCESS;
}