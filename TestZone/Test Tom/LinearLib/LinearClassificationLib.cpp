#if WIN32
#define DLLEXPORT __declspec(dllexport)
#elif
#define DLLEXPORT
#endif
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <cstdint>
#include <iostream>
#include <random>
#include <iomanip>
#include "Eigen/Dense"

extern "C" {
    DLLEXPORT void destroyDoubleArray1D(double *array){
        delete[] array;
    }

    DLLEXPORT void destroyIntArray2D(int **array, int lenOfNPL) {
        for (int firstIter = 0; firstIter < lenOfNPL; firstIter++) {
            delete[] array[firstIter];
        }
        delete[] array;
    }

    DLLEXPORT void destroyFloatArray(float *array) {
        delete[] array;
    }

    DLLEXPORT void printFloatArray(float *array, int32_t lenI) {
        for(int i = 0; i < lenI; i++){
            printf("%d: %lf\n", i, array[i]);
        }
    }

    DLLEXPORT void printIntArray(int *array, int32_t lenI) {
        for(int i = 0; i < lenI; i++){
            printf("%d: %d\n", i, array[i]);
        }
    }

    DLLEXPORT void saveModelLinear(float* modelWeight, char* filePath, int32_t rowsWLen, double efficiency){
        FILE *fp = fopen(filePath, "w");
        if (fp != NULL) {
            fputs("-- Efficiency --\n", fp);
            fprintf(fp, "%.15f\n", efficiency);
            fputs("-- W --\n", fp);
            for(int i = 0; i < rowsWLen; i++) {
                fprintf(fp, "{%.15f}\n", modelWeight[i]);

            }
            fclose(fp);
        }
    }

    DLLEXPORT float* loadModelLinear(char* filePath){
        char *tempSentence = "-- Efficiency --\n";
        double tempD;
        int lenModel = 0;
        int i = 0;
        float tempF;
        FILE *fp = fopen(filePath, "r");
        //init l and d and the model itself
        if (fp != NULL) {
            char *sentence = "-- W --\n";
            char text[2000];
            while (fgets(text, 2000, fp) != NULL) {
                if ((strstr(text, tempSentence)) != NULL) {
                    fscanf(fp, "%lf\n", &tempD);
                }
                if ((strstr(text, sentence)) != NULL) {
                    while(fscanf(fp, "{%f}\n", &tempF) != EOF){
                        lenModel+=1;
                    }
                }
            }
            fclose(fp);
            fp = fopen(filePath, "r");
            float* w = new float[lenModel];
            while (fgets(text, 2000, fp) != NULL) {
                if ((strstr(text, sentence)) != NULL) {
                    if ((strstr(text, tempSentence)) != NULL) {
                        fscanf(fp, "%lf\n", &tempD);
                    }
                    while(fscanf(fp, "{%f}\n", &w[i]) != EOF){
                        i++;
                    }
                }
            }
            fclose(fp);
            return w;
        }
        return nullptr;
    }

    DLLEXPORT float* initModelWeights(int32_t colsWLen) {
        using std::cout;
        using std::endl;
        using std::setprecision;

        std::random_device rd;
        std::default_random_engine e;
        std::uniform_real_distribution<> dis(-1, 1);

        float *w = new float[colsWLen];
        for (int i = 0; i < colsWLen; i++) {
            w[i] = 0.0;
        }

        // Init Linear Model
        for (int n = 0; n < colsWLen; n++) {
            w[n] = float(dis(e));
        }

        return w;
    }

    /*
    DLLEXPORT float* initModelWeightsMultipleOutputs(int32_t colsXLen, int32_t rowsWLen, int32_t rowsYLen) {
        using std::cout;
        using std::endl;
        using std::setprecision;

        std::random_device rd;
        std::default_random_engine e;
        std::uniform_real_distribution<> dis(-1, 1);

        float *w = new float[rowsWLen * rowsYLen];
        for (int i = 0; i < rowsWLen * rowsYLen; i++) {
            w[i] = 0;
        }

        // Init Linear Model
        for (int n = 0; n < (colsXLen + 1) * rowsYLen; n++) {
            w[n] = float(dis(e));
        }

        return w;
    }*/
    DLLEXPORT float predictLinearModelRegressionFloat(float *modelWeights, float *inputs, int32_t rowsWLen) {
        float res = 0.0f;
        // /!\ Possible que cela engendre des soucis plus tard
        // for (int i = 0; i < (sizeof(modelWeights) / sizeof(*modelWeights)); i += 1) { à garder au cas où
        for (int i = 0; i < rowsWLen - 1; i += 1) {
            res += modelWeights[i + 1] * inputs[i];
        }
        float totalSum = 1 * modelWeights[0] + res;
        return totalSum;
    }

    DLLEXPORT int32_t predictLinearModelClassificationFloat(float *modelWeights, float *inputs, int32_t rowsWLen) {
        float res = 0.0f;
        // /!\ Possible que cela engendre des soucis plus tard
        // for (int i = 0; i < (sizeof(modelWeights) / sizeof(*modelWeights)); i += 1) { à garder au cas où
        for (int i = 0; i < rowsWLen - 1; i += 1) {
            res += modelWeights[i + 1] * inputs[i];
        }
        float totalSum = 1 * modelWeights[0] + res;
        if (totalSum >= 0) {
            return 1;
        }
        return -1;
    }

    DLLEXPORT int32_t predictLinearModelClassificationInt(float *modelWeights, int32_t *inputs, int32_t rowsWLen) {
        float res = 0.0f;
        // /!\ Possible que cela engendre des soucis plus tard
        // for (int i = 0; i < (sizeof(modelWeights) / sizeof(*modelWeights)); i += 1) { à garder au cas où
        for (int i = 0; i < rowsWLen - 1; i += 1) {
            res += modelWeights[i + 1] * inputs[i];
        }
        float totalSum = 1 * modelWeights[0] + res;
        if (totalSum >= 0) {
            return 1;
        }
        return -1;
    }

    /*
    DLLEXPORT int32_t *predictLinearModelClassificationFloatMultipleOutputs(float *modelWeights, float *inputs, int32_t rowsWLen, int32_t colsYLen) {
        float res = 0.0f;
        // /!\ Possible que cela engendre des soucis plus tard
        // for (int i = 0; i < (sizeof(modelWeights) / sizeof(*modelWeights)); i += 1) { à garder au cas où
        float *answer = new float[colsYLen];
        for (int i = 0; i < rowsWLen - 1; i += 1) {
            res += modelWeights[i + 1] * inputs[i];
        }
        float totalSum = 1 * modelWeights[0] + res;
        if (totalSum >= 0) {
            return 1;
        }
        return -1;
    }*/

    DLLEXPORT float* trainLinearFloat(float **x, int32_t *y, float *w, int32_t rowsXLen, int32_t rowsWLen, int32_t iter, float learningRate) {
        for (int i = 0; i < iter; i += 1) {
            int k = rand() % rowsXLen;
            int gxk = predictLinearModelClassificationFloat(w, x[k], rowsWLen);
            int yk = y[k];
            int diff = yk - gxk;
            w[0] = w[0] + learningRate * diff * 1;
            // W[1:] = W[1:] + 0.01 * diff * X[k]
            for (int j = 1; j < rowsWLen; j += 1) {
                w[j] = w[j] + learningRate * diff * x[k][j - 1];
            }
        }
        return w;
    }

    DLLEXPORT void trainLinearFloatRegression(float **x, float **y, float *w, int32_t rowsXLen, int32_t colsXLen, int32_t rowsYLen, int32_t colsYLen, int32_t rowsWLen) {
        Eigen::MatrixXf matrixX(colsXLen, rowsXLen+1);
        Eigen::MatrixXf matrixXTranspose;
        Eigen::MatrixXf matrixY(colsYLen, rowsYLen);
        Eigen::MatrixXf matrixW(rowsWLen, 1);
        for(int i = 0; i < colsXLen; i++){
            for(int j = 0; j < rowsXLen+1; j++){
                if(j==0){
                    matrixX(i,j) = 1.0;
                }else{
                    matrixX(i,j) = x[i][j-1];
                }
            }
        }
        for(int i = 0; i < colsYLen; i++){
            for(int j = 0; j < rowsYLen; j++){
                matrixY(i,j) = y[i][j];
            }
        }
        for(int i = 0; i < rowsWLen; i++){
            matrixW(i,0) = w[i];
        }

        matrixXTranspose = matrixX.transpose();
        matrixW = (matrixXTranspose * matrixX);
        matrixW = matrixW.completeOrthogonalDecomposition().pseudoInverse();
        matrixW = matrixW * matrixXTranspose;
        matrixW = matrixW * matrixY;
        //matrixW = ((matrixXTranspose * matrixX).completeOrthogonalDecomposition().pseudoInverse() * matrixXTranspose) * matrixY;

        for(int i = 0; i < rowsWLen; i++){
            w[i] = matrixW(i,0);
        }
    }


    DLLEXPORT float* trainLinearFloatMultipleOutputs(float **x, int32_t *y, float *w, int32_t rowsXLen, int32_t rowsWLen, int32_t iter, float learningRate) {
        for (int i = 0; i < iter; i += 1) {
            int k = rand() % rowsXLen;
            int gxk = predictLinearModelClassificationFloat(w, x[k], rowsWLen);
            float yk = (float)y[k];
            float diff =  (learningRate * (float)yk ) - (learningRate * (float)gxk);
            w[0] = w[0] + learningRate * diff * 1;
            // W[1:] = W[1:] + 0.01 * diff * X[k]
            for (int j = 1; j < rowsWLen; j += 1) {
                w[j] = w[j] + diff * x[k][j - 1];
            }
        }
        return w;
    }

    DLLEXPORT float* trainLinearInt(int32_t **x, int32_t *y, float *w, int32_t rowsXLen, int32_t rowsWLen, int32_t iter, float learningRate) {
        for (int i = 0; i < iter; i += 1) {
            int k = rand() % rowsXLen;
            int gxk = predictLinearModelClassificationInt(w, x[k], rowsWLen);
            int yk = y[k];
            int diff = yk - gxk;
            w[0] = w[0] + learningRate * diff * 1;
            // W[1:] = W[1:] + 0.01 * diff * X[k]
            for (int j = 1; j < rowsWLen; j += 1) {
                w[j] = w[j] + learningRate * diff * x[k][j - 1];
            }
        }
        return w;
    }
    DLLEXPORT int test() {
        return 42;
    }
}
/*
 * Use in case of problem
int run() {
    // Init arrays and length
    // CES VARIABLES SERONT A DONNER DANS LA LIB
    int rowsXLen = 3;
    int colsXLen = 2;
    int rowsYLen = 3;
    int rowsWLen = rowsXLen + 1;
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
        printf("%d\n", predictLinearModelClassificationInt(wMalloc, xMalloc[i], rowsWLen));
    }

    destroyIntArray2D(xMalloc, rowsXLen);
    delete[] wMalloc;
    delete[] yMalloc;
    return EXIT_SUCCESS;


}*/