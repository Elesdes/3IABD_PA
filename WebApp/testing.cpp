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

using namespace std;
using namespace Eigen;

struct MLP {
    int32_t L;
    int32_t *d;

    double ***W;
    double **X;
    double **deltas;
};

extern "C" {
// RBF
DLLEXPORT Matrix<float, -1, -1> fromArrayToEigenMatrix(float **array, int numRows, int numCols) {
    MatrixXf toMatrix;
    toMatrix.resize(numRows, numCols);
    int i = 0;
    do {
        for (int j = 0; j < numCols; j += 1) {
            toMatrix(i, j) = array[i][j];
        }
        i += 1;
    } while (i < numRows);
    return toMatrix;
}

DLLEXPORT float **fromEigenMatrixToPointer(Matrix<float, -1, -1> matrixInput, int numRows, int numCols) {
    auto **toArray = new float *[numRows];
    int i = 0;
    do {
        toArray[i] = new float[numCols];
        for (int j = 0; j < numCols; j += 1) {
            toArray[i][j] = matrixInput(i, j);
        }
        i += 1;
    } while (i < numRows);
    return toArray;
}

DLLEXPORT Matrix<float, -1, -1> rowMatrix(Matrix<float, -1, -1> matrixToConvert, int numRow) {
    MatrixXf row;
    row.resize(1, matrixToConvert.cols());

    for (int i = 0; i < matrixToConvert.cols(); i += 1) {
        row(0, i) = matrixToConvert(numRow, i);
    }
    return row;
}

DLLEXPORT float linalg(Matrix<float, -1, -1> matrixInput) {
    float total = 0.;

    for (int i = 0; i < matrixInput.rows(); i += 1) {
        for (int j = 0; j < matrixInput.cols(); j += 1) {
            total += pow(abs(matrixInput(i, j)), 2);
        }
    }
    total = pow(total, 0.5);
    return total;
}

DLLEXPORT float predictRBFModel(Matrix<float, -1, -1> modelWeights, Matrix<float, -1, -1> modelCenters,
                                Matrix<float, -1, -1> inputs, int32_t modelGamma, int32_t mode) {
    float totalSum = 0;

    for (int i = 0; i < modelCenters.rows(); i += 1) {
        MatrixXf center = rowMatrix(modelCenters, i);
        MatrixXf centerMinusInputs = center - inputs;
        float firstCompute = pow(linalg(centerMinusInputs), 2);
        float secondCompute = exp(-modelGamma * firstCompute);
        totalSum += secondCompute * modelWeights(i);
    }

    switch (mode) {
        case 1:
            if (totalSum >= 0) return 1;
            else return -1;
        case 2:
            return totalSum;
        default:
            cout << "Error";
            break;
    }
}

DLLEXPORT float **newRBFWeights(float **arrayWeights, int32_t nRowsW, int32_t nColsW,
                                float **arrayCenters, int32_t nRowsC, int32_t nColsC,
                                int32_t gamma, int32_t mode) {
    Matrix<float, -1, -1> centers = fromArrayToEigenMatrix(arrayCenters, nRowsC, nColsC);
    Matrix<float, -1, -1> weights = fromArrayToEigenMatrix(arrayWeights, nRowsW, nColsW);

    for (int i = 0; i < weights.rows(); i += 1) {
        MatrixXf inputs = rowMatrix(centers, i);
        weights(i, 0) = predictRBFModel(weights, centers, inputs, gamma, mode);
    }

    return fromEigenMatrixToPointer(weights, weights.rows(), weights.cols());
}

// Linear
DLLEXPORT float *loadModelLinear(char *filePath) {
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
                while (fscanf(fp, "{%f}\n", &tempF) != EOF) {
                    lenModel += 1;
                }
            }
        }
        fclose(fp);
        fp = fopen(filePath, "r");
        float *w = new float[lenModel];
        while (fgets(text, 2000, fp) != NULL) {
            if ((strstr(text, sentence)) != NULL) {
                if ((strstr(text, tempSentence)) != NULL) {
                    fscanf(fp, "%lf\n", &tempD);
                }
                while (fscanf(fp, "{%f}\n", &w[i]) != EOF) {
                    i++;
                }
            }
        }
        fclose(fp);
        return w;
    }
    return nullptr;
}

DLLEXPORT int32_t predictLinearModelClassificationFloat(float *modelWeights, float *inputs, int32_t rowsWLen) {
    float res = 0.0f;
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
    for (int i = 0; i < rowsWLen - 1; i += 1) {
        res += modelWeights[i + 1] * inputs[i];
    }
    float totalSum = 1 * modelWeights[0] + res;
    if (totalSum >= 0) {
        return 1;
    }
    return -1;
}

// MLP
DLLEXPORT double predictMLPInt(MLP *mlp, int32_t *sample_inputs, int32_t is_classification) {
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
                total = tanh(total);
            }
            mlp->X[i][j] = total;
        }
    }
    return mlp->X[mlp->L][1];
}

DLLEXPORT double predictMLPFloat(MLP *mlp, float *sample_inputs, int32_t is_classification) {
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
                total = tanh(total);
            }
            mlp->X[i][j] = total;
        }
    }
    return mlp->X[mlp->L][1];
}


DLLEXPORT double *predictMLPFloatMultipleOutputs(MLP *mlp, float *sample_inputs, int32_t is_classification,
                                                 int32_t lenOneSamplesExpectedOutputs) {
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
                total = tanh(total);
            }
            mlp->X[i][j] = total;
        }
    }
    double *answer = new double[lenOneSamplesExpectedOutputs];
    for (int j = 0; j < lenOneSamplesExpectedOutputs; j++) {
        answer[j] = mlp->X[mlp->L][j + 1];
    }
    return answer;
}

// SVM
DLLEXPORT double *loadSVM(char *filePath) {
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
                while (fscanf(fp, "{%f}\n", &tempF) != EOF) {
                    lenModel += 1;
                }
            }
        }
        fclose(fp);
        fp = fopen(filePath, "r");
        double *w = new double[lenModel];
        while (fgets(text, 2000, fp) != NULL) {
            if ((strstr(text, sentence)) != NULL) {
                if ((strstr(text, tempSentence)) != NULL) {
                    fscanf(fp, "%lf\n", &tempD);
                }
                while (fscanf(fp, "{%lf}\n", &w[i]) != EOF) {
                    i++;
                }
            }
        }
        fclose(fp);
        return w;
    }
    return nullptr;
}

DLLEXPORT int32_t resultSVM(double *x, double *w, int32_t rowsXLen) {
    double result = w[0];
    for (int i = 0; i < rowsXLen; i++) {
        result += w[i + 1] * x[i];
    }
    if (result < 0) {
        return -1;
    }
    return 1;
}
}

