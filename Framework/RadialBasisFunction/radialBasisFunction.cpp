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

extern "C" {
DLLEXPORT int32_t test() {
    return 42;
}

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
    float **toArray = new float *[numRows];
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

DLLEXPORT void saveModelRBF(float **modelWeight, char *filePath, int32_t nRowsW, int32_t nColsW, float efficiency) {
    FILE *fp = fopen(filePath, "w");
    if (fp != NULL) {
        fputs("-- Efficiency --\n", fp);
        fprintf(fp, "%.15f\n", efficiency);
        fputs("-- W --\n", fp);
        for (int i = 0; i < nRowsW; i++) {
            for (int j = 0; j < nColsW; j += 1) {
                fprintf(fp, "{%.15f}\n", modelWeight[i][j]);
            }
        }
        fclose(fp);
    }
}

DLLEXPORT float **loadModelRBF(char *filePath, int32_t numDataset) {
    char *tempSentence = "-- Efficiency --\n";
    double tempD;
    int lenModel = 0;
    int i = 0;
    float tempF;
    float *w;
    FILE *fp = fopen(filePath, "r");
    //init l and d and the model itself
    if (fp != NULL) {
        char *sentence = "-- W --\n";
        char text[2000];
        while (fgets(text, 2000, fp) != NULL) {
            if (strstr(text, tempSentence) != NULL) {
                fscanf(fp, "%lf\n", &tempD);
            }
            if (strstr(text, sentence) != NULL) {
                while (fscanf(fp, "{%f}\n", &tempF) != EOF) {
                    lenModel += 1;
                }
            }
        }
        fclose(fp);
        fp = fopen(filePath, "r");
        w = new float[lenModel];
        while (fgets(text, 2000, fp) != NULL) {
            if (strstr(text, sentence) != NULL) {
                if (strstr(text, tempSentence) != NULL) {
                    fscanf(fp, "%lf\n", &tempD);
                }
                while (fscanf(fp, "{%f}\n", &w[i]) != EOF) {
                    i++;
                }
            }
        }
        fclose(fp);
    }

    MatrixXf weights;
    weights.resize(lenModel, numDataset);
    i = 0;

    do {
        for (int j = 0; j < numDataset; j += 1) {
            weights(i, j) = w[i];
            i += 1;
        }
    } while(w[i] != NULL);

    return fromEigenMatrixToPointer(weights, lenModel, numDataset);
}

DLLEXPORT void destroyFloatArray(float **array, int32_t nRows) {
    for (int i = 0; i < nRows; i++) {
        delete[] array[i];
    }
    delete[] array;
}

DLLEXPORT float **flat(float **arrayToFlat, int numRows, int numCols) {
    MatrixXf matrixToFlat = fromArrayToEigenMatrix(arrayToFlat, numRows, numCols);

    int size = matrixToFlat.rows() * matrixToFlat.cols();

    MatrixXf flatten;
    flatten.resize(1, size);

    int counter = 0;

    for (int i = 0; i < matrixToFlat.rows(); i += 1) {
        for (int j = 0; j < matrixToFlat.cols(); j += 1) {
            flatten(0, counter) = matrixToFlat(i, j);
            if (counter < size) counter += 1;
        }
    }
    return fromEigenMatrixToPointer(flatten, flatten.rows(), flatten.rows());
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

DLLEXPORT int32_t intRandomizer(int lower_bound, int upper_bound) {
    srand(time(NULL));
    return rand() % (upper_bound - lower_bound + 1) + lower_bound;
}

DLLEXPORT float floatRandomizer(float lower_bound, float upper_bound) {
    random_device dev;
    mt19937 rng(dev());
    uniform_int_distribution<mt19937::result_type> dist(lower_bound, upper_bound);

    return dist(rng);
}

DLLEXPORT float **initModelWeights(int32_t numRows, int32_t numCols) {
    MatrixXf W = MatrixXf::Random(numRows, numCols);
    return fromEigenMatrixToPointer(W, W.rows(), W.cols());
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

DLLEXPORT float **newWeights(float **arrayCenters, int32_t nRowsC, int32_t nColsC,
                             float **arrayWeights, int32_t nRowsW, int32_t nColsW,
                             int32_t gamma, int32_t mode) {
    Matrix<float, -1, -1> centers = fromArrayToEigenMatrix(arrayCenters, nRowsC, nColsC);
    Matrix<float, -1, -1> weights = fromArrayToEigenMatrix(arrayWeights, nRowsW, nColsW);
    MatrixXf newWeights;
    newWeights.resize(nRowsW, nColsW);
    for (int i = 0; i < newWeights.rows(); i += 1) {
        MatrixXf inputs = rowMatrix(centers, i);
        for(int j = 0; j < newWeights.cols(); j += 1) {
            newWeights(i, j) = predictRBFModel(weights, centers, inputs, gamma, mode);
        }
    }

    return fromEigenMatrixToPointer(newWeights, newWeights.rows(), newWeights.cols());
}

DLLEXPORT float **trainingClassification(float **arrayX, int32_t nRowsX, int32_t nColsX,
                                         float **arrayCenters, int32_t nRowsC, int32_t nColsC,
                                         float **arrayY, int32_t nRowsY, int32_t nColsY,
                                         float **arrayWeights, int32_t nRowsW, int32_t nColsW,
                                         int32_t gamma, int32_t num_iter) {
    Matrix<float, -1, -1> matrixX = fromArrayToEigenMatrix(arrayX, nRowsX, nColsX);
    Matrix<float, -1, -1> centers = fromArrayToEigenMatrix(arrayCenters, nRowsC, nColsC);
    Matrix<float, -1, -1> matrixY = fromArrayToEigenMatrix(arrayY, nRowsY, nColsY);
    Matrix<float, -1, -1> weights = fromArrayToEigenMatrix(arrayWeights, nRowsW, nColsW);
    for (int i = 0; i < num_iter; i += 1) {
        int k = intRandomizer(1, matrixX.rows() - 1);
        MatrixXf X_k = rowMatrix(matrixX, k);
        int gXk = (int) predictRBFModel(weights, centers, X_k, gamma, 1);
        int diff = matrixY(k) - gXk;
        for (int j = 0; j < weights.rows(); j += 1) {
            MatrixXf center = rowMatrix(centers, j);
            float firstCompute = pow(linalg(X_k - center), 2);
            float secondCompute = exp(-gamma * firstCompute);
            weights(j) = weights(j) + 0.01 * diff * secondCompute;
        }
    }
    return fromEigenMatrixToPointer(weights, weights.rows(), weights.cols());
}

DLLEXPORT float **trainingRegression(float **arrayX, int32_t nRowsX, int32_t nColsX,
                                     float **arrayCenters, int32_t nRowsC, int32_t nColsC,
                                     float **arrayY, int32_t nRowsY, int32_t nColsY,
                                     float **arrayWeights, int32_t nRowsW, int32_t nColsW,
                                     int32_t gamma) {
    Matrix<float, -1, -1> matrixX = fromArrayToEigenMatrix(arrayX, nRowsX, nColsX);
    Matrix<float, -1, -1> centers = fromArrayToEigenMatrix(arrayCenters, nRowsC, nColsC);
    Matrix<float, -1, -1> matrixY = fromArrayToEigenMatrix(arrayY, nRowsY, nColsY);
    Matrix<float, -1, -1> weights = fromArrayToEigenMatrix(arrayWeights, nRowsW, nColsW);

    MatrixXf res;
    res.resize(1, matrixX.rows());

    MatrixXf phi;
    phi.resize(matrixX.rows(), centers.rows());

    for (int i = 0; i < matrixX.rows(); i += 1) {
        MatrixXf inputs = rowMatrix(matrixX, i);
        for (int j = 0; j < centers.rows(); j += 1) {
            MatrixXf center = rowMatrix(centers, j);
            MatrixXf centerMinusInputs = center - inputs;
            float firstCompute = pow(linalg(centerMinusInputs), 2);
            float secondCompute = exp(-gamma * firstCompute);
            phi(i, j) = secondCompute;
        }
    }

    res = phi.inverse() * matrixY;

    return fromEigenMatrixToPointer(res, res.rows(), res.cols());
}
}



