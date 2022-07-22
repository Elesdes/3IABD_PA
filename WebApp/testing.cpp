#if WIN32
#define DLLEXPORT __declspec(dllexport)
#elif
#define DLLEXPORT
#endif

#include <stdlib.h>
#include <stdio.h>

#include <cstdint>
#include <iostream>
#include <random>
#include <iomanip>

#if defined __GNUC__ || defined __APPLE__
#include <Eigen/Dense>
#else
#include <eigen3/Eigen/Dense>
#endif


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
    } while (w[i] != NULL);

    return fromEigenMatrixToPointer(weights, lenModel, numDataset);
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
    MatrixXf newWeights;
    newWeights.resize(nRowsW, nColsW);
    for (int i = 0; i < newWeights.rows(); i += 1) {
        MatrixXf inputs = rowMatrix(centers, i);
        for (int j = 0; j < newWeights.cols(); j += 1) {
            newWeights(i, j) = predictRBFModel(weights, centers, inputs, gamma, mode);
        }
    }

    return fromEigenMatrixToPointer(newWeights, newWeights.rows(), newWeights.cols());
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

DLLEXPORT void destroyFloatArray(float *array) {
    delete[] array;
}

// MLP
DLLEXPORT void destroyDoubleArray3D(double ***array, int32_t *npl, int32_t lenOfNPL) {
    for (int firstIter = 0; firstIter < lenOfNPL; firstIter++) {
        for (int secondIter = 0; secondIter < npl[firstIter - 1] + 1; secondIter++) {
            delete[] array[firstIter][secondIter];//deletes an inner array of integer;
        }
        delete[] array[firstIter];
    }
    delete[] array;
}

DLLEXPORT void destroyDoubleArray2D(double **array, int32_t lenOfNPL) {
    for (int firstIter = 0; firstIter < lenOfNPL; firstIter++) {
        delete[] array[firstIter];
    }
    delete[] array;
}

DLLEXPORT void destroyMlpModel(struct MLP *model) {
    printf("OUi");
    destroyDoubleArray2D(model->X, model->L);
    printf("Non");
    destroyDoubleArray2D(model->deltas, model->L + 1);
    printf("PA");
    destroyDoubleArray3D(model->W, model->d, model->L + 1);
    printf("PE");
    delete[] model->d;
    printf("EUH");
    delete (model);
    printf("FUCK");
}

DLLEXPORT void destroyDoubleArray1D(double *array){
    delete[] array;
}

DLLEXPORT MLP *initiateMLP(int32_t *npl, int32_t lenOfD) {
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
    int *tabNPL = new int[lenOfD];

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
                        //dis(e)
                        tabW[i][j][k] = double(dis(e));
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

    for (int i = 0; i < lenOfD; i++) {
        tabNPL[i] = npl[i];
    }

    mlp->L = lenOfD - 1;
    mlp->d = tabNPL;
    mlp->W = tabW;
    mlp->X = tabX;
    mlp->deltas = tabDeltas;

    return mlp;
}

DLLEXPORT MLP *loadModelMLP(char *filePath) {
    int lenModel;
    int *model;
    double temp;
    FILE *fp = fopen(filePath, "r");
    //init l and d and the model itself
    if (fp != NULL) {
        char *tempSentence = "-- Efficiency --\n";
        char *sentence = "-- L --\n";
        char *sentence2 = "-- d --\n";
        double temp;
        char text[2000];
        while (fgets(text, 2000, fp) != NULL) {
            if ((strstr(text, tempSentence)) != NULL) {
                fscanf(fp, "%lf\n", &temp);
            }
            if ((strstr(text, sentence)) != NULL) {
                fscanf(fp, "%d\n", &lenModel);
                lenModel += 1;
            }
            if ((strstr(text, sentence2)) != NULL) {
                model = new int[lenModel];
                for (int i = 0; i < lenModel; i++) {
                    fscanf(fp, "{%d}\n", &model[i]);
                }
            }

        }
        MLP *mlp = initiateMLP(model, lenModel);

        // set W
        fseek(fp, 0, SEEK_SET);
        sentence = "-- W --\n";
        while (fgets(text, 2000, fp) != NULL) {
            if ((strstr(text, sentence)) != NULL) {
                for (int i = 0; i < mlp->L + 1; i++) {
                    for (int j = 0; j < mlp->d[i - 1] + 1; j++) {
                        for (int k = 0; k < mlp->d[i] + 1; k++) {
                            fscanf(fp, "{%lf}\n", &mlp->W[i][j][k]);
                        }
                    }
                }
            }
        }
        // set X
        fseek(fp, 0, SEEK_SET);
        sentence = "-- X --\n";
        while (fgets(text, 2000, fp) != NULL) {
            if ((strstr(text, sentence)) != NULL) {
                for (int i = 0; i < mlp->L + 1; i++) {
                    for (int j = 0; j < mlp->d[i] + 1; j++) {
                        fscanf(fp, "{%lf}\n", &mlp->X[i][j]);
                    }
                }
            }
        }

        // set deltas
        fseek(fp, 0, SEEK_SET);
        sentence = "-- deltas --\n";
        while (fgets(text, 2000, fp) != NULL) {
            if ((strstr(text, sentence)) != NULL) {
                for (int i = 0; i < mlp->L + 1; i++) {
                    for (int j = 0; j < mlp->d[i] + 1; j++) {
                        fscanf(fp, "{%lf}\n", &mlp->deltas[i][j]);
                    }
                }
            }
        }
        fclose(fp);
        return mlp;
    }
    int modelInit[] = {2, 5, 2, 1};
    return initiateMLP(modelInit, 4);
}

DLLEXPORT double readArray(double *arr, int32_t i){
    return arr[i];
}

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
DLLEXPORT void freeArr(double *tab){
    delete[] tab;
}

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

