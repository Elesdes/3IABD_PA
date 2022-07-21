//
// Created by juanm on 03/07/2022.
//

#include <random>
#include <iostream>
#include <iomanip>
#include <string>

#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

auto asMatrix(float **array, int numRows, int numCols) {
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

auto **asArray(Matrix<float, -1, -1> matrixInput, int numRows, int numCols) {
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

auto rowMatrix(Matrix<float, -1, -1> matrixToConvert, int numRow) {
    MatrixXf row;
    row.resize(1, matrixToConvert.cols());

    for (int i = 0; i < matrixToConvert.cols(); i += 1) {
        row(0, i) = matrixToConvert(numRow, i);
    }
    return row;
}

tuple<int, int> numRowsAndCols(Matrix<float, -1, -1> matrixInput) {
    int rows = matrixInput.rows();
    int cols = matrixInput.cols();
    return {rows, cols};
}

Matrix<float, -1, -1> flat(Matrix<float, -1, -1> matrixToFlat) {
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
    return flatten;
}

float linalg(Matrix<float, -1, -1> matrixInput) {
    float total = 0.;

    for (int i = 0; i < matrixInput.rows(); i += 1) {
        for (int j = 0; j < matrixInput.cols(); j += 1) {
            total = total + pow(abs(matrixInput(i, j)), 2);
        }
    }
    total = pow(total, 0.5);
    return total;
}

int intRandomizer(int lower_bound, int upper_bound) {
    srand(time(NULL));
    return rand() % (upper_bound - lower_bound + 1) + lower_bound;
}

// Switch values in matrix
auto shuffle(Matrix<float, -1, -1> inputMatrix) {
    int randRow1 = 0;
    int randCol1 = 0;
    int randRow2 = 0;
    int randCol2 = 0;

    for (int i = 0; i < inputMatrix.rows(); i += 1) {
        do {
            randRow1 = intRandomizer(0, inputMatrix.rows() - 1);
            randCol1 = intRandomizer(0, inputMatrix.cols() - 1);
            randRow2 = intRandomizer(0, inputMatrix.rows() - 1);
            randCol2 = intRandomizer(0, inputMatrix.cols() - 1);
        } while (randRow1 == randRow2 && randCol1 == randCol2);

        float temp = inputMatrix(randRow1, randCol1);
        inputMatrix(randRow1, randCol1) = inputMatrix(randRow2, randCol2);
        inputMatrix(randRow2, randCol2) = temp;
    }
    return inputMatrix;
}

auto rbfKernel(Matrix<float, -1, -1> data1, Matrix<float, -1, -1> data2, int sigma) {
    MatrixXf delta = data1 - data2;
    delta = delta.cwiseAbs();
    float squaredEuclidean = delta.sum();
    float result = exp(-squaredEuclidean) / pow(2 * sigma, 2);
    return result;
}

auto thirdTerm(Matrix<float, -1, -1> clusterMember, int var) {
    float result = 0;
    for (int i = 0; i < clusterMember.rows(); i += 1) {
        MatrixXf I = rowMatrix(clusterMember, i);
        for (int j = 0; j < clusterMember.rows(); j += 1) {
            MatrixXf J = rowMatrix(clusterMember, j);
            result = result + rbfKernel(I, J, var);
        }
    }
    result /= pow(clusterMember.rows(), 2);
    return result;
}

float second_term(Matrix<float, -1, -1> dataI, Matrix<float, -1, -1> clusterMember, int var) {
    float result = 0;
    for (int i = 0; i < clusterMember.rows(); i += 1) {
        MatrixXf I = rowMatrix(clusterMember, i);
        result = result + rbfKernel(dataI, I, var);
    }
    result = 2 * result / clusterMember.rows();
    return result;
}

int main() {
    MatrixXf X;
    X.resize(4, 2);
    X << 1, 1,
            2, 3,
            3, 3,
            2.5, 3;
    cout << "X :\n" << X << "\n\n";

    X = shuffle(X);
    cout << "Suffled X :\n" << X << "\n";
}