//
// Created by juanm on 20/03/2022.
//

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <iostream>
#include <random>
#include <iomanip>

using std::cout;
using std::endl;
using std::setprecision;

constexpr int FLOAT_MIN = -1;
constexpr int FLOAT_MAX = 1;


int predict_linear_model_classification(float* modelWeights, int* inputs) {
    float res = 0.0f;
    // np.sum(model_weights[1:] * inputs)
    for(int i = 0; i < (sizeof(modelWeights)/sizeof(*modelWeights)); i += 1) {
        res += modelWeights[i + 1] * inputs[i];
    }
    // total_sum = 1 * model_weights[0] + np.sum(model_weights[1:] * inputs)
    float totalSum = 1 * modelWeights[0] + res;
    // return 1 if total_sum >= 0 else -1
    if(totalSum >= 0) {
        return 1;
    }
    return -1;
}

int main() {
    std::random_device rd;
    std::default_random_engine e;
    std::uniform_real_distribution<> dis(-1, 1);

    // Init arrays and length
    int x[3][2] = { {1, 1},
                    {2, 3},
                    {3, 3} };

    int y[3] = {1, -1, -1};

    float w[3];

    int xLen = (sizeof(x)/sizeof(*x));
    //yLen = (sizeof(y)/sizeof(*y));
    int wLen = (sizeof(w)/sizeof(*w));


    // Init Linear Model
    for (int n = 0; n < 3; ++n) {
        w[n] = dis(e);
        printf("%f\n", w[n]);
    }

    for(int i = 0; i < xLen; i += 1) {
        printf("%d\n", predict_linear_model_classification(w, x[i]));
    }

    // Training
    for(int i = 0; i < 10000; i += 1) {
        int k = rand() % xLen;
        int gxk = predict_linear_model_classification(w, x[k]);
        int yk = y[k];
        int diff = yk - gxk;
        w[0] = w[0] + 0.01 * diff * 1;
        // W[1:] = W[1:] + 0.01 * diff * X[k]
        for(int j = 1; j < wLen; j += 1) {
            w[j] = w[j] + 0.01 * diff * x[k][j - 1];
        }
        //printf("%f %f %f\n", w[0], w[1], w[2]);
    }

    // On cherche Y
    for(int i = 0; i < xLen; i += 1) {
        printf("%d\n", predict_linear_model_classification(w, x[i]));
    }

    return 0;
}
