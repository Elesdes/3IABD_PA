//
// Created by erwan on 03/04/2022.
//

#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include <iostream>
#include <random>
#include <iomanip>

struct MLP {
    int32_t L;
    int32_t* d;

    double*** W;
    double** X;
    double** deltas;
};
void destroy_mlp_model(struct MLP * model) {
    delete(model);
}

void destroyDoubleArray3D(double*** array, int* npl, int lenOfNPL) {
    for(int firstIter = 0; firstIter < lenOfNPL; firstIter++){
        for(int secondIter = 0; secondIter < npl[firstIter-1]+1; secondIter++){
            delete[] array[firstIter][secondIter];//deletes an inner array of integer;
        }
        delete[] array[firstIter];
    }
    delete[] array;
}

void destroyDoubleArray2D(double** array, int lenOfNPL) {
    for(int firstIter = 0; firstIter < lenOfNPL; firstIter++){
        delete[] array[firstIter];
    }
    delete[] array;
}


MLP* initiateMLP(int* npl, int lenOfD){
    MLP* mlp = new MLP();
    using std::cout;
    using std::endl;
    using std::setprecision;

    std::random_device rd;
    std::default_random_engine e;
    std::uniform_real_distribution<> dis(-1, 1);

    double*** tabW = new double**[lenOfD];
    double** tabX = new double*[lenOfD];
    double** tabDeltas = new double*[lenOfD];

    for(int i=0; i<lenOfD; i++){
        if(i==0){
            tabW[i] = new double*[0];
        }else{
            tabW[i] = new double*[npl[i-1]+1];
            for(int j=0; j<npl[i-1]+1; j++){
                tabW[i][j] = new double[npl[i]+1];
                for(int k=0; k<npl[i]+1; k++){
                    if(k==0){
                        tabW[i][j][k]=0;
                    }else{
                        tabW[i][j][k]=float(dis(e));
                    }
                }
            }
        }
    }

    for(int i = 0; i<lenOfD; i++){
        tabX[i] = new double[npl[i]+1];
        tabDeltas[i] = new double[npl[i]+1];
        for(int j = 0; j<npl[i]+1; j++){
            if(j==0){
                tabX[i][j] = 1.0;
            }else{
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

    mlp->L = lenOfD-1;
    mlp->d = npl;
    mlp->W = tabW;
    mlp->X = tabX;
    mlp->deltas = tabDeltas;


    destroyDoubleArray2D(tabX,lenOfD);
    destroyDoubleArray2D(tabDeltas, lenOfD);
    destroyDoubleArray3D(tabW,npl,lenOfD);
    return mlp;
}

double predictMLP(MLP* mlp, int* sample_inputs, int is_classification){
    double total = 0.0;
    for(int i = 0; i < mlp->d[0]; i++){
        mlp->X[0][i+1] = sample_inputs[i];
    }

    for(int i = 1; i < mlp->L + 1; i++){
        for(int j = 1; j < mlp->d[i] + 1; j++){
            total = 0.0;
            for(int k = 0; k < mlp->d[i-1] + 1; k++){
                total += mlp->W[i][k][j] * mlp->X[i - 1][k];
            }
            if(i < mlp->L || is_classification){
                total = tanh(total);
            }
            mlp->X[i][j] = total;
        }
    }
    // mlp->X[mlp->L][1:] en python donc à faire gaffe!
    return mlp->X[mlp->L][1];
}

void trainMLP(MLP* mlp, int** allSamplesInputs, int lenAllSamplesInputs, int lenOneSamplesInputs, int* allSamplesExpectedOutputs, int lenSamplesExpectedOutputs, float learningRate, int isClassification, int nbIter){
    int k;
    int sampleExpectedOutput[1];
    double semiGradient;
    double total = 0.0;
    double temp;
    for(int _ = 0; _ < nbIter; _++){
        int* sampleInputs = new int[lenOneSamplesInputs];
        k = rand()%lenAllSamplesInputs;
        sampleInputs = allSamplesInputs[k];
        sampleExpectedOutput[0] = allSamplesExpectedOutputs[k];
        temp = predictMLP(mlp, sampleInputs, isClassification);
        delete[] sampleInputs;
        for(int j = 1; j < mlp->d[mlp->L] + 1; j++){
            semiGradient = mlp->X[mlp->L][j] - sampleExpectedOutput[j - 1];
            if(isClassification){
                semiGradient = semiGradient * (pow((1 - mlp->X[mlp->L][j]), 2));
            }
            mlp->deltas[mlp->L][j] = semiGradient;
        }
        for(int L = mlp->L + 1; L >= 1 ;L--){
            for(int i = 1; mlp->d[L - 1] + 1; i++){
                total = 0.0;
                for(int j = 1; j < mlp->d[L] + 1; j++){
                    total += mlp->W[L][i][j] * mlp->deltas[L][j];
                }
                total = (exp((1 - mlp->X[L - 1][i], 2)) * total);
                mlp->deltas[L - 1][i] = total;
            }
        }
        for(int L = 1; mlp->L + 1; L++){
            for(int i = 0; mlp->d[L - 1] + 1; i++){
                for(int j = 0; mlp->d[L] + 1; j++){
                    mlp->W[L][i][j] -= learningRate * mlp->X[L - 1][i] * mlp->deltas[L][j];
                }
            }
        }
    }
}


int main() {
    srand(time(NULL));
    int x[4][2] = {{0,0},{1,0},{0,1},{1,1}};
    int y[4][1] = {{-1},{1},{1},{1}};
    int lenAllX = 4;
    int lenOneX = 2;
    int lenAllY = 4;
    int lenOneY = 1;
    int model[] = {2, 5, 2, 1};

    printf("Before : \n");
    /*
    for(int i = 0; i < 4; i++){
        for(int j = 0; j < 2; j++){
            printf("%d ", x[i][j]);
        }
    }*/
    MLP* mlp = initiateMLP(model,3); // len(model) à donner
    for(int i = 0; i < lenAllX; i++){
        printf("%lf \n", predictMLP(mlp, x[i], 1));
    }
    trainMLP(mlp, x,lenAllX, lenOneX, y, lenAllY,0.1, 1, 10000);
    for(int i = 0; i < lenAllX; i++){
        printf("%lf \n", predictMLP(mlp, x[i], 1));
    }

    destroy_mlp_model(mlp);
    /*
    std::cout<<mlp->L<<std::endl;
    std::cout<<mlp->d<<std::endl;
    std::cout<<mlp->W<<std::endl;
    std::cout<<mlp->X<<std::endl;
    std::cout<<mlp->deltas<<std::endl;
     */
    return EXIT_SUCCESS;
}