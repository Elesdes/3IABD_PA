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

void destroyIntArray2D(int** array, int lenOfNPL) {
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
    int* tabNPL = new int[lenOfD];

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
                        //dis(e)
                        tabW[i][j][k]=double(dis(e));
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

    for(int i = 0; i<lenOfD; i++){
        tabNPL[i] = npl[i];
    }

    /*
    printf("TabW\n");
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
    }*/



    mlp->L = lenOfD-1;
    mlp->d = tabNPL;
    mlp->W = tabW;
    mlp->X = tabX;
    mlp->deltas = tabDeltas;

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
    int* sampleInputs = new int[lenOneSamplesInputs];
    for(int _ = 0; _ < nbIter; _++){
        k = rand()%lenAllSamplesInputs;
        for(int iter = 0; iter < lenOneSamplesInputs; iter++){
            sampleInputs[iter] = allSamplesInputs[k][iter];
        }
        sampleExpectedOutput[0] = allSamplesExpectedOutputs[k];
        predictMLP(mlp, sampleInputs, isClassification);
        for(int j = 1; j < mlp->d[mlp->L] + 1; j++){
            semiGradient = mlp->X[mlp->L][j] - sampleExpectedOutput[j - 1];
            if(isClassification){
                semiGradient = semiGradient * (1.0 - (pow((mlp->X[mlp->L][j]), 2)));
            }
            mlp->deltas[mlp->L][j] = semiGradient;
        }
        for(int L = mlp->L; L > 0;L--){
            for(int i = 1; i < mlp->d[L - 1] + 1; i++){
                total = 0.0;
                for(int j = 1; j < mlp->d[L] + 1; j++){
                    total += mlp->W[L][i][j] * mlp->deltas[L][j];
                }
                total = (1 - pow((mlp->X[L - 1][i]), 2)) * total;
                mlp->deltas[L - 1][i] = total;
            }
        }
        for(int L = 1; L < mlp->L + 1; L++){
            for(int i = 0; i < mlp->d[L - 1] + 1; i++){
                for(int j = 0; j < mlp->d[L] + 1; j++){
                    mlp->W[L][i][j] -= learningRate * mlp->X[L - 1][i] * mlp->deltas[L][j];
                }
            }
        }
    }
    delete[] sampleInputs;
}


void saveModelMLP(MLP* mlp, char* filePath, int lenModel){
    FILE *fp = fopen(filePath, "w");
    if (fp != NULL) {
        // L here
        fputs("-- L --\n", fp);
        fprintf(fp, "%d\n",mlp->L);
        // d here
        fputs("-- d --\n", fp);
        for(int i = 0; i < lenModel; i++){
            fprintf(fp, "{%d}\n",mlp->d[i]);
        }
        // W here
        fputs("-- W --\n", fp);
        for(int i = 1; i < mlp->L+1; i++){
            for(int j = 0; j < mlp->d[i-1]+1; j++){
                for(int k = 0; k < mlp->d[i]+1; k++){
                    fprintf(fp, "{%lf}\n", mlp->W[i][j][k]);
                }
            }
        }
        // X here
        fputs("-- X --\n", fp);
        for(int i = 0; i<mlp->L+1; i++){
            for(int j = 0; j<mlp->d[i]+1; j++){
                fprintf(fp, "{%lf}\n", mlp->X[i][j]);
            }
        }
        // deltas here
        fputs("-- deltas --\n", fp);
        for(int i = 0; i<mlp->L+1; i++){
            for(int j = 0; j<mlp->d[i]+1; j++){
                fprintf(fp, "{%lf}\n", mlp->deltas[i][j]);
            }
        }
        fclose(fp);
    }
}

MLP *loadModelMLP(char* filePath){
    int lenModel;
    int* model;
    double temp;
    FILE *fp = fopen(filePath, "r");
    //init l and d and the model itself
    if (fp != NULL) {
        char *sentence = "-- L --\n";
        char *sentence2 = "-- d --\n";
        char text[2000];
        while (fgets(text, 2000, fp) != NULL) {
            if ((strstr(text, sentence)) != NULL) {
                fscanf(fp, "%d\n", &lenModel);
                lenModel+=1;
            }
            if ((strstr(text, sentence2)) != NULL) {
                model = new int[lenModel];
                for(int i = 0; i < lenModel; i++){
                    fscanf(fp, "{%d}\n", &model[i]);
                }
            }

        }
        MLP* mlp = initiateMLP(model, lenModel);

        // set W
        fseek(fp,0,SEEK_SET);
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
        fseek(fp,0,SEEK_SET);
        sentence = "-- X --\n";
        while (fgets(text, 2000, fp) != NULL) {
            if ((strstr(text, sentence)) != NULL) {
                for(int i = 0; i<mlp->L + 1; i++){
                    for(int j = 0; j<mlp->d[i]+1; j++){
                        fscanf(fp, "{%lf}\n", &mlp->X[i][j]);
                    }
                }
            }
        }

        // set deltas
        fseek(fp,0, SEEK_SET);
        sentence = "-- deltas --\n";
        while (fgets(text, 2000, fp) != NULL) {
            if ((strstr(text, sentence)) != NULL) {
                for(int i = 0; i<mlp->L + 1; i++){
                    for(int j = 0; j<mlp->d[i]+1; j++){
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


void trainAndSaveMLP(int** x, int** y, int* model, int lenAllX, int lenOneX, int lenAllY, int lenOneY, int lenModel, int isClassification){
    srand(time(NULL));
    char *filePath = "../save/save.txt";
    int **xMalloc = new int *[lenAllX];
    int *yMalloc = new int[lenAllY];

    for (int i = 0; i < lenAllX; i++) {
        xMalloc[i] = new int[lenOneX];
    }

    for (int i = 0; i < lenAllX; i++) {
        for (int j = 0; j < lenOneX; j++) {
            xMalloc[i][j] = x[i][j];
        }
    }
    for (int i = 0; i < lenAllY; i++) {
        yMalloc[i] = y[i][0];
    }

    MLP *mlp = initiateMLP(model, lenModel); // len(model) à donner
    for (int i = 0; i < lenAllX; i++) {
        predictMLP(mlp, xMalloc[i], isClassification);
    }
    trainMLP(mlp, xMalloc, lenAllX, lenOneX, yMalloc, lenAllY, 0.01, 1, 10000);
    for (int i = 0; i < lenAllX; i++) {
        predictMLP(mlp, xMalloc[i], 1);
    }

    saveModelMLP(mlp, filePath, lenModel);

    destroyDoubleArray2D(mlp->X, mlp->L);
    destroyDoubleArray2D(mlp->deltas, mlp->L + 1);
    destroyDoubleArray3D(mlp->W, model, mlp->L + 1);
    delete[] mlp->d;
    destroyIntArray2D(xMalloc, lenAllX);
    delete[] yMalloc;
    destroy_mlp_model(mlp);
}

double useLoadedMLP(int* x, char* filePath, int isClassification){
    srand(time(NULL));
    double answer;

    MLP* mlp = loadModelMLP(filePath);
    answer = predictMLP(mlp, x, isClassification);

    destroyDoubleArray2D(mlp->deltas, mlp->L + 1);
    destroyDoubleArray3D(mlp->W, mlp->d, mlp->L + 1);
    destroyDoubleArray2D(mlp->X, mlp->L);
    delete[] mlp->d;
    destroy_mlp_model(mlp);
    return answer;
}

int main() {
    srand(time(NULL));
    int x[4][2] = {{0, 0},
                   {1, 0},
                   {0, 1},
                   {1, 1}};
    int y[4][1] = {{-1},
                   {1},
                   {1},
                   {1}};
    int lenAllX = 4;
    int lenOneX = 2;
    int lenAllY = 4;
    int lenOneY = 1;
    int model[] = {2, 5, 2, 1};
    int lenModel = 4;

    int **xMalloc = new int *[lenAllX];
    int *yMalloc = new int[lenAllY];

    for (int i = 0; i < lenAllX; i++) {
        xMalloc[i] = new int[lenOneX];
    }

    for (int i = 0; i < lenAllX; i++) {
        for (int j = 0; j < lenOneX; j++) {
            xMalloc[i][j] = x[i][j];
        }
    }
    for (int i = 0; i < lenAllY; i++) {
        yMalloc[i] = y[i][0];
    }

    MLP *mlp = initiateMLP(model, lenModel); // len(model) à donner
    //MLP* mlp = loadModelMLP("../save/save.txt");


    printf("Before : \n");
    for (int i = 0; i < lenAllX; i++) {
        printf("%f \n", predictMLP(mlp, xMalloc[i], 1));
    }
    trainMLP(mlp, xMalloc, lenAllX, lenOneX, yMalloc, lenAllY, 0.01, 1, 10000);
    printf("After training: \n");
    for (int i = 0; i < lenAllX; i++) {
        printf("%f \n", predictMLP(mlp, xMalloc[i], 1));
    }
    char *filePath = "../save/save.txt";
    saveModelMLP(mlp, filePath, lenModel);
    /*
    printf("MLP:\nself.d : \n");
    for(int i = 0; i < lenModel; i++){
        printf("%d ",mlp->d[i]);
    }
    printf("\nself.l : \n%d\nself.w : \n",mlp->L);
    for(int i = 0; i < lenModel; i++){
        for(int j = 0; j < model[i-1]+1; j++){
            for(int k = 0; k < model[i]+1; k++){
                printf("[%d][%d][%d] : %f\n", i, j, k, mlp->W[i][j][k]);
            }
        }
    }
    printf("self.x : \n");
    for(int i=0; i<lenModel; i++){
        for(int j=0; j<model[i]+1; j++){
            printf("[%d][%d] : %f\n", i, j, mlp->X[i][j]);
        }
    }
    printf("self.deltas : \n");
    for(int i=0; i<lenModel; i++){
        for(int j=0; j<model[i]+1; j++){
            printf("[%d][%d] : %f\n", i, j, mlp->deltas[i][j]);
        }
    }
     */

    destroyDoubleArray2D(mlp->X, mlp->L);
    destroyDoubleArray2D(mlp->deltas, mlp->L + 1);
    destroyDoubleArray3D(mlp->W, model, mlp->L + 1);
    delete[] mlp->d;


    destroyIntArray2D(xMalloc, lenAllX);
    delete[] yMalloc;
    destroy_mlp_model(mlp);
    return EXIT_SUCCESS;
}