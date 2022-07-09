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

extern "C" {
    DLLEXPORT void freeArr(double *tab){
        delete[] tab;
    }


    DLLEXPORT double *initSVMWeight(int32_t rowsWLen) {
        using std::cout;
        using std::endl;
        using std::setprecision;

        std::random_device rd;
        std::default_random_engine e;
        std::uniform_real_distribution<> dis(-1, 1);
        double *wMalloc = new double[rowsWLen];

        for (int i = 0; i < rowsWLen; i++) {
            //wMalloc[i] = double(dis(e)); A decommenter pour plus "d'accuracy"
            wMalloc[i] = 1;
        }
        wMalloc[0] = 0;

        return wMalloc;
    }

    DLLEXPORT double *initSVMWeightDerive(int32_t rowsDeriveLen) {
        double *deriveMalloc = new double[rowsDeriveLen];
        for (int i = 0; i < rowsDeriveLen; i++) {
            deriveMalloc[i] = 0;
        }

        return deriveMalloc;
    }

    DLLEXPORT double getHingeLoss(double *x, int32_t &y, double *w, int32_t rowsXLen, int32_t rowsWLen) {
        double loss = 0;
        double temp = 0;
        if (y == 1) {
            for (int i = 0; i < rowsXLen; i++) {
                temp += w[i + 1] * x[i];
            }
            temp = temp + w[0];
            loss = 1 - temp;
        } else {
            for (int i = 0; i < rowsXLen; i++) {
                temp += w[i + 1] * x[i];
            }
            temp = temp + w[0];
            loss = 1 + temp;
        }
        if (loss < 0) loss = 0;
        return loss;
    }

    // slope:a
    // intercept:b
    // derivative of w: dw
    // derivative of intercept: db

    DLLEXPORT double getSVMCost(double **x, int32_t *y, double *w, double *deriveMalloc, int32_t colsXLen, int32_t rowsXLen,
                                int32_t rowsYLen, int32_t rowsWLen) {
        // hinge loss
        double cost = 0;
        for (int i = 0; i < rowsXLen + 1; i++) {
            deriveMalloc[i] = 0;
        }
        for (int i = 0; i < rowsYLen; i++) {
            double loss = getHingeLoss(x[i], y[i], w, rowsXLen, rowsWLen);
            cost += loss;
            // when loss = 0, all derivatives are 0
            if (loss > 0) {
                //dw = marge of each points
                for (int j = 0; j < rowsXLen; j++) {
                    deriveMalloc[j + 1] += (-x[i][j] * y[i]);
                }
                deriveMalloc[0] += (-y[i]);
            }
        }
        cost /= rowsYLen;
        for (int j = 0; j < rowsXLen + 1; j++) {
            deriveMalloc[j] /= rowsYLen;
        }
        return cost;
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

    DLLEXPORT double *
    trainSVM(double **x, int32_t *y, double *w, double *deriveMalloc, int32_t colsXLen, int32_t rowsXLen, int32_t rowsYLen,
             int32_t rowsWLen, double lrate, double threshold) {
        int iter = 0;
        int isBelowThreshold = 1;
        while (true) {
            double cost = getSVMCost(x, y, w, deriveMalloc, colsXLen, rowsXLen, rowsYLen, rowsWLen);
            if (iter % 1000 == 0) {
                printf("Iter: %d cost = %lf ", iter, cost);
                for (int j = 1; j < colsXLen; j++) {
                    printf("dw%d = %lf ", j, deriveMalloc[j]);
                }
                printf("db = %lf\n", deriveMalloc[0]);
            }
            iter++;
            for (int j = 0; j < rowsXLen + 1; j++) {
                if (abs(deriveMalloc[j]) > threshold) {
                    isBelowThreshold = 0;
                }
            }
            if (isBelowThreshold) {
                printf("y = ");
                for (int j = 1; j < rowsXLen + 1; j++) {
                    printf("%.12f *x%d + ", w[j], j);
                }
                printf("%.12f\n", w[0]);
                break;
            } else {
                isBelowThreshold = 1;
            }
            for (int j = 0; j < rowsXLen + 1; j++) {
                w[j] -= lrate * deriveMalloc[j];
            }
        }
        return w;
    }

    DLLEXPORT void saveSVM(double *w, char *filePath, int32_t rowsWLen, double efficiency) {
        FILE *fp = fopen(filePath, "w");
        if (fp != NULL) {
            fputs("-- Efficiency --\n", fp);
            fprintf(fp, "%.15f\n", efficiency);
            fputs("-- W --\n", fp);
            for(int i = 0; i < rowsWLen; i++) {
                fprintf(fp, "{%.15f}\n", w[i]);
            }
            fclose(fp);
        }
    }

    DLLEXPORT double* loadSVM(char* filePath){
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
            double* w = new double[lenModel];
            while (fgets(text, 2000, fp) != NULL) {
                if ((strstr(text, sentence)) != NULL) {
                    if ((strstr(text, tempSentence)) != NULL) {
                        fscanf(fp, "%lf\n", &tempD);
                    }
                    while(fscanf(fp, "{%lf}\n", &w[i]) != EOF){
                        i++;
                    }
                }
            }
            fclose(fp);
            return w;
        }
        return nullptr;
    }

    /*
    DLLEXPORT int run(double **x, int32_t *y, int32_t colsXLen, int32_t rowsXLen, int32_t rowsYLen) {
        //double X1[] = {1, 2, 3};
        //double X2[] = {1, 3, 3};
        //double X[][] = {{1,1},{2,3},{3,3}};
        //int Y[] = {1, -1, -1};
        double **xMalloc = new double *[colsXLen];
        int *yMalloc = new int[rowsYLen];
        double *wMalloc = new double[rowsXLen + 1];
        double *deriveMalloc = new double[rowsXLen + 1];
        double lrate = 0.0005;
        double threshold = 0.001;

        for (int i = 0; i < colsXLen; i++) {
            xMalloc[i] = new double[rowsXLen];
            for (int j = 0; j < rowsXLen; j++) {
                xMalloc[i][j] = x[i][j];
            }
        }
        for (int i = 0; i < rowsYLen; i++) {
            yMalloc[i] = y[i];
        }

        wMalloc = initSVMWeight(rowsXLen + 1);
        deriveMalloc = initSVMWeightDerive(rowsXLen + 1);

        wMalloc = trainSVM(xMalloc, yMalloc, wMalloc, deriveMalloc, colsXLen, rowsXLen, rowsYLen, colsXLen + 1, lrate,
                           threshold);

        for (int i = 0; i < colsXLen; i++) {
            delete[] xMalloc[i];
        }
        delete[] xMalloc;
        delete[] yMalloc;
        delete[] wMalloc;
        delete[] deriveMalloc;
        return EXIT_SUCCESS;
    }*/

    DLLEXPORT int test() {
        return 42;
    }
}

