#include "mtrx_utils.h"
#include <stdlib.h>
#include <stdio.h>

float** transpose(float** mtrx, int rows, int cols){
    //printf("AAA\n");

    float** transposed = (float**)malloc(rows*sizeof(float*));
    for (int k = 0; k < rows; k++){
        transposed[k] = (float*) malloc(cols*sizeof(float));
    }
    // Transpõe
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            //printf("BEFORE CURSE %d %d (%d %d)\n", i, j, rows, cols);
            //printf("%f\n", mtrx[10][0]);
            transposed[i][j] = mtrx[j][i];
        }
    }

    //printf("OKAY\n");
    return transposed;
}