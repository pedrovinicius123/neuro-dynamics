#include "mtrx_utils.h"
#include <stdlib.h>
#include <stdio.h>

float** transpose(float** mtrx, int rows, int cols){
    //printf("AAA\n");

    float** transposed = (float**)malloc((size_t)cols * sizeof(float*));
    for (int k = 0; k < cols; k++){
        transposed[k] = (float*) malloc((size_t)rows * sizeof(float));
    }
    // Transpõe
    for (int i = 0; i < cols; i++){
        for (int j = 0; j < rows; j++){
            transposed[i][j] = mtrx[j][i];
        }
    }

    //printf("OKAY\n");
    return transposed;
}