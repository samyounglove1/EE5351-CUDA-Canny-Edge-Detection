#ifndef KERNELS_H
#define KERNELS_H

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "types.h"

#define maxReduceBlockSize 256
#define CONVERT_BLOCK_SIZE 512

// helper funcs
__global__ void maxReduction(float* arrIn, float* maxOut, unsigned int size);
__global__ void floatArrToUnsignedChar(float* inImage, unsigned char* outImage, int imgSize);


__global__ void cudaDoubleThreshold(float* f_nmsIn, float* f_threshOut,
                                    float f_lowThreshRatio, float f_highThreshRatio,
                                    U32 u32_width, U32 u32_height, float f_max);
__global__ void cudaHysteresis(float* f_threshIn, float* f_hystOut, U32 u32_width, U32 u32_height);



#endif
