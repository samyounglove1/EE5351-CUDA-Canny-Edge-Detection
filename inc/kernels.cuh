#ifndef KERNELS_H
#define KERNELS_H

#include <iostream>
#include <stdio.h>
#include "types.h"


#define GAUSS_KERNEL_SIZE 5
#define GAUSS_KERNEL_SUM 159
#define GKS_DIV_2 (GAUSS_KERNEL_SIZE >> 1)
#define GAUSS_TILE_SIZE 12
#define GAUSS_BLOCK_SIZE (GAUSS_TILE_SIZE + GAUSS_KERNEL_SIZE - 1)//yields 256 thread 2D blocks
#define GRADIENT_KERNEL_SIZE 3
#define GRADIENT_TILE_SIZE 14
#define GRADIENT_BLOCK_SIZE (GRADIENT_TILE_SIZE + GRADIENT_KERNEL_SIZE - 1)
#define GRADIENT_NORM_BLOCK_SIZE 16
#define NON_MAX_BLOCK_SIZE 16
#define maxReduceBlockSize 256
#define CONVERT_BLOCK_SIZE 512


// helper funcs
__global__ void maxReduction(float* arrIn, float* maxOut, unsigned int size);
__global__ void maxReduction_2(float* arrIn, float* maxOut, U32 size);
__global__ void floatArrToUnsignedChar(float* inImage, unsigned char* outImage, int imgSize);
__global__ void gradientNormalize(float* gradient, float maxGradient, int width, int height);


// canney edge steps
__global__ void nonMaximumSupression(float* inGradient, float* inAngle, float* outSupressedGradient, int width, int height);
__global__ void cudaDoubleThreshold(float* f_nmsIn, float* f_threshOut,
                                    float f_lowThreshRatio, float f_highThreshRatio,
                                    U32 u32_width, U32 u32_height, float f_max);
__global__ void cudaHysteresis(float* f_threshIn, float* f_hystOut, U32 u32_width, U32 u32_height);



#endif
