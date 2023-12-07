#include <iostream>
#include <stdio.h>
#include "../inc/kernels.h"

//forward declare functions
float utilGetMax(float* arr, unsigned int size);

#define GAUSS_KERNEL_SIZE 5
#define GAUSS_KERNEL_SUM 159
#define GKS_DIV_2 (GAUSS_KERNEL_SIZE >> 1)
#define GAUSS_TILE_SIZE 12
#define GAUSS_BLOCK_SIZE (GAUSS_TILE_SIZE + GAUSS_KERNEL_SIZE - 1)//yields 256 thread 2D blocks

__constant__ unsigned char G_Filter_Kernel[GAUSS_KERNEL_SIZE*GAUSS_KERNEL_SIZE];

__global__ void gaussianFilter(unsigned char* inImage, float* outImage, int width, int height){
    __shared__ unsigned char ins[GAUSS_BLOCK_SIZE][GAUSS_BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_o = blockIdx.y * GAUSS_TILE_SIZE + ty; //output row
    int col_o = blockIdx.x * GAUSS_TILE_SIZE + tx; //output column
    int row_i = row_o - GKS_DIV_2; //input row
    int col_i = col_o - GKS_DIV_2; //input column

    //load elements into tile
    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width)
        ins[ty][tx] = inImage[row_i * width + col_i];
    else
        ins[ty][tx] = 0.0;

    //sync up thread processes
    __syncthreads();

    //find the weighted sum
    if (tx < GAUSS_TILE_SIZE && ty < GAUSS_TILE_SIZE) {
        float pVal = 0;
        for (int y = 0; y < GAUSS_KERNEL_SIZE; y++) {
            for (int x = 0; x < GAUSS_KERNEL_SIZE; x++) {
                pVal += G_Filter_Kernel[y * GAUSS_KERNEL_SIZE + x] * ins[y + ty][x + tx];
            }
        }

        //conditionally write output
        if (row_o < height && col_o < width) {
            outImage[row_o * width + col_o] = pVal / GAUSS_KERNEL_SUM;
        }
    }
}


#define GRADIENT_KERNEL_SIZE 3
#define GRADIENT_TILE_SIZE 14
#define GRADIENT_BLOCK_SIZE (GRADIENT_TILE_SIZE + GRADIENT_KERNEL_SIZE - 1)

__constant__ float Sobel_Kernel_X[GRADIENT_KERNEL_SIZE*GRADIENT_KERNEL_SIZE];
__constant__ float Sobel_Kernel_Y[GRADIENT_KERNEL_SIZE*GRADIENT_KERNEL_SIZE];

__global__ void gradientCalculation(float* inImage, float* outGradient, float* outSlope, int width, int height) {
    __shared__ float ins[GRADIENT_BLOCK_SIZE][GRADIENT_BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int n = GRADIENT_KERNEL_SIZE >> 1;

    int row = blockIdx.y * GRADIENT_TILE_SIZE + ty - n;
    int col = blockIdx.x * GRADIENT_TILE_SIZE + tx - n;

    // Load input tile into shared memory
    if ((row >= 0) && (row < height) && (col >= 0) && (col < width)) {
        ins[ty][tx] = inImage[row * width + col];
    }
    else {
        ins[ty][tx] = 0.0;
    }

    __syncthreads();

    if ((tx >= n) && (tx < GRADIENT_BLOCK_SIZE - n) && 
        (ty >= n) && (ty < GRADIENT_BLOCK_SIZE - n)) {
        
        // Calculate x and y gradients individually
        float xval = 0, yval = 0;
        for (int j = 0; j < GRADIENT_KERNEL_SIZE; j++) {
            for (int i = 0; i < GRADIENT_KERNEL_SIZE; i++) {
                xval += ins[ty - n + j][tx - n + i] * Sobel_Kernel_X[j * GRADIENT_KERNEL_SIZE + i];
                yval += ins[ty - n + j][tx - n + i] * Sobel_Kernel_Y[j * GRADIENT_KERNEL_SIZE + i];
            }
        }

        if((row < height) && (col < width)) {
            outGradient[row * width + col] = hypotf(xval, yval);
            outSlope[row * width + col] = atan2f(yval, xval);
        }
    }
}


#define GRADIENT_NORM_BLOCK_SIZE 16

__global__ void gradientNormalize(float* gradient, float maxGradient, int width, int height) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * GRADIENT_NORM_BLOCK_SIZE + ty;
    int col = blockIdx.x * GRADIENT_NORM_BLOCK_SIZE + tx;

    if (row < height && col < width) {
        gradient[row * width + col] = gradient[row * width + col] / maxGradient * 255;
    }
}


#define NON_MAX_BLOCK_SIZE 16

__global__ void nonMaximumSupression(float* inGradient, float* inAngle, float* outSupressedGradient, int width, int height) {
    // Could use shared memory here in the future, benefits would be minimal but would still be benefits.

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * NON_MAX_BLOCK_SIZE + ty;
    int col = blockIdx.x * NON_MAX_BLOCK_SIZE + tx;

    if (row < height - 1 && row > 0 && col > 0 && col < width - 1) {
        float pi = 3.141593;
        // Gradient values of pixels directly in front of and behind this thread's pixel along angle
        int q = 255, r = 255;

        // Convert from radians to degrees
        float angle = inAngle[row * width + col];
        angle *= (180 / pi);
        if (angle < 0) {
            angle += 180;
        }
        
        // Prevent edge pixels from grabbing out of bounds
        int row_plus  = (row == height) ? row : row + 1;
        int row_minus = (row == 0)      ? row : row - 1;
        int col_plus  = (col == width)  ? col : col + 1;
        int col_minus = (col == 0)      ? col : col - 1;

        // 0 degrees
        if (angle < 22.5 || angle >= 157.5) {
            q = inGradient[row * width + col_plus ];
            r = inGradient[row * width + col_minus];
        }
        // 45 degrees
        else if (angle < 67.5) {
            q = inGradient[row_plus  * width + col_minus];
            r = inGradient[row_minus * width + col_plus ];
        }
        // 90 degrees
        else if (angle < 112.5) {
            q = inGradient[row_plus  * width + col];
            r = inGradient[row_minus * width + col];
        }
        // 135 degrees
        else {
            q = inGradient[row_minus * width + col_minus];
            r = inGradient[row_plus  * width + col_plus ];
        }

        // If pixel is not the max of its two neighbors along angle, set to 0
        float gradient = inGradient[row * width + col];
        if (gradient >= q && gradient >= r) {
            outSupressedGradient[row * width + col] = gradient;
        }
        else {
            outSupressedGradient[row * width + col] = 0.0;
        }
    }
}

//room for later kernel functions, try to keep macro definitions for each function above def for said function
/**
*
*
*   PUT CODE IN THIS AREA TO KEEP THINGS ORDERED!!!
*
*
*/




void doCudaCannyInjectStage(    unsigned char* outImage, unsigned char* inImage, 
                                double* timestamps, 
                                int width, int height, int stage, 
                                float* injection) {
    //TODO timing event setup
    cudaEvent_t start, lStart, lEnd;//start will overarc the whole thing
    float time = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&lStart);
    cudaEventCreate(&lEnd); //lStart and lEnd are local start/ends

    unsigned int imgSize = width*height;
    
    cudaEventRecord(start);
    cudaEventRecord(lStart);
    //gaussian filter (maybe other parameter loading/allocation)

    unsigned char* dImageIn;
    float* dGaussFilterOut; //floats will be more accurate after weighted division
    unsigned char gKernel[GAUSS_KERNEL_SIZE*GAUSS_KERNEL_SIZE] = 
        {
        2, 4, 5, 4, 2,
        4, 9,12, 9, 4,
        5,12,15,12, 5,
        2, 4, 5, 4, 2,
        4, 9,12, 9, 4
        };//gaussian 5x5 kernel hardcoded so we don't have to compute *yet*


    cudaMalloc(&dImageIn, sizeof(unsigned char)*imgSize);
    cudaMalloc(&dGaussFilterOut, sizeof(float)*imgSize);

    if (stage == 0) {
        cudaMemcpy(dGaussFilterOut, injection, sizeof(float)*imgSize, cudaMemcpyHostToDevice);//force override
    } else if (stage < 0) { //otherwise behave as normally // if stage greater, we skip this step entirely
        cudaMemcpy(dImageIn, inImage, sizeof(unsigned char)*imgSize, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(G_Filter_Kernel, gKernel, 25*sizeof(unsigned char));
    
        //gaussian filter kernel pre-call stuff
        dim3 blockSize(GAUSS_BLOCK_SIZE, GAUSS_BLOCK_SIZE);
        dim3 gridSize(ceil((float) width / GAUSS_TILE_SIZE), ceil((float) height / GAUSS_TILE_SIZE));
        gaussianFilter<<<gridSize, blockSize>>>(dImageIn, dGaussFilterOut, width, height);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(lEnd);
    cudaEventSynchronize(lEnd);
    cudaEventElapsedTime(&time, lStart, lEnd);
    timestamps[0] = time;//store to timestamp array

    //memory efficient behavior should possibly free input device array that 
    //was used in previous kernel call while waiting for current kernel call to finish
    //i.e. free dImageIn while waiting for kernel call to find gradient intensity to finish?
    
    cudaFree(dImageIn); //free earlier step data, no longer of use

    //put next stage functions here, be sure to update later call free's and kernel func input vars

    //step 2 gradient calculation
    cudaEventRecord(start);
    cudaEventRecord(lStart);

    //allocations
    float* dEdgeGradient;
    float* dDirections;
    cudaMalloc(&dEdgeGradient, sizeof(float)*imgSize);
    cudaMalloc(&dDirections, sizeof(float)*imgSize);
    //declare sobel filters as const memory
    if (stage == 1) { //directly inject step 2 results from serial instead of executing this in parallel
        cudaMemcpy(dEdgeGradient, injection, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
    } else if (stage < 1) {
        float sobelKernelx[GRADIENT_KERNEL_SIZE*GRADIENT_KERNEL_SIZE] = {
            -1,  0,  1,
            -2,  0,  2,
            -1,  0,  1
        };
        float sobelKernely[GRADIENT_KERNEL_SIZE*GRADIENT_KERNEL_SIZE] = {
             1,  2,  1,
             0,  0,  0,
            -1, -2, -1
        };
        cudaMemcpyToSymbol(Sobel_Kernel_X, sobelKernelx, GRADIENT_KERNEL_SIZE*GRADIENT_KERNEL_SIZE*sizeof(float));
        cudaMemcpyToSymbol(Sobel_Kernel_Y, sobelKernely, GRADIENT_KERNEL_SIZE*GRADIENT_KERNEL_SIZE*sizeof(float));

        dim3 blockSizeGrad(GRADIENT_BLOCK_SIZE, GRADIENT_BLOCK_SIZE);
        dim3 gridSizeGrad(ceil((float) width / GRADIENT_TILE_SIZE), ceil((float) height / GRADIENT_TILE_SIZE));
        gradientCalculation<<<gridSizeGrad, blockSizeGrad>>>(dGaussFilterOut, dEdgeGradient, dDirections, width, height);
        cudaDeviceSynchronize();

        float maxGradient = utilGetMax(dEdgeGradient, imgSize);

        dim3 blockSizeGradNorm(GRADIENT_NORM_BLOCK_SIZE, GRADIENT_NORM_BLOCK_SIZE);
        dim3 gridSizeGradNorm(ceil((float) width / GRADIENT_NORM_BLOCK_SIZE), ceil((float) height / GRADIENT_NORM_BLOCK_SIZE));
        gradientNormalize<<<gridSizeGradNorm, blockSizeGradNorm>>>(dEdgeGradient, maxGradient, width, height);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(lEnd);
    cudaEventSynchronize(lEnd);
    cudaEventElapsedTime(&time, lStart, lEnd);
    timestamps[1] = time;//store to timestamp array

    cudaFree(dGaussFilterOut);

    //step 3 non-max suppression
    cudaEventRecord(start);
    cudaEventRecord(lStart);

    //allocations
    float* dNmsOutput;
    cudaMalloc(&dNmsOutput, sizeof(float)*imgSize);

    if (stage == 2) {
        cudaMemcpy(dNmsOutput, injection, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
    } else if (stage < 2) {
        dim3 blockSizeNonMax(NON_MAX_BLOCK_SIZE, NON_MAX_BLOCK_SIZE);
        dim3 gridSizeNonMax(ceil((float) width / NON_MAX_BLOCK_SIZE), ceil((float) height / NON_MAX_BLOCK_SIZE));
        nonMaximumSupression<<<gridSizeNonMax, blockSizeNonMax>>>(dEdgeGradient, dDirections, dNmsOutput, width, height);
        cudaDeviceSynchronize();
    }

    cudaEventRecord(lEnd);
    cudaEventSynchronize(lEnd);
    cudaEventElapsedTime(&time, lStart, lEnd);
    timestamps[2] = time;//store to timestamp array

    cudaFree(dEdgeGradient);
    cudaFree(dDirections);

    // START step 4
    //step 4 double thresholding
    cudaEventRecord(start);
    cudaEventRecord(lStart);

    //allocations
    float* dThreshOut;
    cudaMalloc(&dThreshOut, sizeof(float)*imgSize);

    if (stage == 3) {
        cudaMemcpy(dThreshOut, injection, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
    } else if (stage < 3) {
        //further allocations and thread configs here
        // dim3 dblThreshold_dimGrid(16, 16, 1);
        // dim3 dblThreshold_dimBlock(ceil((float)width / 16), ceil((float)height / 16), 1);
        // cudaDoubleThreshold<<<dblThreshold_dimGrid, dblThreshold_dimBlock>>>(dNmsOutput, dThreshOut, width, height, 0.05, 0.09);
        // cudaDeviceSynchronize();

        //placeholder to ensure framework functionality
        cudaMemcpy(dThreshOut, dNmsOutput, sizeof(float)*imgSize, cudaMemcpyDeviceToDevice);
    }

    cudaEventRecord(lEnd);
    cudaEventSynchronize(lEnd);
    cudaEventElapsedTime(&time, lStart, lEnd);
    timestamps[3] = time;//store to timestamp array

    cudaFree(dNmsOutput);
    // END step 4

    //step 5 edge tracking via hysterersis
    cudaEventRecord(start);
    cudaEventRecord(lStart);

    float* dHysteresisOut;
    cudaMalloc(&dHysteresisOut, sizeof(float)*imgSize);
    //further allocations and thread configs here

    //placeholder to ensure framework functionality
    cudaMemcpy(dHysteresisOut, dThreshOut, sizeof(float)*imgSize, cudaMemcpyDeviceToDevice);

    cudaEventRecord(lEnd);
    cudaEventSynchronize(lEnd);
    cudaEventElapsedTime(&time, lStart, lEnd);
    timestamps[4] = time;//store to timestamp array

    cudaFree(dThreshOut);

    //end variable cast to unsigned char behavior
    //need this because our intermediary operations will be working with floats for greater accuracy
    cudaEventRecord(start);
    cudaEventRecord(lStart);
    unsigned char* dImageOut;
    cudaMalloc(&dImageOut, sizeof(unsigned char)*imgSize);
    unsigned int nBlocks = ceil((float) imgSize / CONVERT_BLOCK_SIZE);
    floatArrToUnsignedChar<<<nBlocks, CONVERT_BLOCK_SIZE>>>(dHysteresisOut, dImageOut, imgSize);
    
    cudaEventRecord(lEnd);
    cudaEventSynchronize(lEnd);
    cudaEventElapsedTime(&time, lStart, lEnd);
    timestamps[5] = time;//store to timestamp array
    
    //overall timing record
    cudaEventElapsedTime(&time, start, lEnd);
    timestamps[6] = time;
    
    cudaDeviceSynchronize();
    
    cudaFree(dHysteresisOut);

    cudaMemcpy(outImage, dImageOut, sizeof(unsigned char)*imgSize, cudaMemcpyDeviceToHost);

    //final free operations, subject to change
    cudaFree(dImageOut);

    cudaEventDestroy(start);
    cudaEventDestroy(lStart);
    cudaEventDestroy(lEnd);
}





float utilGetMax(float* arr, unsigned int size) {
    //arr should already be a device array
    float* maxOut;//allocate array to hold resulting max values
    unsigned int divSize = ceil((float) size / (maxReduceBlockSize * 2));
    cudaMalloc(&maxOut, sizeof(float)*divSize);

    maxReduction<<<divSize, maxReduceBlockSize>>>(arr, maxOut, size);
    while (divSize > 1) {
        cudaDeviceSynchronize();
        unsigned int tempDivSize = divSize;
        divSize = ceil((float) divSize / (maxReduceBlockSize * 2));
        maxReduction<<<divSize, maxReduceBlockSize>>>(maxOut, maxOut, tempDivSize);
    }
    cudaDeviceSynchronize();
    float res = 0;
    cudaMemcpy(&res, maxOut, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(maxOut);
    
    return res;
}