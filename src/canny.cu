#include <iostream>
#include <stdio.h>
#include "../inc/kernels.cuh"


float utilGetMax(float* arr, U32 size);


float utilGetMax(float* arr, U32 size) 
{
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

    // printf("found max %f\n", res);

    return res;
}


// __constant__'s have static scope, must gaussianFilter() and gradientCalculation() must stay in this file
__constant__ unsigned char G_Filter_Kernel[GAUSS_KERNEL_SIZE*GAUSS_KERNEL_SIZE];
__constant__ float Sobel_Kernel_X[GRADIENT_KERNEL_SIZE*GRADIENT_KERNEL_SIZE];
__constant__ float Sobel_Kernel_Y[GRADIENT_KERNEL_SIZE*GRADIENT_KERNEL_SIZE];


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




__global__ void gradientCalculation(float* inImage, float* outGradient, float* outSlope, int width, int height) {
    __shared__ float ins[GRADIENT_BLOCK_SIZE][GRADIENT_BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int n = GRADIENT_KERNEL_SIZE >> 1;

    int row_o = blockIdx.y * GRADIENT_TILE_SIZE + ty; 
    int col_o = blockIdx.x * GRADIENT_TILE_SIZE + tx; 
    int row = row_o - n;
    int col = col_o - n;

    // Load input tile into shared memory
    if ((row >= 0) && (row < height) && (col >= 0) && (col < width)) {
        ins[ty][tx] = inImage[row * width + col];
    }
    else {
        ins[ty][tx] = 0.0;
    }

    __syncthreads();

    if (tx < GRADIENT_TILE_SIZE && ty < GRADIENT_TILE_SIZE) {
        
        // Calculate x and y gradients individually
        float xval = 0, yval = 0;
        for (int j = 0; j < GRADIENT_KERNEL_SIZE; j++) {
            for (int i = 0; i < GRADIENT_KERNEL_SIZE; i++) {
                xval += ins[j + ty][i + tx] * Sobel_Kernel_X[j * GRADIENT_KERNEL_SIZE + i];
                yval += ins[j + ty][i + tx] * Sobel_Kernel_Y[j * GRADIENT_KERNEL_SIZE + i];
            }
        }

        if(row_o < height && col_o < width) {
            outGradient[row_o * width + col_o] = hypotf(xval, yval);
            outSlope[row_o * width + col_o] = atan2f(yval, xval);
        }
    }
}

#define DOUBLE_THRESH_HYST_BLOCK_SIZE 32
#define DOUBLE_THRESH_HYST_TILE_SIZE 30
__global__ void doubleThreshHysteris(float* nmsIn, unsigned char* hystOut, float lowThreshRatio, float highThreshRatio, unsigned int width, unsigned int height, float max) {
    float highThresh = max*highThreshRatio;
    float lowThresh = highThresh*lowThreshRatio;
    
    //one pixel wide halo
    __shared__ float inMem[DOUBLE_THRESH_HYST_BLOCK_SIZE][DOUBLE_THRESH_HYST_BLOCK_SIZE];

    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;

    int rowO = blockIdx.y * DOUBLE_THRESH_HYST_TILE_SIZE + ty;
    int colO = blockIdx.x * DOUBLE_THRESH_HYST_TILE_SIZE + tx;
    int rowI = rowO - 1; // offset for the 1 pixel halo 
    int colI = colO - 1; // offset for the 1 pixel halo 

    bool found = false;

    if (rowI >= 0 && rowI < height && colI >= 0 && colI < width) {
        inMem[ty][tx] = nmsIn[rowI * width + colI];
        if (inMem[ty][tx] >= highThresh) {
            inMem[ty][tx] = 255;
        } else if (inMem[ty][tx] >= lowThresh) {
            inMem[ty][tx] = 25;
            found = true;
        }
    } else {
        inMem[ty][tx] = 0;
    }

    __syncthreads(); //wait for all of inmem to be set

    rowO -= 1;//account for halo offset
    colO -= 1;

    //do hysteresis only if needed
    if ((tx != 0 && ty != 0 && tx != DOUBLE_THRESH_HYST_BLOCK_SIZE - 1 && ty != DOUBLE_THRESH_HYST_BLOCK_SIZE - 1) && //halo indices
        (rowO < height && colO < width)) {//within image
        if (inMem[ty][tx] == 255) {
            hystOut[rowO * width + colO] = 255;
        } else if (found) {
            //check neighbors
            if (inMem[ty-1][tx-1] == 255 ||
                inMem[ty-1][tx] == 255 ||
                inMem[ty-1][tx+1] == 255 ||
                inMem[ty][tx-1] == 255 ||
                inMem[ty][tx+1] == 255 ||
                inMem[ty+1][tx-1] == 255 ||
                inMem[ty+1][tx] == 255 ||
                inMem[ty+1][tx+1] == 255) {
                hystOut[rowO * width + colO] = 255;
            } else {
                hystOut[rowO * width + colO] = 0;
            }
        }
    }
}



void doCudaCannyInjectStage(    unsigned char* outImage, unsigned char* inImage, 
                                double* timestamps, 
                                int width, int height, int stage, 
                                float* injection) {
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
    // cudaMemset(&dImageIn, 0, sizeof(unsigned char)*imgSize);
    // cudaMemset(&dGaussFilterOut, 0.0f, sizeof(float)*imgSize);

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

    //step 4 double thresholding
    cudaEventRecord(lStart);

    //experimental additions
    //allocations
    unsigned char* dImageOut;
    cudaMalloc(&dImageOut, sizeof(unsigned char)*imgSize);
    cudaMemset(dImageOut, 0, sizeof(unsigned char)*imgSize);
    //no stage injection for this
    float max = utilGetMax(dNmsOutput, imgSize);
    dim3 threshHystBlockSize(DOUBLE_THRESH_HYST_BLOCK_SIZE, DOUBLE_THRESH_HYST_BLOCK_SIZE);
    dim3 threshHystGridSize(ceil((float) width / DOUBLE_THRESH_HYST_TILE_SIZE), ceil((float) height / DOUBLE_THRESH_HYST_TILE_SIZE));
    doubleThreshHysteris<<<threshHystGridSize, threshHystBlockSize>>>(dNmsOutput, dImageOut, 0.05, 0.09, width, height, max);
    cudaDeviceSynchronize();

    cudaFree(dNmsOutput);



    // //allocations
    // float* dThreshOut;
    // cudaMalloc(&dThreshOut, sizeof(float)*imgSize);
    // if (stage == 3) {
    //     cudaMemcpy(dThreshOut, injection, sizeof(float)*imgSize, cudaMemcpyHostToDevice);
    // } else if (stage < 3) {
    //     float f_max = utilGetMax(dNmsOutput, width * height);
        
    //     // TODO remove
    //     // printf("cuda f_max: %f\n", f_max);

    //     dim3 dblThreshold_dimBlock(16, 16, 1);
    //     dim3 dblThreshold_dimGrid(ceil((float)width / 16), ceil((float)height / 16), 1);

    //     cudaDoubleThreshold<<<dblThreshold_dimGrid, dblThreshold_dimBlock>>>(dNmsOutput, dThreshOut, 0.05, 0.09, width, height, f_max);
    //     cudaDeviceSynchronize();
    // }

    // cudaEventRecord(lEnd);
    // cudaEventSynchronize(lEnd);
    // cudaEventElapsedTime(&time, lStart, lEnd);
    // timestamps[3] = time;//store to timestamp array

    // cudaFree(dNmsOutput);

    // // step 5 edge tracking via hysterersis
    // cudaEventRecord(lStart);

    // float* dHysteresisOut;
    // cudaMalloc(&dHysteresisOut, sizeof(float)*imgSize);
    // //further allocations and thread configs here
    // if (stage == 4) {
    //     cudaMemcpy(dHysteresisOut, dThreshOut, sizeof(float)*imgSize, cudaMemcpyDeviceToDevice);
    // } else if (stage < 4) {
    //     dim3 hysteresis_dimBlock(16, 16, 1);
    //     dim3 hysteresis_dimGrid(ceil((float)width / 16), ceil((float)height / 16), 1);
    //     cudaHysteresis<<<hysteresis_dimGrid, hysteresis_dimBlock>>>(dThreshOut, dHysteresisOut, width, height);
    //     cudaDeviceSynchronize();
    // }

    // cudaEventRecord(lEnd);
    // cudaEventSynchronize(lEnd);
    // cudaEventElapsedTime(&time, lStart, lEnd);
    // timestamps[4] = time;//store to timestamp array

    // cudaFree(dThreshOut);

    //end variable cast to unsigned char behavior
    //need this because our intermediary operations will be working with floats for greater accuracy
    // cudaEventRecord(lStart);
    // unsigned char* dImageOut;
    // cudaMalloc(&dImageOut, sizeof(unsigned char)*imgSize);
    // // cudaMemset(&dImageOut, 0, sizeof(unsigned char) * imgSize);
    // unsigned int nBlocks = ceil((float) imgSize / CONVERT_BLOCK_SIZE);
    // floatArrToUnsignedChar<<<nBlocks, CONVERT_BLOCK_SIZE>>>(dHysteresisOut, dImageOut, imgSize);
    
    // cudaEventRecord(lEnd);
    // cudaEventSynchronize(lEnd);
    // cudaEventElapsedTime(&time, lStart, lEnd);
    // timestamps[5] = time;//store to timestamp array
    
    //overall timing record

    cudaEventRecord(lEnd);
    cudaEventSynchronize(lEnd);
    cudaEventElapsedTime(&time, lStart, lEnd);
    timestamps[3] = 0;//not applicable in experimental branch
    timestamps[4] = 0;//not applicable in experimental branch
    timestamps[5] = time;//store to timestamp array

    cudaEventElapsedTime(&time, start, lEnd);
    timestamps[6] = time;
    
    cudaDeviceSynchronize();
    
    // cudaFree(dHysteresisOut);

    cudaMemcpy(outImage, dImageOut, sizeof(unsigned char)*imgSize, cudaMemcpyDeviceToHost);

    //final free operations, subject to change
    cudaFree(dImageOut);
    cudaEventDestroy(start);
    cudaEventDestroy(lStart);
    cudaEventDestroy(lEnd);
}

