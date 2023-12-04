#include <iostream>
#include <stdio.h>

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


//room for later kernel functions, try to keep macro definitions for each function above def for said function
/**
*
*
*   PUT CODE IN THIS AREA TO KEEP THINGS ORDERED!!!
*
*
*/

//may need to optimize dimensioning for this operation
#define CONVERT_BLOCK_SIZE 512

__global__ void floatArrToUnsignedChar(float* inImage, unsigned char* outImage, int imgSize) {
    int t = threadIdx.x;
    int id = blockIdx.x * CONVERT_BLOCK_SIZE + t;//will use stride behavior like in histogramming
    unsigned int stride = blockDim.x * gridDim.x;

    while(id < imgSize) {
        outImage[id] = (unsigned char) inImage[id];
        id += stride;
    }
}


void doCudaCannyInjectStage(unsigned char* outImage, unsigned char* inImage, double* timestamps, int width, int height, int stage, float* injection) {
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
        //further allocations and thread configs here
        
        //placeholder to ensure framework functionality
        cudaMemcpy(dEdgeGradient, dGaussFilterOut, sizeof(float)*imgSize, cudaMemcpyDeviceToDevice);
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
        //further allocations and thread configs here
        
        //placeholder to ensure framework functionality
        cudaMemcpy(dNmsOutput, dEdgeGradient, sizeof(float)*imgSize, cudaMemcpyDeviceToDevice);
    }

    cudaEventRecord(lEnd);
    cudaEventSynchronize(lEnd);
    cudaEventElapsedTime(&time, lStart, lEnd);
    timestamps[2] = time;//store to timestamp array

    cudaFree(dEdgeGradient);
    cudaFree(dDirections);

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
        
        //placeholder to ensure framework functionality
        cudaMemcpy(dThreshOut, dNmsOutput, sizeof(float)*imgSize, cudaMemcpyDeviceToDevice);
    }

    cudaEventRecord(lEnd);
    cudaEventSynchronize(lEnd);
    cudaEventElapsedTime(&time, lStart, lEnd);
    timestamps[3] = time;//store to timestamp array

    cudaFree(dNmsOutput);

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


#define maxReduceBlockSize 256
__global__ void maxReduction(float* arrIn, float* maxOut, unsigned int size) {
    __shared__ float data[maxReduceBlockSize*2];

    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockDim.x*blockIdx.x;

    //preload 2 elements per thread
    data[t] = (start + t < size) ? arrIn[start + t] : 0;
    data[t + maxReduceBlockSize] = (start + maxReduceBlockSize + t < size) ? arrIn[start + maxReduceBlockSize + t] : 0;

    for (unsigned int stride = maxReduceBlockSize; stride >= 1; stride >>=1) {
        __syncthreads();
        if (t < stride)
            if (data[t] < data[t + stride])
                data[t] = data[t + stride];
    }

    if (t == 0) //last active thread will be holding max value, should write it back
        maxOut[blockIdx.x] = data[t];
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
        divSize = ceil((float) size / (maxReduceBlockSize * 2));
        maxReduction<<<divSize, maxReduceBlockSize>>>(maxOut, maxOut, tempDivSize);
    }
    cudaDeviceSynchronize();
    float res = 0;
    cudaMemcpy(&res, maxOut, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(maxOut);
    
    return res;
}