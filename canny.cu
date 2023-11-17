#include <iostream>
#include <stdio.h>

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

void do_cuda_canny(unsigned char* outImage, unsigned char* inImage, int width, int height) {
    //TODO timing event setup

    unsigned int imgSize = width*height;

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
    cudaMemcpy(dImageIn, inImage, sizeof(unsigned char)*imgSize, cudaMemcpyHostToDevice);

    cudaMalloc(&dGaussFilterOut, sizeof(float)*imgSize);

    cudaMemcpyToSymbol(G_Filter_Kernel, gKernel, 25*sizeof(unsigned char));

    //gaussian filter kernel pre-call stuff
    dim3 blockSize(GAUSS_BLOCK_SIZE, GAUSS_BLOCK_SIZE);
    dim3 gridSize(ceil((float) width / GAUSS_TILE_SIZE), ceil((float) height / GAUSS_TILE_SIZE));
    gaussianFilter<<<gridSize, blockSize>>>(dImageIn, dGaussFilterOut, width, height);
    cudaDeviceSynchronize();

    //memory efficient behavior should possibly free input device array that 
    //was used in previous kernel call while waiting for current kernel call to finish
    //i.e. free dImageIn while waiting for kernel call to find gradient intensity to finish?
    
    //put next stage functions here, be sure to update later call free's and kernel func input vars







    //end variable cast to unsigned char behavior
    //need this because our intermediary operations will be working with floats for greater accuracy
    unsigned char* dImageOut;
    cudaMalloc(&dImageOut, sizeof(unsigned char)*imgSize);
    unsigned int nBlocks = ceil((float) imgSize / CONVERT_BLOCK_SIZE);
    floatArrToUnsignedChar<<<nBlocks, CONVERT_BLOCK_SIZE>>>(dGaussFilterOut, dImageOut, imgSize);
    
    //intermediary free operation example WILL CHANGE!!!
    cudaFree(dImageIn); //done after kernel call to allow CPU to run this while GPU runs kernel function?
    
    cudaDeviceSynchronize();
    

    cudaMemcpy(outImage, dImageOut, sizeof(unsigned char)*imgSize, cudaMemcpyDeviceToHost);

    //final free operations, subject to change
    cudaFree(dImageOut);
    cudaFree(dGaussFilterOut);
}
