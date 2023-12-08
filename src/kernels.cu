#include "../inc/kernels.h"





__global__ void cudaDoubleThreshold(float* f_nmsIn, float* f_threshOut,
                                    float f_lowThreshRatio, float f_highThreshRatio,
                                    U32 u32_width, U32 u32_height, float f_max)
{
    U32 row = threadIdx.y  + blockIdx.y * blockDim.y;
    U32 col = threadIdx.x + blockIdx.x * blockDim.x;

    if (col < u32_width && row < u32_height)
    {
        float f_highThresh = f_max * f_highThreshRatio;
        float f_lowThresh = f_highThresh * f_lowThreshRatio;

        U32 idx = row * u32_width + col;

        if (f_nmsIn[idx] >= f_highThresh)
            f_threshOut[idx] = 255;
        else if (f_nmsIn[idx] >= f_lowThresh)
            f_threshOut[idx] = 25;
        else
            f_threshOut[idx] = 0;
    }
}


__global__ void cudaHysteresis(float* f_threshIn, float* f_hystOut, U32 u32_width, U32 u32_height)
{
    U32 col = threadIdx.x + blockIdx.x * blockDim.x;
    U32 row = threadIdx.y + blockIdx.y * blockDim.y;

    if (0 < col && col < u32_width - 1 && 0 < row && row < u32_height - 1)
    {
        U32 id = row * u32_width + col - u32_width;
        for (int i = 0; i < 2; ++i)
        {
            if (f_threshIn[id - 1] == 255 || f_threshIn[id] == 255 || f_threshIn[id + 1] == 255)
                f_hystOut[id] = 255;
            id += u32_width;
        }

    }

    // if (f_threshIn[col] == 255)
    //     f_hystOut[col] = 255;
    // else if (f_threshIn[col - 1] || f_threshIn[co])
}


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


//may need to optimize dimensioning for this operation


__global__ void floatArrToUnsignedChar(float* inImage, unsigned char* outImage, int imgSize) {
    int t = threadIdx.x;
    int id = blockIdx.x * CONVERT_BLOCK_SIZE + t;//will use stride behavior like in histogramming
    unsigned int stride = blockDim.x * gridDim.x;

    while(id < imgSize) {
        outImage[id] = (unsigned char) inImage[id];
        id += stride;
    }
}
