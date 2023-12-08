#include "../inc/kernels.cuh"


/***************
helper functions
***************/
__global__ void gradientNormalize(float* gradient, float maxGradient, int width, int height) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * GRADIENT_NORM_BLOCK_SIZE + ty;
    int col = blockIdx.x * GRADIENT_NORM_BLOCK_SIZE + tx;

    if (row < height && col < width) {
        gradient[row * width + col] = gradient[row * width + col] / maxGradient * 255;
    }
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
    __syncthreads();
    if (t == 0) //last active thread will be holding max value, should write it back
        maxOut[blockIdx.x] = data[t];
}


__global__ void floatArrToUnsignedChar(float* inImage, unsigned char* outImage, int imgSize) {
    int t = threadIdx.x;
    int id = blockIdx.x * CONVERT_BLOCK_SIZE + t;//will use stride behavior like in histogramming
    unsigned int stride = blockDim.x * gridDim.x;

    while(id < imgSize) {
        outImage[id] = (unsigned char) inImage[id];
        id += stride;
    }
}


/*********************
canney edge steps 3- 5
*********************/
__global__ void nonMaximumSupression(float* inGradient, float* inAngle, float* outSupressedGradient, int width, int height) {
    // Could use shared memory here in the future, benefits would be minimal but would still be benefits.

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * NON_MAX_BLOCK_SIZE + ty;
    int col = blockIdx.x * NON_MAX_BLOCK_SIZE + tx;

    if (row < height - 1 && row > 0 && col > 0 && col < width - 1) {
        float pi = M_PI;
        // Gradient values of pixels directly in front of and behind this thread's pixel along angle
        int q = 255, r = 255;

        // Convert from radians to degrees
        float angle = inAngle[row * width + col];
        angle *= (180 / pi);
        if (angle < 0) {
            angle += 180;
        }
        
        // Prevent edge pixels from grabbing out of bounds
        int row_plus  = (row == height - 1) ? row : row + 1;
        int row_minus = (row == 0)      ? row : row - 1;
        int col_plus  = (col == width - 1)  ? col : col + 1;
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
}


