#include "../inc/kernels.h"

// serial for reference

// /* double threshold takes the result of the previous step and better defines 2 strengths of edges, it does so using
// * provided high and low threshold ratios (high thresh should still be relatively low) and represent percentages of the
// * maximum pixel value. Pixels exceeding or equal to the high threshold should be set to a max pixel value while pixel
// * less than the low threshold are zeroed, pixels between thresholds should be set to a common lower value but still retained
// */
// void serialDoubleThreshold(float* nmsIn, float* threshOut, int width, int height, float lowThreshRatio, float highThreshRatio) {
//     //first need to find max again
//     float max = 0;
//     for (int y = 0; y < height; y++) {
//         for (int x = 0; x < width; x++) {
//             max = (max < nmsIn[y * width + x]) ? nmsIn[y * width + x] : max;
//         }
//     }

//     float highThresh = max * highThreshRatio;
//     float lowThresh = highThresh * lowThreshRatio;

//     for (int y = 0; y < height; y++) {
//         for (int x = 0; x < width; x++) {
//             unsigned int id = y * width + x;
//             if (nmsIn[id] >= highThresh)
//                 threshOut[id] = 255;
//             else if (nmsIn[id] >= lowThresh) //above low thresh and under high due to previous IF
//                 threshOut[id] = 25;
//             else
//                 threshOut[id] = 0;
//         }
//     }
// }


/**
 *  @brief 
 *  
 *  @param f_nmsIn 
 *  @param f_threshOut 
 *  @param u32_width 
 *  @param u32_height 
 *  @param f_lowThresRatio 
 *  @param f_highThreshRatio 
 */
__global__ void cudaDoubleThreshold(float* f_nmsIn, float* f_threshOut,
                                    U32 u32_width, U32 u32_height,
                                    float f_lowThreshRatio, float f_highThreshRatio)
{
    float f_max = 0.0f;

    // TODO parallelize
    for (int y = 0; y < u32_height; y++) {
        for (int x = 0; x < u32_width; x++) {
            f_max = (f_max < f_nmsIn[y * u32_width + x]) ? f_nmsIn[y * u32_width + x] : f_max;
        }
    }

    // maxReduction(f_nmsIn, &f_max, u32_width);

    // find max again
    // printf("cuda max %f\n", f_max);

    float f_highThresh = f_max * f_highThreshRatio;
    float f_lowThresh = f_highThresh * f_lowThreshRatio;


    U32 row = threadIdx.y  + blockIdx.y * blockDim.y;
    U32 col = threadIdx.x + blockIdx.x * blockDim.x;

    if (blockIdx.x == 0 && blockIdx.y == 0)
    {
        printf("row %d col %d\n", row, col);
    }

    if (row > 0 && col > 0 && row < u32_height - 1 && col < u32_width)
    {
        U32 idx = row * col;

        
        // FILE *fptr = fopen("idx.txt", "w");
        // fprintf(fptr, "%d\n", idx);
        // fclose(fptr);
        if (f_nmsIn[idx] >= f_highThresh)
            f_threshOut[idx] = 255;
        else if (f_nmsIn[idx] >= f_lowThresh)
            f_threshOut[idx] = 25;
        else
            f_threshOut[idx] = 0;
    }

    // if (threadID )

    // python code:
    // M, N = img.shape // i believe this is the width and height args
    // res = np.zeros((MN, N), dtype=np.in32) // this is f_threshOut
    //
    // weak = np.int32(25)  // these are in the if else if of the serial
    // strong = np.int32(255)

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
