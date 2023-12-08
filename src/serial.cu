#include <iostream>
#include <stdio.h>
#include <chrono>
#include <cmath>

#define M_PI           3.14159265358979323846

void serialGaussianFilter(unsigned char* imageIn, float* imageOut, int width, int height){
    int gKernelSize = 5;
    int gKernelSum = 159;
    int gKDiv2 = gKernelSize >> 1;

    unsigned char gKernel[gKernelSize*gKernelSize] = 
    {
    2, 4, 5, 4, 2,
    4, 9,12, 9, 4,
    5,12,15,12, 5,
    2, 4, 5, 4, 2,
    4, 9,12, 9, 4
    };

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0;

            for (int n = -1*gKDiv2; n <= gKDiv2; n++) { //rows of kernel
                if (y + n < 0 || y + n >= height)
                    continue; //row of kernel corresponds to out of bounds index of matrix
                int nId = n + gKDiv2;//restore index
                for (int m = -1*gKDiv2; m <= gKDiv2; m++) { //columns of kernel
                    if (x + m < 0 || x + m >= width)
                        continue;//column of kernel corresponds to out of bounds index of image
                    int mId = m + gKDiv2;//restore m index
                    sum += gKernel[nId*gKernelSize + mId]*imageIn[(y + n)*width + (x + m)];
                }
            }

            sum /= gKernelSum;
            imageOut[y * width + x] = sum;
        }
    }
}


/**
* Sobel Filter
* apply convolution using both vertical and horizontal sobel filters to determine local edge gradients
* said gradients are the magnitude of the vector formed by the x/y direction convolutions
* also determine gradient direction for each pixel using arctan2 and the x/y direction convolutions
* once all gradients are found, determine the max gradient magnitude in the image and clamp the whole 
* image based on that value such that the maximum gradient value in the image becomes 255
*/
void serialSobelFilter(float* imageIn, float* edgeGradient, float* direction, int width, int height) {
    int filterSize = 3;
    int filterDiv = 1;
    int Kx[filterSize*filterSize] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    int Ky[filterSize*filterSize] = {1, 2, 1, 0, 0, 0, -1, -2, -1};

    float max = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float ix = 0;
            float iy = 0;

            for (int n = -1*filterDiv; n <= filterDiv; n++) { //rows of kernel
                if (y + n < 0 || y + n >= height)
                    continue; //row of kernel corresponds to out of bounds index of matrix
                int nId = n + filterDiv;//restore index
                for (int m = -1*filterDiv; m <= filterDiv; m++) { //columns of kernel
                    if (x + m < 0 || x + m >= width)
                        continue;//column of kernel corresponds to out of bounds index of image
                    int mId = m + filterDiv;//restore m index
                    ix += Kx[nId*filterSize + mId]*imageIn[(y + n)*width + (x + m)];
                    iy += Ky[nId*filterSize + mId]*imageIn[(y + n)*width + (x + m)];
                }
            }

            unsigned int id = y * width + x;
            edgeGradient[id] = std::hypot(ix, iy);
            direction[id] = std::atan2(iy, ix);
            if (edgeGradient[id] > max)
                max = edgeGradient[id];
        }
    }

    //normalize edgeGradient values
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float val = edgeGradient[y * width + x];
            val /= max;
            val *= 255;
            edgeGradient[y * width + x] = val;
        }
    }
}


/**
* Idea behind non-maximum supression is to analyze neighboring pixels in the gradient direction to determine whether
* the current pixel is stronger (greater) than it's neighbors in the direction of the gradient. If either pixel in the 
* gradient direction is more intense than the current one, only the more intense pixel is kept while other are set to 0
*/
void serialNonMaxSuppression(float* edgeGradientIn, float* directions, float* nsmOutput, int width, int height) {
    for (int y = 1; y < (height - 1); y++) { //exclude 1 pixel border
        for (int x = 1; x < (width - 1); x++) {
            int q = 255;
            int r = 255;
            float angle = directions[y * width + x] * 180 / M_PI; //convert to degrees
            angle = (angle < 0) ? angle + 180 : angle;// always positive

            //select directionally adjacent pixels
            if ((0 <= angle && angle < 22.5) || (157.5 <= angle && angle <= 180)) { //right or left
                q = edgeGradientIn[y * width + x + 1];// 1 pix right
                r = edgeGradientIn[y * width + x - 1];
            } else if (22.5 <= angle && angle < 67.5) { //up-right
                q = edgeGradientIn[(y + 1) * width + x - 1];
                r = edgeGradientIn[(y - 1) * width + x + 1];
            } else if (67.5 <= angle && angle < 112.5) { //up
                q = edgeGradientIn[(y + 1) * width + x];
                r = edgeGradientIn[(y - 1) * width + x];
            } else if (112.5 <= angle && angle < 157.5) { //up-left
                q = edgeGradientIn[(y + 1) * width + x + 1];
                r = edgeGradientIn[(y - 1) * width + x - 1];
            }

            int id = y * width + x;
            if (edgeGradientIn[id] >= q && edgeGradientIn[id] >= r)
                nsmOutput[id] = edgeGradientIn[id];
            else
                nsmOutput[id] = 0;
        }
    }
}


/**
* double threshold takes the result of the previous step and better defines 2 strengths of edges, it does so using
* provided high and low threshold ratios (high thresh should still be relatively low) and represent percentages of the
* maximum pixel value. Pixels exceeding or equal to the high threshold should be set to a max pixel value while pixel
* less than the low threshold are zeroed, pixels between thresholds should be set to a common lower value but still retained
*/
void serialDoubleThreshold( float* nmsIn, float* threshOut, 
                            int width, int height, 
                            float lowThreshRatio, float highThreshRatio) {
    //first need to find max again
    float max = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            max = (max < nmsIn[y * width + x]) ? nmsIn[y * width + x] : max;
        }
    }

    // TODO remove
    // printf("serial max: %f\n", max);

    float highThresh = max * highThreshRatio;
    float lowThresh = highThresh * lowThreshRatio;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned int id = y * width + x;
            if (nmsIn[id] >= highThresh)
                threshOut[id] = 255;
            else if (nmsIn[id] >= lowThresh) //above low thresh and under high due to previous IF
                threshOut[id] = 25;
            else
                threshOut[id] = 0;
        }
    }
}

/**
*   Edge tracking by hysteresis
*   based on threshold results, hysteresis is the process of transforming weak pixels into strong ones if an only if
*   at least one of the neighboring (including diagonal) pixels is a strong one. This is essentially a morphological dilation
*   with the kernel 
*   1 1 1
*   1 1 1
*   1 1 1
*   where 1's only interact with strong values of 255
*/
void serialHysteresis(float* threshIn, float* hystOut, int width, int height) {
    for (int y = 1; y < (height - 1); y++) {
        for (int x = 1; x < (width - 1); x++) { //skip 1px borders
            //check all neighboring values
            //try same row first
            int id = y * width + x;//current cell to process
            if (threshIn[id] == 255) {
                //cell already strong, simply add to output and continue
                hystOut[id] = 255;
                continue;
            } else if (threshIn[id - 1] == 255 || threshIn[id + 1] == 255) {
                //left or right neighbor is strong
                hystOut[id] = 255;
                continue;
            }
            //check row above and below
            id -= width;//subtract to move up one row while keeping same x dimension
            if (threshIn[id - 1] == 255 || threshIn[id] == 255 || threshIn[id + 1] == 255) {
                hystOut[id] = 255;
                continue;
            }
            id += 2*width;//move down 2 rows
            if (threshIn[id - 1] == 255 || threshIn[id] == 255 || threshIn[id + 1] == 255) {
                hystOut[id] = 255;
                continue;
            } //otherwise no neighbor is strong and value of output remains 0
        }
    }
}

//simple element type cast to make array compatible with OpenCV image matrix element type
void serialFloatArrToUnsignedChar(float* imageIn, unsigned char* imageOut, int imgSize) {
    for (unsigned int i = 0; i < imgSize; i++) {
        imageOut[i] = imageIn[i];
    }
}

//timing related 'using' calls
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;


void doSerialCannyExtractStage(unsigned char* outImage, unsigned char* inImage, double* timestamps, int width, int height, int stage, float* extracted) {
    // cudaEvent_t start, stop, noiseReduction, gradientCalculation, nonMaxSupression, doubleThresh, edgeTrack;
    auto start = high_resolution_clock::now();
    auto t0 = start;

    //first malloc and copy data to a float array
    unsigned int imgSize = width*height;
    float* postGaussFilter = (float*) malloc(sizeof(float)*imgSize);
    serialGaussianFilter(inImage, postGaussFilter, width, height);
    auto t1 = high_resolution_clock::now();
    duration<double, std::milli> elapsedMs = t1 - t0;
    // printf("Gaussian Filter Noise Reduction Ran in %lf ms\n", elapsedMs.count());
    timestamps[0] = elapsedMs.count();
    if (stage == 0) memcpy(extracted, postGaussFilter, sizeof(float)*imgSize);

    //sobel filter application
    t0 = high_resolution_clock::now();
    float* edgeGradient = (float*) malloc(sizeof(float)*imgSize);
    float* directions = (float*) malloc(sizeof(float)*imgSize);
    serialSobelFilter(postGaussFilter, edgeGradient, directions, width, height);
    t1 = high_resolution_clock::now();
    elapsedMs = t1 - t0;
    // printf("Sobel Filtering Ran in %lf ms\n", elapsedMs.count());
    timestamps[1] = elapsedMs.count();
    if (stage == 1) memcpy(extracted, edgeGradient, sizeof(float)*imgSize);

    //non maximum suppression
    t0 = high_resolution_clock::now();
    float* nmsOutput = (float*) calloc(imgSize, sizeof(float));
    serialNonMaxSuppression(edgeGradient, directions, nmsOutput, width, height);
    t1 = high_resolution_clock::now();
    elapsedMs = t1 - t0;
    // printf("Non-Maximum Supression Ran in %lf ms\n", elapsedMs.count());
    timestamps[2] = elapsedMs.count();
    if (stage == 2) memcpy(extracted, nmsOutput, sizeof(float)*imgSize);

    //double thresholding
    t0 = high_resolution_clock::now();
    float* threshOut = (float*) calloc(imgSize, sizeof(float));
    serialDoubleThreshold(nmsOutput, threshOut, width, height, .05, 0.09);
    t1 = high_resolution_clock::now();elapsedMs = t1 - t0;
    // printf("Double Thresholding Ran in %lf ms\n", elapsedMs.count());
    timestamps[3] = elapsedMs.count();
    if (stage == 3) memcpy(extracted, threshOut, sizeof(float)*imgSize);

    //edge tracking by hysteresis
    t0 = high_resolution_clock::now();
    float* hysteresisOut = (float*) calloc(imgSize, sizeof(float));
    serialHysteresis(threshOut, hysteresisOut, width, height);
    t1 = high_resolution_clock::now();elapsedMs = t1 - t0;
    // printf("Edge Tracking By Hysteresis Ran in %lf ms\n", elapsedMs.count());
    timestamps[4] = elapsedMs.count();


    //reset start of function time
    t0 = high_resolution_clock::now();
    serialFloatArrToUnsignedChar(hysteresisOut, outImage, imgSize);
    t1 = high_resolution_clock::now();
    elapsedMs = t1 - t0;
    // printf("Float to unsigned char conversion ran in %lf ms\n", elapsedMs.count());
    timestamps[5] = elapsedMs.count();
    elapsedMs = t1 - start;
    // printf("Time taken by CPU implementation %lf ms\n", elapsedMs.count());
    timestamps[6] = elapsedMs.count();

    free(postGaussFilter);
    free(edgeGradient);
    free(directions);
    free(nmsOutput);
    free(threshOut);
    free(hysteresisOut);
}
