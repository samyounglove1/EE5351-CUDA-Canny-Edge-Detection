#include <stdio.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void do_cuda_canny(unsigned char* outImage, unsigned char* inImage, int width, int height);

int main(int argc, char *argv[]) {
    //TODO: implement some kind of argument based or interactive image selection
    cv::Mat greyImg;    
    cv::Mat baseImg = cv::imread("/home/jans/EE5351/canny_cuda/lenna.png");
    cv::cvtColor(baseImg, greyImg, cv::COLOR_BGR2GRAY);
    greyImg.convertTo(greyImg, CV_8U);

    int w = greyImg.cols;
    int h = greyImg.rows;

    cv::Mat edgeImg(h, w, CV_8UC1, cv::Scalar::all(0));    
    
    // printf("Hello World from CPU!\n");
    do_cuda_canny((uint8_t*) edgeImg.data, (uint8_t*) greyImg.data, w, h);
    // do_cuda_canny2();;
    // std::cout << "Width: " << greyImg.cols << "\nHeight: " << greyImg.rows << std::endl;
    // cv::imshow("Image", greyImg);

    cv::imshow("Original Image", greyImg);
    cv::imshow("Post CUDA Image", edgeImg);

    int k = cv::waitKey(0);

    return 0;
}