#include <stdio.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void do_cuda_canny(unsigned char* outImage, unsigned char* inImage, int width, int height);
void do_serial_canny(unsigned char* outImage, unsigned char* inImage, int width, int height);

//simple function to compare OpenCV mat/image equivalency
bool equal(const cv::Mat & a, const cv::Mat & b)
{
    if ( (a.rows != b.rows) || (a.cols != b.cols) )
        return false;
    cv::Scalar s = sum( a - b );
    return (s[0]==0) && (s[1]==0) && (s[2]==0);
}





int main(int argc, char *argv[]) {
    //TODO: implement some kind of argument based or interactive image selection
    cv::Mat greyImg;    
    cv::Mat baseImg = cv::imread("/home/jans/EE5351/canny_cuda/dog_flag.jpg");
    cv::cvtColor(baseImg, greyImg, cv::COLOR_BGR2GRAY);
    greyImg.convertTo(greyImg, CV_8U);

    int w = greyImg.cols;
    int h = greyImg.rows;

    cv::Mat edgeImgCuda(h, w, CV_8UC1, cv::Scalar::all(0));    
    cv::Mat edgeImgSerial(h, w, CV_8UC1, cv::Scalar::all(0));



    do_serial_canny((uint8_t*) edgeImgSerial.data, (uint8_t*) greyImg.data, w, h);

    do_cuda_canny((uint8_t*) edgeImgCuda.data, (uint8_t*) greyImg.data, w, h);


    
    if (equal(edgeImgSerial, edgeImgCuda)) {
        printf ("Images are equivalent!\n");
    } else {
        printf ("Images are NOT equivalent!\n");
    }

    // std::cout << "Width: " << greyImg.cols << "\nHeight: " << greyImg.rows << std::endl;
    // cv::imshow("Image", greyImg);

    cv::imshow("Original Image", greyImg);
    cv::imshow("Post Serial Image", edgeImgSerial);
    cv::imshow("Post CUDA Image", edgeImgCuda);

    int k = cv::waitKey(0);

    return 0;
}