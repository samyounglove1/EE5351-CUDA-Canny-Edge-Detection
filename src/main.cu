#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



void doCudaCannyInjectStage(unsigned char* outImage, unsigned char* inImage, double* timestamps, int width, int height, int stage, float* injection);
void doCudaCanny(unsigned char* outImage, unsigned char* inImage, double* timestamps, int width, int height) {
    doCudaCannyInjectStage(outImage, inImage, timestamps, width, height, -1, nullptr);
}

// void exDoCudaCanny(unsigned char* outImage, unsigned char* inImage, double* timestamps, int width, int height);

// void doSerialCanny(unsigned char* outImage, unsigned char* inImage, double* timestamps, int width, int height);
void doSerialCannyExtractStage(unsigned char* outImage, unsigned char* inImage, double* timestamps, int width, int height, int stage, float* extracted);
//default call
void doSerialCanny(unsigned char* outImage, unsigned char* inImage, double* timestamps, int width, int height) {
    //simply call extract stage variant with args that wont cause deviated behavior
    doSerialCannyExtractStage(outImage, inImage, timestamps, width, height, -1, nullptr);
}

void multiImageBenchmarkTests();
void videoDemo();


//simple function to compare OpenCV mat/image equivalency
bool equal(const cv::Mat & a, const cv::Mat & b)
{
    // std::cout << "pixel values " << (int) a.at<int>(0, 0) << " " << (int) b.at<int>(0, 0) << std::endl;
    if ( (a.rows != b.rows) || (a.cols != b.cols) )
        return false;
    cv::Scalar s = sum( a - b );
    return (s[0]==0) && (s[1]==0) && (s[2]==0);
}


void prettyPrintBenchmarks(std::string name, double* serialBenchmarks, double* cudaBenchmarks, bool equal, bool first, bool parallelOnly) {
    if (first) {
        std::cout << std::right << std::setw(9) << std::setfill(' ') << "Image"  << std::setw(3) << " | ";
        std::cout << std::right << std::setw(9) << std::setfill(' ') << "Impl"   << std::setw(3) << " | ";
        std::cout << std::right << std::setw(11) << std::setfill(' ') << "Step 1" << std::setw(3) << " | ";
        std::cout << std::right << std::setw(11) << std::setfill(' ') << "Step 2" << std::setw(3) << " | ";
        std::cout << std::right << std::setw(11) << std::setfill(' ') << "Step 3" << std::setw(3) << " | ";
        std::cout << std::right << std::setw(11) << std::setfill(' ') << "Step 4" << std::setw(3) << " | ";
        std::cout << std::right << std::setw(11) << std::setfill(' ') << "Step 5" << std::setw(3) << " | ";
        std::cout << std::right << std::setw(11) << std::setfill(' ') << "Cast Time"   << std::setw(3) << " | ";
        std::cout << std::right << std::setw(11) << std::setfill(' ') << "Total"  << std::setw(3) << " | ";
        std::cout << std::right << std::setw(12) << std::setfill(' ') << "Pass/Speedup"  << std::endl;
    }

    //print stats for serial
    if (!parallelOnly) {
        std::cout << std::right << std::setw(9) << std::setfill(' ') << name << std::setw(3) << " | ";
        std::cout << std::right << std::setw(9) << std::setfill(' ') << "Serial" << std::setw(3) << " | ";
        std::cout << std::right << std::setw(9) << std::setfill(' ') << serialBenchmarks[0] << "ms" << std::setw(3) << " | ";
        std::cout << std::right << std::setw(9) << std::setfill(' ') << serialBenchmarks[1] << "ms" << std::setw(3) << " | ";
        std::cout << std::right << std::setw(9) << std::setfill(' ') << serialBenchmarks[2] << "ms" << std::setw(3) << " | ";
        std::cout << std::right << std::setw(9) << std::setfill(' ') << serialBenchmarks[3] << "ms" << std::setw(3) << " | ";
        std::cout << std::right << std::setw(9) << std::setfill(' ') << serialBenchmarks[4] << "ms" << std::setw(3) << " | ";
        std::cout << std::right << std::setw(9) << std::setfill(' ') << serialBenchmarks[5] << "ms" << std::setw(3) << " | ";
        std::cout << std::right << std::setw(9) << std::setfill(' ') << serialBenchmarks[6] << "ms" << std::setw(3) << " | ";
        std::cout << std::right << std::setw(12) << std::setfill(' ') << ((equal) ? "PASSED" : "FAILED") << std::endl;
    }
    //print stats for parallel
    std::cout << std::right << std::setw(9) << std::setfill(' ') << name  << std::setw(3) << " | ";
    std::cout << std::right << std::setw(9) << std::setfill(' ') << "Parallel"  << std::setw(3) << " | ";
    std::cout << std::right << std::setw(9) << std::setfill(' ') << cudaBenchmarks[0] << "ms" << std::setw(3) << " | ";
    std::cout << std::right << std::setw(9) << std::setfill(' ') << cudaBenchmarks[1] << "ms" << std::setw(3) << " | ";
    std::cout << std::right << std::setw(9) << std::setfill(' ') << cudaBenchmarks[2] << "ms" << std::setw(3) << " | ";
    std::cout << std::right << std::setw(9) << std::setfill(' ') << cudaBenchmarks[3] << "ms" << std::setw(3) << " | ";
    std::cout << std::right << std::setw(9) << std::setfill(' ') << cudaBenchmarks[4] << "ms" << std::setw(3) << " | ";
    std::cout << std::right << std::setw(9) << std::setfill(' ') << cudaBenchmarks[5] << "ms" << std::setw(3) << " | ";
    std::cout << std::right << std::setw(9) << std::setfill(' ') << cudaBenchmarks[6] << "ms" << std::setw(3) << " | ";
    if (!parallelOnly) {
        float speedup = serialBenchmarks[6] / cudaBenchmarks[6];
        std::cout << std::right << std::setw(12) << std::setfill(' ') << std::to_string(speedup) + 'X';
    }
    std::cout << std::endl;

    //print stats for serial
    if (!parallelOnly) {
        std::cout << std::right << std::setw(9) << std::setfill(' ') << name << std::setw(3) << " | ";
        std::cout << std::right << std::setw(9) << std::setfill(' ') << "Speedups" << std::setw(3) << " | ";
        std::cout << std::right << std::setw(10) << std::setfill(' ') << serialBenchmarks[0] / cudaBenchmarks[0] << "X" << std::setw(3) << " | ";
        std::cout << std::right << std::setw(10) << std::setfill(' ') << serialBenchmarks[1] / cudaBenchmarks[1] << "X" << std::setw(3) << " | ";
        std::cout << std::right << std::setw(10) << std::setfill(' ') << serialBenchmarks[2] / cudaBenchmarks[2] << "X" << std::setw(3) << " | ";
        std::cout << std::right << std::setw(10) << std::setfill(' ') << serialBenchmarks[3] / cudaBenchmarks[3] << "X" << std::setw(3) << " | ";
        std::cout << std::right << std::setw(10) << std::setfill(' ') << serialBenchmarks[4] / cudaBenchmarks[4] << "X" << std::setw(3) << " | ";
        std::cout << std::right << std::setw(10) << std::setfill(' ') << serialBenchmarks[5] / cudaBenchmarks[5] << "X" << std::setw(3) << " | ";
        std::cout << std::right << std::setw(10) << std::setfill(' ') << serialBenchmarks[6] / cudaBenchmarks[6] << "X" << std::setw(3) << " | ";
        // std::cout << std::right << std::setw(12) << std::setfill(' ') << ((equal) ? "PASSED" : "FAILED") << std::endl;
        std::cout << std::endl;
    }
}


/**
* Valid arguments and denotations
* for simplicity, only the first '-x' argument and its value will be recognized
* -i <imagePath> :: will force code to run on a single image at the specified path
* -d <stageValue> :: useful for latter || stage construction, will extract the image data from
*                    the specified stage in serial and use it as the input the following stage in ||
*                    note: this mode will force the image we're operating on to be the lenna image for simplicity
* -m <modeValue> :: overriden by -i, this will specify a mode of operation, options are to be as follows:
*                   0 == default benchmarking with the array of images in the ./images folder
*                   1 == parallel stress test via live video playback, will attempt to match framerate as closely as can
*
* if no arguments give, program will run as if -m 0 was the only argument given
*/
int main(int argc, char *argv[]) {
    char mode = 'm';
    int modeVar = 0;

    if (argc > 1) {
        //we'll assume proper input formatting
        mode = argv[1][1];
        if (mode == 'm') {
            modeVar = std::atoi(argv[2]);
        }
    }

    if (mode == 'm') {//mode based benchmarking or demo
        if (modeVar == 0) {
            multiImageBenchmarkTests();
        } else if (modeVar == 1) {
            videoDemo();
        }
        
    } else if (mode == 'i') { //user specified image
        
        cv::namedWindow("Post Serial Image", cv::WINDOW_NORMAL);
        cv::namedWindow("Post CUDA Image", cv::WINDOW_NORMAL);
        cv::Mat baseImage = cv::imread(argv[2]);
        if (baseImage.empty()) {
            printf("Failed to load image!\n");
            return -1;
        } else {
            double* serialBenchmarks = (double*) malloc(sizeof(double)*7);
            double* cudaBenchmarks = (double*) malloc(sizeof(double)*7);

            cv::Mat greyImage;
            cv::cvtColor(baseImage, greyImage, cv::COLOR_BGR2GRAY);
            greyImage.convertTo(greyImage, CV_8U);//set pixel values to unsigned-8bit
            int w = greyImage.cols;
            int h = greyImage.rows;

            cv::Mat edgeImgCuda(h, w, CV_8UC1, cv::Scalar::all(0));  
            cv::Mat edgeImgSerial(h, w, CV_8UC1, cv::Scalar::all(0));

            doSerialCanny((uint8_t*) edgeImgSerial.data, (uint8_t*) greyImage.data, serialBenchmarks, w, h);
            if (argc == 4 && argv[3][1] == 'x') {
                // exDoCudaCanny((uint8_t*) edgeImgCuda.data, (uint8_t*) greyImage.data, cudaBenchmarks, w, h);
            } else {
                doCudaCanny((uint8_t*) edgeImgCuda.data, (uint8_t*) greyImage.data, cudaBenchmarks, w, h);
            }

            prettyPrintBenchmarks("custom", serialBenchmarks, cudaBenchmarks, equal(edgeImgSerial, edgeImgCuda), true, false);

            cv::imshow("Original Image", greyImage);
            cv::imshow("Post Serial Image", edgeImgSerial);
            cv::imshow("Post CUDA Image", edgeImgCuda);

            int k = cv::waitKey(0);

            free(serialBenchmarks);
            free(cudaBenchmarks);
        }

    } else if (mode == 'd') {//extraction/injection debug mode
        cv::Mat baseImage = cv::imread("../media/images/256x256.jpg");
        int stage = std::atoi(argv[2]);

        double* serialBenchmarks = (double*) malloc(sizeof(double)*7);
        double* cudaBenchmarks = (double*) malloc(sizeof(double)*7);

        cv::Mat greyImage;
        cv::cvtColor(baseImage, greyImage, cv::COLOR_BGR2GRAY);
        greyImage.convertTo(greyImage, CV_8U);//set pixel values to unsigned-8bit
        int w = greyImage.cols;
        int h = greyImage.rows;

        //initialize images for eventual storage of the canny edge detections
        cv::Mat edgeImgCuda(h, w, CV_8UC1, cv::Scalar::all(0));  
        cv::Mat edgeImgSerial(h, w, CV_8UC1, cv::Scalar::all(0));

        float* injection = (float*) malloc(sizeof(float)*w*h);

        doSerialCannyExtractStage((uint8_t*) edgeImgSerial.data, (uint8_t*) greyImage.data, serialBenchmarks, w, h, stage, injection);
        doCudaCannyInjectStage((uint8_t*) edgeImgCuda.data, (uint8_t*) greyImage.data, cudaBenchmarks, w, h, stage, injection);

        prettyPrintBenchmarks("custom", serialBenchmarks, cudaBenchmarks, equal(edgeImgSerial, edgeImgCuda), true, false);

        cv::imshow("Original Image", greyImage);
        cv::imshow("Post Serial Image", edgeImgSerial);
        cv::imshow("Post CUDA Image", edgeImgCuda);

        free(serialBenchmarks);
        free(cudaBenchmarks);

        int k = cv::waitKey(0);
    }

    return 0;
}

void multiImageBenchmarkTests() {
    int numTests = 6;
    std::string tests[numTests] = {"256x256", "512x512", "1Kx1K", "2Kx2K", "4Kx4K", "4Kx6K"};

    double* serialBenchmarks = (double*) malloc(sizeof(double)*7);
    double* cudaBenchmarks = (double*) malloc(sizeof(double)*7);

    bool first = true;
    for (int i = 0; i < numTests; i++) {
        cv::Mat baseImage = cv::imread("../media/images/" + tests[i] + ".jpg");
        if (baseImage.empty()) {
            printf("Failed to load image!\n");
            return;
        } else {
            cv::Mat greyImage;
            cv::cvtColor(baseImage, greyImage, cv::COLOR_BGR2GRAY);
            greyImage.convertTo(greyImage, CV_8U);//set pixel values to unsigned-8bit
            int w = greyImage.cols;
            int h = greyImage.rows;
    
            //initialize images for eventual storage of the canny edge detections
            cv::Mat edgeImgCuda(h, w, CV_8UC1, cv::Scalar::all(0));  
            cv::Mat edgeImgSerial(h, w, CV_8UC1, cv::Scalar::all(0));
    
            doSerialCanny((uint8_t*) edgeImgSerial.data, (uint8_t*) greyImage.data, serialBenchmarks, w, h);
            doCudaCanny((uint8_t*) edgeImgCuda.data, (uint8_t*) greyImage.data, cudaBenchmarks, w, h);
    
            prettyPrintBenchmarks(tests[i], serialBenchmarks, cudaBenchmarks, equal(edgeImgSerial, edgeImgCuda), first, false);
            first = false;
    
            // cv::imshow("Original Image", greyImage);
            // cv::imshow("Post Serial Image", edgeImgSerial);
            // cv::imshow("Post CUDA Image", edgeImgCuda);
    
            // int k = cv::waitKey(0);
    
        }
    }

    free(serialBenchmarks);
    free(cudaBenchmarks);
}

void videoDemo() {
    std::cout << cv::getBuildInformation() << std::endl;
    cv::VideoCapture cap("../media/videos/2kvid.mp4");

    // Check if camera opened successfully
    if(!cap.isOpened()){
        std::cout << "Error opening video stream or file" << std::endl;
        return;
    }

    double* cudaBenchmarks = (double*) malloc(sizeof(double)*7);
    int fps = (int) cap.get(cv::CAP_PROP_FPS);
    
    bool first = true;
    int frameNum = 1;

    cv::namedWindow("Edge Detect Video", cv::WINDOW_NORMAL);
    cv::namedWindow("Base Video", cv::WINDOW_NORMAL);
    cv::waitKey(0);
    
    while(1){
    
        cv::Mat frame;
        // Capture frame-by-frame
        cap >> frame;
    
        // If the frame is empty, break immediately
        if (frame.empty())
            break;

    
        cv::Mat greyImage;
        cv::cvtColor(frame, greyImage, cv::COLOR_BGR2GRAY);
        greyImage.convertTo(greyImage, CV_8U);//set pixel values to unsigned-8bit
        int w = greyImage.cols;
        int h = greyImage.rows;
    
        //initialize images for eventual storage of the canny edge detections
        cv::Mat edgeImgCuda(h, w, CV_8UC1, cv::Scalar::all(0));  
    
        //placeholder
        // doSerialCanny((uint8_t*) edgeImgCuda.data, (uint8_t*) greyImage.data, cudaBenchmarks, w, h);
        doCudaCanny((uint8_t*) edgeImgCuda.data, (uint8_t*) greyImage.data, cudaBenchmarks, w, h);
    
        prettyPrintBenchmarks(std::to_string(frameNum), nullptr, cudaBenchmarks, true, first, true);
        first = false;
        frameNum++;
    
        cv::imshow("Base Video", frame);
        cv::imshow("Edge Detect Video", edgeImgCuda);
    
        // Press  ESC on keyboard to exit
        char c=(char)cv::waitKey(1000/fps);
        if(c==27)
            break;
    }

    free(cudaBenchmarks);
    // When everything done, release the video capture object
    cap.release();
    
    // Closes all the frames
    cv::destroyAllWindows();
    
    return;
}