#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    
    VideoCapture cap("./video.mp4");

    while(true) {
        Mat frame, real;
        cap >> frame;
        cap >> real;

        if(frame.empty()) {
            break;
        }

        //Preprocess frame
        cvtColor(frame, frame, COLOR_BGR2GRAY);
        GaussianBlur(frame, frame, Size(7, 7), 0);
        Canny(frame, frame, 150, 250);
        
        //Draw contours at real frame
        std::vector<std::vector<Point>> contours;
        std::vector<Vec4i> hierarchy;
        findContours(frame, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        for(size_t i = 0; i < contours.size(); i++) {
            drawContours(real, contours, i, Scalar(0, 255, 0), 2, 8, hierarchy);
        }

        imshow("Frame", frame);
        imshow("Real", real);

        if(waitKey(30) == 27) {
            break;
        }
    }

    return 0;
}