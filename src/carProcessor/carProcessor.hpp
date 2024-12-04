#ifndef CAR_PROCESSOR_HPP
#define CAR_PROCESSOR_HPP

#include <opencv2/opencv.hpp>

using namespace cv;

int loadCarCascade(CascadeClassifier &carCascade);
void processFrame(Mat &frame, std::vector<Rect> &cars_left, std::vector<Rect> &cars_right, CascadeClassifier &carCascade);
void drawCars(Mat &frame, std::vector<Rect> &cars_left, std::vector<Rect> &cars_right);

#endif