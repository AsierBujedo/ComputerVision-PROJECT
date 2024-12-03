#ifndef DATE_PROCESSOR_HPP
#define DATE_PROCESSOR_HPP

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

Mat preprocessROI(const Mat& roi);
std::string extractTextFromROI(const Mat& frame, Rect roi);

#endif
