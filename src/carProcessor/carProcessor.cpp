#include "carProcessor.hpp"
#include <algorithm>

std::vector<Point> roi_left = {
    Point(320, 870),
    Point(1305, 135),
    Point(1510, 195),
    Point(800, 1020)
};

std::vector<Point> roi_right = {
    Point(800, 1020),
    Point(1510, 195),
    Point(1715, 255),
    Point(1240, 1050)
};

int loadCarCascade(CascadeClassifier &carCascade){
    if (!carCascade.load("../cars.xml")) {
        std::cerr << "Error cargando el clasificador Haar Cascade" << std::endl;
        return -1;
    } else {
        return 0;
    }
}

void processFrame(Mat &frame, std::vector<Rect> &cars_left, std::vector<Rect> &cars_right, CascadeClassifier &carCascade) {
    Mat gray, closed, dilated;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, gray, Size(5, 5), 0);

    Mat kernel_close = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
    morphologyEx(gray, closed, MORPH_CLOSE, kernel_close);

    Mat kernel_dilate = Mat::ones(3, 3, CV_8U);
    dilate(closed, dilated, kernel_dilate);

    Mat mask_left = Mat::zeros(dilated.size(), dilated.type());
    Mat mask_right = Mat::zeros(dilated.size(), dilated.type());
    fillPoly(mask_left, std::vector<std::vector<Point>>{roi_left}, Scalar(255));
    fillPoly(mask_right, std::vector<std::vector<Point>>{roi_right}, Scalar(255));

    Mat masked_gray_left, masked_gray_right;
    bitwise_and(dilated, mask_left, masked_gray_left);
    bitwise_and(dilated, mask_right, masked_gray_right);

    double scaleFactor = frame.rows > 720 ? 1.05 : 1.1;
    int minNeighbors = frame.rows > 720 ? 4 : 5;

    std::vector<Rect> raw_cars_left, raw_cars_right;
    carCascade.detectMultiScale(masked_gray_left, raw_cars_left, scaleFactor, minNeighbors, 0, Size(30, 30));
    carCascade.detectMultiScale(masked_gray_right, raw_cars_right, scaleFactor, minNeighbors, 0, Size(30, 30));

    // Reducción de falsas detecciones
    auto filterDetections = [](const std::vector<Rect> &raw_cars) {
        std::vector<Rect> filtered_cars;
        for (const auto &car : raw_cars) {
            double aspect_ratio = static_cast<double>(car.width) / car.height;
            int area = car.width * car.height;

            if (area > 900 && aspect_ratio > 0.5 && aspect_ratio < 2.5) {  // Área mínima y proporción válida
                filtered_cars.push_back(car);
            }
        }
        return filtered_cars;
    };

    cars_left = filterDetections(raw_cars_left);
    cars_right = filterDetections(raw_cars_right);
}


void drawCars(Mat &frame, std::vector<Rect> &cars_left, std::vector<Rect> &cars_right) {
    polylines(frame, roi_left, true, Scalar(0, 255, 0), 2);
    polylines(frame, roi_right, true, Scalar(255, 0, 255), 2);

    for (const auto& car : cars_left) {
        rectangle(frame, car, Scalar(255, 0, 0), 2);
    }
    for (const auto& car : cars_right) {
        rectangle(frame, car, Scalar(0, 0, 255), 2);
    }
}
