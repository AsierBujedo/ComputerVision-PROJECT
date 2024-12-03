#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>  

#include "dateProcessor/dateProcessor.hpp" 
#include "carProcessor/carProcessor.hpp" 

using namespace cv;

void print_coords(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        std::cout << "x: " << x << ", y: " << y << std::endl;
    }
}

int main() {
    namedWindow("Car Detection", WINDOW_NORMAL);
    setMouseCallback("Car Detection", print_coords);

    CascadeClassifier car_cascade;
    if (!car_cascade.load("./cars.xml")) {
        std::cerr << "Error cargando el clasificador Haar Cascade" << std::endl;
        return -1;
    }

    VideoCapture cap("./video.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error abriendo el archivo de video" << std::endl;
        return -1;
    }

    std::ofstream csvFile("car_counts.csv");
    csvFile << "Time,LeftLaneCars,RightLaneCars\n";

    Rect dateRoi(650, 1055, 200, 25);

    std::vector<Point> roi_left = {
        Point(610, 715),
        Point(1305, 135),
        Point(1510, 195),
        Point(930, 850)
    };

    std::vector<Point> roi_right = {
        Point(930, 850),
        Point(1510, 195),
        Point(1715, 255),
        Point(1350, 900)
    };

    while (true) {
        Mat frame;
        cap >> frame;

        if (frame.empty()) {
            break;
        }

        std::string dateTime = extractTextFromROI(frame, dateRoi);

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

        std::vector<Rect> cars_left, cars_right;
        car_cascade.detectMultiScale(masked_gray_left, cars_left, 1.1, 5, 0, Size(30, 30));
        car_cascade.detectMultiScale(masked_gray_right, cars_right, 1.1, 5, 0, Size(30, 30));

        csvFile << dateTime << "," << cars_left.size() << "," << cars_right.size() << "\n";

        rectangle(frame, dateRoi, Scalar(0, 255, 0), 2);
        polylines(frame, roi_left, true, Scalar(0, 255, 0), 2);
        polylines(frame, roi_right, true, Scalar(255, 0, 255), 2);

        for (const auto& car : cars_left) {
            rectangle(frame, car, Scalar(255, 0, 0), 2);
        }
        for (const auto& car : cars_right) {
            rectangle(frame, car, Scalar(0, 0, 255), 2);
        }

        imshow("Car Detection", frame);

        std::cout << "Time: " << dateTime 
                  << " | Left lane: " << cars_left.size() 
                  << " | Right lane: " << cars_right.size() << std::endl;

        if (waitKey(30) == 27) {
            break;
        }
    }

    cap.release();
    csvFile.close();
    destroyAllWindows();

    return 0;
}
