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
    loadCarCascade(car_cascade);

    VideoCapture cap("../video.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error abriendo el archivo de video" << std::endl;
        return -1;
    }

    std::ofstream csvFile("car_counts.csv");
    csvFile << "Time,LeftLaneCars,RightLaneCars\n";

    int frameSkip = 50; 
    int frameCount = 0;

    while (true) {
        Mat frame;
        cap >> frame;

        if (frame.empty()) {
            break;
        }

        // Saltar fotogramas
        if (frameCount++ % frameSkip != 0) {
            continue;
        }

        std::string dateTime = extractTextFromROI(frame);

        std::vector<Rect> cars_left, cars_right;
        processFrame(frame, cars_left, cars_right, car_cascade);

        csvFile << dateTime << "," << cars_left.size() << "," << cars_right.size() << std::endl;

        drawCars(frame, cars_left, cars_right);
        drawDate(frame);

        imshow("Car Detection", frame);

        std::cout << "Time: " << dateTime 
                  << " | Left lane: " << cars_left.size() 
                  << " | Right lane: " << cars_right.size() << std::endl;

        if (waitKey(1) == 27) { 
            break;
        }
    }

    cap.release();
    csvFile.close();
    destroyAllWindows();

    return 0;
}
