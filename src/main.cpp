#include <iostream>
#include <fstream>
#include <iomanip>
#include <ctime>

#include <opencv2/opencv.hpp>

#include "dateProcessor/dateProcessor.hpp"
#include "carProcessor/carProcessor.hpp"

using namespace cv;

void print_coords(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        std::cout << "x: " << x << ", y: " << y << std::endl;
    }
}

std::string calculateTimestamp(const std::string& startDateTime, int elapsedSeconds) {
    std::tm startTm = {};
    std::istringstream ss(startDateTime);
    ss >> std::get_time(&startTm, "%d/%m/%Y %H:%M");

    if (ss.fail()) {
        std::cerr << "Error al interpretar la fecha inicial: " << startDateTime << std::endl;
        return "";
    }

    std::time_t startEpoch = std::mktime(&startTm);
    startEpoch += elapsedSeconds;
    std::tm* resultTm = std::localtime(&startEpoch);

    std::ostringstream out;
    out << std::put_time(resultTm, "%d/%m/%Y %H:%M");

    return out.str();
}

int main() {
    namedWindow("Car Detection", WINDOW_NORMAL);
    setMouseCallback("Car Detection", print_coords);

    CascadeClassifier car_cascade;
    loadCarCascade(car_cascade);

    VideoCapture cap("../video/video.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error abriendo el archivo de video" << std::endl;
        return -1;
    }

    std::string startDateTime = "12/10/2024 08:30";
    double fps = cap.get(CAP_PROP_FPS);
    int frameSkip = 1;
    int frameCount = 0;

    std::ofstream csvFile("car_counts.csv");
    csvFile << "Time,LeftLaneCars,RightLaneCars\n";

    while (true) {
        Mat frame;
        cap >> frame;

        if (frame.empty()) {
            break;
        }

        // if (frameCount++ % frameSkip != 0) {
        //     continue;
        // }

        int elapsedSeconds = static_cast<int>(frameCount / fps);
        std::string timestamp = calculateTimestamp(startDateTime, elapsedSeconds);
        std::string dateTime = extractTextFromROI(frame);

        std::vector<Rect> cars_left, cars_right;
        processFrame(frame, cars_left, cars_right, car_cascade);

        csvFile << timestamp << "," << cars_left.size() << "," << cars_right.size() << std::endl;

        drawCars(frame, cars_left, cars_right);
        drawDate(frame);

        imshow("Car Detection", frame);

        std::cout << "Timestamp: " << timestamp
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
