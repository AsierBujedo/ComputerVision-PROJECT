#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

void print_coords(int event, int x, int y, int flags, void* userdata) {
    if (event == EVENT_LBUTTONDOWN) {
        std::cout << "x: " << x << ", y: " << y << std::endl;
    }
}

int main() {

    namedWindow("Car Detection", WINDOW_AUTOSIZE);
    setMouseCallback("Car Detection", print_coords);

    // Cargar el clasificador Haar Cascade para coches
    CascadeClassifier car_cascade;
    if (!car_cascade.load("cars.xml")) {
        std::cerr << "Error loading cars.xml" << std::endl;
        return -1;
    }

    VideoCapture cap("./video.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error opening video file" << std::endl;
        return -1;
    }

    // ROI Left lane
    std::vector<Point> roi_left = {
        Point(610, 715),       // Down-Left point
        Point(1305, 135),       // Top-Left point
        Point(1510, 195),      // Top-Right point
        Point(930, 850)        // Down-Right Point
    };

    // ROI Right lane
    std::vector<Point> roi_right = {
        Point(930, 850),      // Down-Left point
        Point(1510, 195),      // Top-Left point 
        Point(1715, 255),      // Top-Right point
        Point(1350, 900)       // Down-Right Point
    };

    while (true) {
        Mat frame;
        cap >> frame;

        if (frame.empty()) {
            break;
        }

        // Convert frame to grayscale
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Create masks pero ROI
        Mat mask_left = Mat::zeros(gray.size(), gray.type());
        fillPoly(mask_left, std::vector<std::vector<Point>>{roi_left}, Scalar(255));

        Mat mask_right = Mat::zeros(gray.size(), gray.type());
        fillPoly(mask_right, std::vector<std::vector<Point>>{roi_right}, Scalar(255));

        // Apply both masks to the image (grayscale)
        Mat masked_gray_left, masked_gray_right;
        bitwise_and(gray, mask_left, masked_gray_left);
        bitwise_and(gray, mask_right, masked_gray_right);

        // Detect cars per ROI
        std::vector<Rect> cars_left, cars_right;
        car_cascade.detectMultiScale(masked_gray_left, cars_left, 1.1, 5, 0, Size(30, 30));
        car_cascade.detectMultiScale(masked_gray_right, cars_right, 1.1, 5, 0, Size(30, 30));

        // Draw rectangles defining cars in the left lane
        for (size_t i = 0; i < cars_left.size(); i++) {
            // Filter by size to avoid an over-detection
            if (cars_left[i].width > 50 && cars_left[i].height > 30) {
                rectangle(frame, cars_left[i], Scalar(255, 0, 0), 2); // Azul para el carril izquierdo
            }
        }

        // Draw rectangles defining cars in the right lane
        for (size_t i = 0; i < cars_right.size(); i++) {
            // Filter by size to avoid an over-detection
            if (cars_right[i].width > 50 && cars_right[i].height > 30) {
                rectangle(frame, cars_right[i], Scalar(0, 0, 255), 2); // Rojo para el carril derecho
            }
        }

        // Draw ROIs at the original frame setting different colours
        polylines(frame, roi_left, true, Scalar(0, 255, 0), 2); 
        polylines(frame, roi_right, true, Scalar(255, 0, 255), 2);

        imshow("Car Detection", frame);

        if (waitKey(30) == 27) {
            break;
        }
    }

    cap.release();
    return 0;
}
