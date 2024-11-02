#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    // Load the car cascade classifier
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

    // The 4 point have been selected manually. They represent the region of interest (ROI) where we want to detect cars
    std::vector<Point> roi_corners = {
        Point(190, 833),    
        Point(1295, 174),   
        Point(1705, 205),   
        Point(1363, 1035)   
    };

    while (true) {
        Mat frame;
        cap >> frame;

        if (frame.empty()) {
            break;
        }

        // Preprocessing starts here

        // Convert the frame to grayscale
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        Mat mask = Mat::zeros(gray.size(), gray.type()); // Create a mask with the same size as the frame, initialized to black (0)
        fillPoly(mask, std::vector<std::vector<Point>>{roi_corners}, Scalar(255)); // Fill the polygon on the mask with white (255) for the area of interest

        // Apply the mask to the grayscale image
        Mat masked_gray;
        bitwise_and(gray, mask, masked_gray);

        // Detect cars within the masked area
        std::vector<Rect> cars;
        car_cascade.detectMultiScale(masked_gray, cars, 1.1, 5, 0, Size(30, 30));

        // Draw rectangles around detected cars in the original frame
        for (size_t i = 0; i < cars.size(); i++) {
            rectangle(frame, cars[i], Scalar(255, 0, 0), 2);
        }

        // Draw the polygonal ROI on the original frame for visualization
        polylines(frame, roi_corners, true, Scalar(0, 255, 0), 2);

        // Display the frame with car detections and the ROI
        imshow("Car Detection", frame);

        // Exit the loop if the ESC key is pressed
        if (waitKey(30) == 27) {
            break;
        }
    }

    cap.release();
    return 0;
}
