#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>

using namespace cv;
using namespace std;

// Función para extraer texto desde una ROI usando Tesseract
string extractTextFromROI(const Mat& frame, Rect roi) {
    Mat roiImage = frame(roi);
    cvtColor(roiImage, roiImage, COLOR_BGR2GRAY); // Convertir a escala de grises
    threshold(roiImage, roiImage, 100, 255, THRESH_BINARY); // Binarizar para mejor OCR

    tesseract::TessBaseAPI ocr;
    ocr.Init(nullptr, "eng", tesseract::OEM_LSTM_ONLY);
    ocr.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
    ocr.SetImage(roiImage.data, roiImage.cols, roiImage.rows, 1, roiImage.step);

    string text = string(ocr.GetUTF8Text());
    ocr.End();

    return text;
}

int main() {
    // Cargar el clasificador Haar Cascade para coches
    CascadeClassifier car_cascade;
    if (!car_cascade.load("./cars.xml")) {
        cerr << "Error loading cars.xml" << endl;
        return -1;
    }

    VideoCapture cap("./video.mp4");
    if (!cap.isOpened()) {
        cerr << "Error opening video file" << endl;
        return -1;
    }

    // Configurar archivo CSV
    ofstream csvFile("car_counts.csv");
    csvFile << "Time,LeftLaneCars,RightLaneCars\n";

    // ROI para fecha/hora (ajustar según el video)
    Rect dateRoi(50, 50, 200, 50);

    // ROI de carriles
    vector<Point> roi_left = {
        Point(610, 715),
        Point(1305, 135),
        Point(1510, 195),
        Point(930, 850)
    };

    vector<Point> roi_right = {
        Point(930, 850),
        Point(1510, 195),
        Point(1715, 255),
        Point(1350, 900)
    };

    Mat frame;
    while (cap.read(frame)) {
        // Extraer texto de la ROI de fecha/hora
        string dateTime = extractTextFromROI(frame, dateRoi);

        // Procesar imagen para detección
        Mat gray, closed, dilated;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(5, 5), 0);

        Mat kernel_close = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
        morphologyEx(gray, closed, MORPH_CLOSE, kernel_close);

        Mat kernel_dilate = Mat::ones(3, 3, CV_8U);
        dilate(closed, dilated, kernel_dilate);

        // Crear máscaras y detectar coches
        Mat mask_left = Mat::zeros(dilated.size(), dilated.type());
        Mat mask_right = Mat::zeros(dilated.size(), dilated.type());
        fillPoly(mask_left, vector<vector<Point>>{roi_left}, Scalar(255));
        fillPoly(mask_right, vector<vector<Point>>{roi_right}, Scalar(255));

        Mat masked_gray_left, masked_gray_right;
        bitwise_and(dilated, mask_left, masked_gray_left);
        bitwise_and(dilated, mask_right, masked_gray_right);

        vector<Rect> cars_left, cars_right;
        car_cascade.detectMultiScale(masked_gray_left, cars_left, 1.1, 5, 0, Size(30, 30));
        car_cascade.detectMultiScale(masked_gray_right, cars_right, 1.1, 5, 0, Size(30, 30));

        // Guardar resultados en CSV
        csvFile << dateTime << "," << cars_left.size() << "," << cars_right.size() << "\n";

        // Dibujar ROI y detecciones
        polylines(frame, roi_left, true, Scalar(0, 255, 0), 2);
        polylines(frame, roi_right, true, Scalar(255, 0, 255), 2);

        for (const auto& car : cars_left) {
            rectangle(frame, car, Scalar(255, 0, 0), 2);
        }
        for (const auto& car : cars_right) {
            rectangle(frame, car, Scalar(0, 0, 255), 2);
        }

        imshow("Car Detection", frame);

        if (waitKey(30) == 27) {
            break;
        }
    }

    cap.release();
    csvFile.close();
    destroyAllWindows();

    return 0;
}
