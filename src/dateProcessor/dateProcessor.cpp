#include "dateProcessor.hpp"
#include <tesseract/baseapi.h>

Mat preprocessROI(const Mat& roi) {
    Mat gray, blurred, binary;

    cvtColor(roi, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(5, 5), 0);
    adaptiveThreshold(blurred, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
    resize(binary, binary, Size(), 3.0, 3.0, INTER_LINEAR);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(binary, binary, MORPH_OPEN, kernel);

    Mat labels, stats, centroids;
    int numComponents = connectedComponentsWithStats(binary, labels, stats, centroids);

    for (int i = 1; i < numComponents; ++i) {
        int area = stats.at<int>(i, CC_STAT_AREA);
        if (area < 50) { 
            binary.setTo(0, labels == i);
        }
    }

    morphologyEx(binary, binary, MORPH_CLOSE, kernel);

    return binary;
}

std::string extractTextFromROI(const Mat& frame, Rect roi) {
    Mat roiImage = frame(roi);
    Mat processedROI = preprocessROI(roiImage);

    imshow("Processed ROI", processedROI);

    tesseract::TessBaseAPI ocr;
    if (ocr.Init(nullptr, "spa", tesseract::OEM_LSTM_ONLY) != 0) {
        std::cerr << "Error inicializando Tesseract" << std::endl;
        return "";
    }
    ocr.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
    ocr.SetVariable("tessedit_char_whitelist", "0123456789:-amp ");
    ocr.SetImage(processedROI.data, processedROI.cols, processedROI.rows, 1, processedROI.step);

    std::string text = std::string(ocr.GetUTF8Text());
    text.erase(std::remove(text.begin(), text.end(), '\n'), text.end());
    ocr.End();

    return text;
}

