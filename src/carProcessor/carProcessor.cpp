#include "carProcessor.hpp"
#include <glm/glm.hpp>
#include <algorithm>
#include <optional>
#include <vector>

static std::vector<Point> roi_left = {
    Point(320, 870),
    Point(1200, 320),
    Point(1210, 250),
    Point(1335, 240),
    Point(1410, 310),
    Point(800, 1020)
};

static std::vector<Point> roi_right = {
    Point(800, 1020),
    Point(1410, 310),
    Point(1440, 260),
    Point(1525, 245),
    Point(1610, 295),
    Point(1240, 1050)
};

static std::vector<glm::vec2> last_positions_left;
static std::vector<glm::vec2> last_positions_right;

static glm::mat3 homography_inverse = glm::inverse(glm::mat3(
    1.08680211e-02,  8.00229688e-02, -9.28824499e+01,
    -2.84379452e-02,  2.91213634e-02, -1.29940476e+01,
    -5.47771087e-04, -2.32384026e-03,  1.00000000e+00
));

static std::optional<float> getVelocity(glm::vec2 &pos, bool is_left) {
    std::vector<glm::vec2> &last_positions = is_left ? last_positions_left : last_positions_right;

    // Calculate distances to all previous positions
    std::vector<float> distances;
    for (const auto &last_pos : last_positions) {
        distances.push_back(glm::distance(pos, last_pos));
    }

    // Find the closest previous position
    auto min_distance = std::min_element(distances.begin(), distances.end());
    int min_index = std::distance(distances.begin(), min_distance);

    // If the closest previous position is too far, return no velocity
    if (*min_distance > 50.0f) {
        return std::nullopt;
    }

    // Calculate homographic position
    glm::vec3 hom_pos = glm::vec3(pos.x, pos.y, 1.0f);
    glm::vec3 hom_last_pos = glm::vec3(last_positions[min_index].x, last_positions[min_index].y, 1.0f);
    hom_pos = homography_inverse * hom_pos;
    hom_last_pos = homography_inverse * hom_last_pos;
    hom_pos /= hom_pos.z;
    hom_last_pos /= hom_last_pos.z;

    // Calculate velocity
    float velocity = glm::distance(hom_pos, hom_last_pos) * 30.0f;  // 30 fps

    return velocity;
}

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
    GaussianBlur(gray, gray, Size(9, 9), 0);

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

    // Guardar posiciones de los coches
    for (const auto &car : cars_left) {
        last_positions_left.push_back(glm::vec2(car.x + car.width / 2, car.y + car.height / 2));
    }
    for (const auto &car : cars_right) {
        last_positions_right.push_back(glm::vec2(car.x + car.width / 2, car.y + car.height / 2));
    }
}

void drawCars(Mat &frame, std::vector<Rect> &cars_left, std::vector<Rect> &cars_right) {
    polylines(frame, roi_left, true, Scalar(0, 255, 0), 2);
    polylines(frame, roi_right, true, Scalar(255, 0, 255), 2);

    for (const auto& car : cars_left) {
        glm::vec2 pos(car.x + car.width / 2, car.y + car.height / 2);
        auto velocity = getVelocity(pos, true);
        if (velocity.has_value()) {
            putText(frame, std::to_string(velocity.value() * 30 * 3.6) + " km/h", Point(car.x, car.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
        }
        rectangle(frame, car, Scalar(255, 0, 0), 2);
    }
    for (const auto& car : cars_right) {
        glm::vec2 pos(car.x + car.width / 2, car.y + car.height / 2);
        auto velocity = getVelocity(pos, false);
        if (velocity.has_value()) {
            putText(frame, std::to_string(velocity.value() * 30 * 3.6) + " km/h", Point(car.x, car.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
        }
        rectangle(frame, car, Scalar(0, 0, 255), 2);
    }

    // Eliminar posiciones antiguas
    last_positions_left.clear();
    last_positions_right.clear();
}