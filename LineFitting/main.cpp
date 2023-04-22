#include <iostream>
#include <vector>
#include <cmath>
#include "opencv4/opencv2/opencv.hpp"

using namespace cv;
using namespace std;

struct Point2D {
    double x, y;

    Point2D(double _x = 0, double _y = 0) :x(_x), y(_y) {}

    // 重载加号运算符，用于点的加法
    Point2D operator + (const Point2D& rhs) const {
        return Point2D(x + rhs.x, y + rhs.y);
    }

    // 重载减号运算符，用于点的减法
    Point2D operator - (const Point2D& rhs) const {
        return Point2D(x - rhs.x, y - rhs.y);
    }

    double length() const { // 计算向量长度
        return sqrt(x * x + y * y);
    }

    double distance(const Point2D& p) const { // 计算点到点的距离
        return (*this - p).length();
    }
};

struct Line2D {
    double k, b;

    Line2D(double _k = 0, double _b = 0) :k(_k), b(_b) {}

    Point2D intersection(const Line2D& other) const { // 计算两条直线的交点
        Point2D p;
        p.x = (other.b - b) / (k - other.k);
        p.y = k * p.x + b;
        return p;
    }
};

Line2D fitLine(const vector<Point2D>& points) { // 拟合直线
    int n = points.size();
    double x_sum = 0, y_sum = 0, xy_sum = 0, x2_sum = 0;
    for (int i = 0; i < n; ++i) {
        x_sum += points[i].x;
        y_sum += points[i].y;
        xy_sum += points[i].x * points[i].y;
        x2_sum += points[i].x * points[i].x;
    }
    double denom = n * x2_sum - x_sum * x_sum;
    double k = (n * xy_sum - x_sum * y_sum) / denom;
    double b = (y_sum - k * x_sum) / n;
    return Line2D(k, b);
}

float getRandom(float min, float max){
    float rand_num = static_cast<float>(rand()) / RAND_MAX;
    return min + rand_num * (max - min);
}

constexpr size_t kPointsSize = 3;
constexpr int kMax = 500;
constexpr int kMin = 0;

int main() {
    std::srand(std::time(nullptr));
    Mat image(512, 512, CV_8UC3, Scalar(255, 255, 255)); // 新建一张白色背景图
    vector<Point2D> points;
    for (auto i = 0; i < kPointsSize; i++)
    {
        points.emplace_back(getRandom(kMin, kMax), getRandom(kMin, kMax));
    }
    // 将原始散点画到图像上
    for (int i = 0; i < points.size(); ++i) {
        circle(image, Point(points[i].x, points[i].y), 2, Scalar(0, 0, 255), -1);
    }

    // 拟合直线
    Line2D line = fitLine(points);

    // 计算直线的端点
    Point2D p1(0, line.b);
    Point2D p2(image.cols, line.k * image.cols + line.b);

    // 将拟合出的直线画到图像上
    cv::line(image, Point(p1.x, p1.y), Point(p2.x, p2.y), Scalar(0, 255, 0), 2);

    imshow("Fitting line", image);
    waitKey(0);

    return 0;
}