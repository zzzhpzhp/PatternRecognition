#include <iostream>
#include <vector>
#include <random>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

constexpr int kMax = 500;
constexpr int kMin = 0;
constexpr int kDim = 6;
constexpr int kNumsPoints = 2 * kDim;
float getRandom(float min, float max){
    float rand_num = static_cast<float>(rand()) / RAND_MAX;
    return min + rand_num * (max - min);
}

int main()
{
    std::vector<cv::Point2d> pts;
    std::srand(std::time(nullptr));
    std::vector<double> x, y;
    for (int i = 0; i < kNumsPoints; i++)
    {
        double xi = getRandom(kMin, kMax);
        double yi = getRandom(kMin, kMax);
        pts.push_back(cv::Point2d(xi, yi));
        x.push_back(xi);
        y.push_back(yi);
    }


    cv::Mat img = cv::Mat::zeros(500, 500, CV_8UC3);
    for (int i = 0; i < pts.size(); i++)
    {
        cv::Point pt(pts[i].x, pts[i].y);
        cv::circle(img, pt, 2, cv::Scalar(255, 255, 255), -1);
    }
//    cv::imshow("Random points", img);

    // 拟合曲线
    cv::Mat A = cv::Mat::zeros(kNumsPoints, kDim, CV_64FC1);
    cv::Mat b = cv::Mat::zeros(kNumsPoints, 1, CV_64FC1);
    for (int i = 0; i < kNumsPoints; i++)
    {
        for (int j = 0; j < kDim; j++)
        {
            A.at<double>(i, j) = pow(pts[i].x, j);
        }
        b.at<double>(i, 0) = pts[i].y;
    }
    cv::Mat xfit;
    cv::solve(A, b, xfit, cv::DECOMP_QR);
    std::cout << "Result: " << std::endl;
    for (int i = 0; i < kDim; i++)
    {
        std::cout << xfit.at<double>(i, 0) << "  ";
    }
    std::cout << std::endl;

    // 绘制拟合曲线
    std::vector<cv::Point> curve_pts;
    for (double xi = 0; xi <= 1000; xi += 1)
    {
        double yi = 0;
        for (int i = 0; i < kDim; i++)
        {
            yi += pow(xi, i) * xfit.at<double>(i, 0);
        }
        cv::Point pt(xi , yi);
        curve_pts.push_back(pt);
    }
    cv::polylines(img, curve_pts, false, cv::Scalar(0, 255, 0), 2);

    cv::imshow("Fitted Curve", img);
    cv::waitKey();

    return 0;
}