

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
// 生成随机数据
int N = 2000;
// 初始化聚类中心
int K = 10;
int iter = 200;
int BORDER_SIZE = 200;

int sign(int val)
{
    if (val % 2 == 0)
    { return 1; }
    return -1;
}

int main()
{
    srand(time(0));
    Mat data(N, 2, CV_32F);
    RNG rng(0);
//    rng.fill(data, RNG::NORMAL, 50, 90);
    int center_x = rand() % N, center_y = rand() % N;
    size_t index_cent = 0;

    for (int i = 0; i < K; i++)
    {
        center_x = rand() % N;
        center_x = std::min(center_x, N - BORDER_SIZE);
        center_x = std::max(center_x, BORDER_SIZE);
        center_y = rand() % N;
        center_y = std::min(center_y, N - BORDER_SIZE);
        center_y = std::max(center_y, BORDER_SIZE);
        std::cout << center_x << " " << center_x << std::endl;
        for (int j = 0; j < N / K; j++)
        {
            int tx = (rand() % BORDER_SIZE) * sign(rand()) + center_x, ty =
                    (rand() % BORDER_SIZE) * sign(rand()) + center_y;
            data.row(index_cent).col(0) = tx;
            data.row(index_cent).col(1) = ty;
            index_cent++;

//            std::cout << tx << " " << ty << std::endl;
        }
    }

    Mat centers(K, 2, CV_32F);
    for (int i = 0; i < K; i++)
    {
        centers.row(i).col(0) = rand() % 1000;
        centers.row(i).col(1) = rand() % 1000;
    }

    // 运行K-Means聚类
    Mat labels(N, 1, CV_32S);
    while (iter-- > 0)
    {
        // E步：计算每个样本应该属于哪个聚类中心
        for (int i = 0; i < N; i++)
        {
            int min_idx = -1;
            float min_dist = FLT_MAX;
            for (int j = 0; j < K; j++)
            {
                float dist = norm(data.row(i) - centers.row(j));
                if (dist < min_dist)
                {
                    min_idx = j;
                    min_dist = dist;
                }
            }
            labels.at<int>(i) = min_idx;
        }

        // M步：计算每个聚类中心的坐标
        Mat new_centers(K, 2, CV_32F, Scalar(0));
        Mat num_points(K, 1, CV_32S, Scalar(0));
        for (int i = 0; i < N; i++)
        {
            int cluster_idx = labels.at<int>(i);
            new_centers.row(cluster_idx) = new_centers.row(cluster_idx) + data.row(i);
            num_points.at<int>(cluster_idx)++;
        }
        for (int i = 0; i < K; i++)
        {
            if (num_points.at<int>(i) > 0)
            {
                new_centers.row(i) = new_centers.row(i) / num_points.at<int>(i);
            }
            else
            {
                // 如果没有样本属于该聚类，则随机选择一个样本作为新的聚类中心
                int idx = rng.uniform(0, N);
                data.row(idx).copyTo(new_centers.row(i));
            }
        }
        new_centers.copyTo(centers);
    }

    // 显示结果
    Mat img(N, N, CV_8UC3, Scalar(255, 255, 255));
    for (int i = 0; i < N; i++)
    {
        int cluster_idx = labels.at<int>(i);
        circle(img, Point(data.at<float>(i, 0), data.at<float>(i, 1)), 4,
               Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), -1);
    }
    for (int i = 0; i < K; i++)
    {
        circle(img, Point(centers.at<float>(i, 0), centers.at<float>(i, 1)), 10, Scalar(0, 0, 255), -1);
    }
    resize(img, img, Size(), 0.5, 0.5);
    imshow("K-Means Clustering", img);
    waitKey(0);
    return 0;
}