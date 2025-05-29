#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main()
{
    cout << "\nHello Opencv !" << endl;
    string path = "D:/test.png"; // 确保路径正确
    cv::Mat img = imread(path);
    if (img.empty()) {
        cout << "Error: Could not load image at " << path << endl;
        return -1;
    }
    imshow("img", img);
    waitKey(0);
    return 0;
}