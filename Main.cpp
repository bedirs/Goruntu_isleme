#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4996)

#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat img = cv::imread("resim.jpeg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) img = cv::imread("resim.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Resim bulunamadi!" << std::endl;
        return -1;
    }

    cv::Mat prewittDikeyKernel = (cv::Mat_<float>(3, 3) <<
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1);
    cv::Mat prewittYatayKernel = (cv::Mat_<float>(3, 3) <<
        -1, -1, -1,
        0, 0, 0,
        1, 1, 1);

    cv::Mat prewittDikey, prewittYatay;
    cv::filter2D(img, prewittDikey, CV_8U, prewittDikeyKernel);
    cv::filter2D(img, prewittYatay, CV_8U, prewittYatayKernel);


    cv::Mat sobelDikeyKernel = (cv::Mat_<float>(3, 3) <<
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1);
    cv::Mat sobelYatayKernel = (cv::Mat_<float>(3, 3) <<
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1);

    cv::Mat sobelDikey, sobelYatay;
    cv::filter2D(img, sobelDikey, CV_8U, sobelDikeyKernel);
    cv::filter2D(img, sobelYatay, CV_8U, sobelYatayKernel);

    cv::imwrite("prewitt_dikey.jpg", prewittDikey);
    cv::imwrite("prewitt_yatay.jpg", prewittYatay);
    cv::imwrite("sobel_dikey.jpg", sobelDikey);
    cv::imwrite("sobel_yatay.jpg", sobelYatay);

    std::cout << "Tum ciktilar basariyla kaydedildi!" << std::endl;
    return 0;
}