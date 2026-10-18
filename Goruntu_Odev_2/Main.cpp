#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4996)

#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

cv::Mat gaus5x5(const cv::Mat& src) {
    int h = src.rows;
    int w = src.cols;

    cv::Mat out = cv::Mat::zeros(h, w, CV_8UC1);
    const float k[5][5] = {
        { 2,  4,  5,  4, 2 },
        { 4,  9, 12,  9, 4 },
        { 5, 12, 15, 12, 5 },
        { 4,  9, 12,  9, 4 },
        { 2,  4,  5,  4, 2 }
    };
    const float toplam = 159.0f;

    for (int y = 2; y < h - 2; ++y) {
        for (int x = 2; x < w - 2; ++x) {
            float carpimToplami = 0.0f;
            for (int ky = -2; ky <= 2; ++ky) {
                for (int kx = -2; kx <= 2; ++kx) {
                    carpimToplami += src.at<uchar>(y + ky, x + kx) * k[ky + 2][kx + 2];
                }
            }
            out.at<uchar>(y, x) = static_cast<uchar>(carpimToplami / toplam);
        }
    }
    return out;
}

void sobel(const cv::Mat& src, cv::Mat& mag, cv::Mat& aci) {
    int h = src.rows;
    int w = src.cols;
    mag = cv::Mat::zeros(h, w, CV_32F);
    aci = cv::Mat::zeros(h, w, CV_32F);

    int Gx[3][3] = {
        { -1, 0, 1 },
        { -2, 0, 2 },
        { -1, 0, 1 }
    };
    int Gy[3][3] = {
        { -1, -2, -1 },
        {  0,  0,  0 },
        {  1,  2,  1 }
    };

    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            float toplamX = 0.0f;
            float toplamY = 0.0f;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    uchar p = src.at<uchar>(y + ky, x + kx);
                    toplamX += p * Gx[ky + 1][kx + 1];
                    toplamY += p * Gy[ky + 1][kx + 1];
                }
            }

            mag.at<float>(y, x) = std::sqrt(toplamX * toplamX + toplamY * toplamY);

            float derece = std::atan2(toplamY, toplamX) * 180.0f / CV_PI;
            if (derece < 0.0f) derece += 180.0f;
            aci.at<float>(y, x) = derece;
        }
    }
}

int main() {

    cv::Mat img = cv::imread("resim.jpeg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        img = cv::imread("resim.jpg", cv::IMREAD_GRAYSCALE);
    }
    if (img.empty()) {
        std::cerr << "Hata: resim dosyasi bulunamadi!" << std::endl;
        return -1;
    }

    
    cv::Mat yumusak = gaus5x5(img);

    cv::Mat buyukluk, aci;
    sobel(yumusak, buyukluk, aci);

    cv::Mat gosterBuyukluk;
    cv::normalize(buyukluk, gosterBuyukluk, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    std::string win1 = "1. Orijinal Gri Resim";
    std::string win2 = "2. Gauss 5x5 Sonucu";
    std::string win3 = "3. Sobel Gradyan (Kenar Siddeti)";

    cv::namedWindow(win1, cv::WINDOW_NORMAL);
    cv::namedWindow(win2, cv::WINDOW_NORMAL);
    cv::namedWindow(win3, cv::WINDOW_NORMAL);

    cv::resizeWindow(win1, 380, 260);
    cv::resizeWindow(win2, 380, 260);
    cv::resizeWindow(win3, 380, 260);

    cv::moveWindow(win1, 30, 50);
    cv::moveWindow(win2, 430, 50);
    cv::moveWindow(win3, 830, 50);

    cv::imshow(win1, img);
    cv::imshow(win2, yumusak);
    cv::imshow(win3, gosterBuyukluk);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}