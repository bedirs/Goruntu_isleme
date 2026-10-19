#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

typedef cv::Mat(*ColorConvertFunc)(const cv::Mat&);

cv::Mat customBgrToGray(const cv::Mat& src) {
    int h = src.rows;
    int w = src.cols;
    cv::Mat dst = cv::Mat::zeros(h, w, CV_8UC1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            int b = p[0];
            int g = p[1];
            int r = p[2];

            int yVal = (r * 4899 + g * 9617 + b * 1868 + (1 << 13)) >> 14;
            dst.at<uchar>(y, x) = static_cast<uchar>(std::clamp(yVal, 0, 255));
        }
    }
    return dst;
}

cv::Mat customBgrToYCrCb(const cv::Mat& src) {
    int h = src.rows;
    int w = src.cols;
    cv::Mat dst = cv::Mat::zeros(h, w, CV_8UC3);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            cv::Vec3b p = src.at<cv::Vec3b>(y, x);
            float b = static_cast<float>(p[0]);
            float g = static_cast<float>(p[1]);
            float r = static_cast<float>(p[2]);

            float Y = 0.299f * r + 0.587f * g + 0.114f * b;
            float Cr = 128.0f + 0.5f * r - 0.418688f * g - 0.081312f * b;
            float Cb = 128.0f - 0.168736f * r - 0.331264f * g + 0.5f * b;

            dst.at<cv::Vec3b>(y, x)[0] = static_cast<uchar>(std::clamp(std::round(Y), 0.0f, 255.0f));
            dst.at<cv::Vec3b>(y, x)[1] = static_cast<uchar>(std::clamp(std::round(Cr), 0.0f, 255.0f));
            dst.at<cv::Vec3b>(y, x)[2] = static_cast<uchar>(std::clamp(std::round(Cb), 0.0f, 255.0f));
        }
    }
    return dst;
}

void karsilastirVeRaporla(const std::string& uzayAdi, const cv::Mat& testImg, const cv::Mat& refImg) {
    double toplamHata = 0.0;
    long long sifirFarkPiksel = 0;
    long long toplamEleman = static_cast<long long>(testImg.rows) * testImg.cols * testImg.channels();

    int h = testImg.rows;
    int w = testImg.cols;
    int ch = testImg.channels();

    for (int y = 0; y < h; ++y) {
        const uchar* ptr1 = testImg.ptr<uchar>(y);
        const uchar* ptr2 = refImg.ptr<uchar>(y);
        for (int x = 0; x < w * ch; ++x) {
            int fark = std::abs(static_cast<int>(ptr1[x]) - static_cast<int>(ptr2[x]));
            toplamHata += fark;
            if (fark == 0) sifirFarkPiksel++;
        }
    }

    std::cout << "--- " << uzayAdi << " Karsilastirmasi ---" << std::endl;
    std::cout << "Toplam Eleman Sayisi : " << toplamEleman << std::endl;
    std::cout << "Farki 0 Olan Deger   : " << sifirFarkPiksel << std::endl;
    std::cout << "Toplam Mutlak Hata   : " << toplamHata << std::endl;
    if (toplamHata == 0.0) {
        std::cout << "DURUM: Tam Uyum! Beklenen Hata SIFIR saglandi." << std::endl;
    }
    else {
        std::cout << "DURUM: Kucuk kayan nokta (float) yuvarlama sapmasi mevcut." << std::endl;
    }
    std::cout << "---------------------------------------" << std::endl;
}

int main() {
    cv::Mat img = cv::imread("resim.jpeg");
    if (img.empty()) img = cv::imread("resim.jpg");
    if (img.empty()) {
        std::cerr << "Hata: resim bulunamadi!" << std::endl;
        return -1;
    }

    ColorConvertFunc donusturucu = nullptr;

    donusturucu = customBgrToGray;
    cv::Mat customGray = donusturucu(img);
    cv::Mat cvGray;
    cv::cvtColor(img, cvGray, cv::COLOR_BGR2GRAY);
    karsilastirVeRaporla("BGR -> GRAY", customGray, cvGray);

    donusturucu = customBgrToYCrCb;
    cv::Mat customYCrCb = donusturucu(img);
    cv::Mat cvYCrCb;
    cv::cvtColor(img, cvYCrCb, cv::COLOR_BGR2YCrCb);
    karsilastirVeRaporla("BGR -> YCrCb", customYCrCb, cvYCrCb);

    cv::namedWindow("Orijinal", cv::WINDOW_NORMAL);
    cv::namedWindow("Custom Gray", cv::WINDOW_NORMAL);
    cv::namedWindow("Custom YCrCb", cv::WINDOW_NORMAL);

    cv::resizeWindow("Orijinal", 380, 260);
    cv::resizeWindow("Custom Gray", 380, 260);
    cv::resizeWindow("Custom YCrCb", 380, 260);

    cv::moveWindow("Orijinal", 30, 50);
    cv::moveWindow("Custom Gray", 430, 50);
    cv::moveWindow("Custom YCrCb", 830, 50);

    cv::imshow("Orijinal", img);
    cv::imshow("Custom Gray", customGray);
    cv::imshow("Custom YCrCb", customYCrCb);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}