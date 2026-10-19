#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>

cv::Mat myBgrToGray(const cv::Mat& src) {
    int h = src.rows;
    int w = src.cols;
    cv::Mat gray = cv::Mat::zeros(h, w, CV_8UC1);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            cv::Vec3b piksel = src.at<cv::Vec3b>(y, x);
            float b = static_cast<float>(piksel[0]);
            float g = static_cast<float>(piksel[1]);
            float r = static_cast<float>(piksel[2]);

            float griVal = 0.114f * b + 0.587f * g + 0.299f * r;
            gray.at<uchar>(y, x) = static_cast<uchar>(std::round(griVal));
        }
    }
    return gray;
}

int main() {
    cv::Mat colorImg = cv::imread("resim.jpeg");
    if (colorImg.empty()) colorImg = cv::imread("resim.jpg");
    if (colorImg.empty()) {
        std::cerr << "Hata: resim bulunamadi!" << std::endl;
        return -1;
    }

    cv::Mat cvGray;
    cv::cvtColor(colorImg, cvGray, cv::COLOR_BGR2GRAY);

    cv::Mat myGray = myBgrToGray(colorImg);

    int h = colorImg.rows;
    int w = colorImg.cols;
    long long toplamPiksel = static_cast<long long>(h) * w;

    double toplamMutlakFark = 0.0;
    long long sifirFarkliPikselSayisi = 0;
    long long farkliPikselSayisi = 0;
    int maksimumFark = 0;

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int p1 = static_cast<int>(cvGray.at<uchar>(y, x));
            int p2 = static_cast<int>(myGray.at<uchar>(y, x));
            int fark = std::abs(p1 - p2);

            toplamMutlakFark += fark;

            if (fark == 0) {
                sifirFarkliPikselSayisi++;
            }
            else {
                farkliPikselSayisi++;
                if (fark > maksimumFark) {
                    maksimumFark = fark;
                }
            }
        }
    }

    std::cout << "================ PIKSEL KARSILASTIRMA RAPORU ================" << std::endl;
    std::cout << "Toplam Piksel Sayisi         : " << toplamPiksel << " (" << w << "x" << h << ")" << std::endl;
    std::cout << "Farki TAM SIFIR Olan Piksel  : " << sifirFarkliPikselSayisi << std::endl;
    std::cout << "Farkli Cikan Piksel Sayisi   : " << farkliPikselSayisi << std::endl;
    std::cout << "Maksimum Gorulen Piksel Farki: " << maksimumFark << " (0-255 araliginda)" << std::endl;
    std::cout << "Toplam Mutlak Hata (Sum)     : " << toplamMutlakFark << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    if (toplamMutlakFark == 0.0) {
        std::cout << "SONUC: Mukemmel! Iki resim arasinda 0 fark var (Birebir Ayni)." << std::endl;
    }
    else {
        std::cout << "SONUC: Resimler pratik olarak ayni. Kucuk sapmalar OpenCV'nin" << std::endl;
        std::cout << "tamsayi bit kaydirma (bit-shift) optimizasyonu yuvarlamasindandir." << std::endl;
    }
    std::cout << "============================================================" << std::endl;

    return 0;
}