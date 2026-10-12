#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4996)

#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>

int main() {
    // 1. Resmi gri tonlamalı oku
    cv::Mat img = cv::imread("resim.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) img = cv::imread("resim.jpeg", cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Hata: resim.jpg veya resim.jpeg bulunamadi!" << std::endl;
        return -1;
    }

    cv::TickMeter tm;

    // 2. Gauss ile hafif yumuşatma (3x3)
    cv::Mat gauss_img;
    cv::GaussianBlur(img, gauss_img, cv::Size(3, 3), 0);

    // =========================================================
    // 3. Prewitt Dikey
    // =========================================================
    float dataPrewittDikey[] = {
        -1, 0, 1,
        -1, 0, 1,
        -1, 0, 1
    };
    cv::Mat kernelDikey(3, 3, CV_32F, dataPrewittDikey);
    cv::Mat dikeySonuc;
    tm.start();
    cv::filter2D(gauss_img, dikeySonuc, CV_8U, kernelDikey);
    tm.stop();
    double timePrewittDikey = tm.getTimeMilli();
    tm.reset();

    // =========================================================
    // 4. Prewitt Yatay
    // =========================================================
    float dataPrewittYatay[] = {
        -1, -1, -1,
         0,  0,  0,
         1,  1,  1
    };
    cv::Mat kernelYatay(3, 3, CV_32F, dataPrewittYatay);
    cv::Mat yataySonuc;
    tm.start();
    cv::filter2D(gauss_img, yataySonuc, CV_8U, kernelYatay);
    tm.stop();
    double timePrewittYatay = tm.getTimeMilli();
    tm.reset();

    // =========================================================
    // 5. Sobel Dikey
    // =========================================================
    float dataSobelDikey[] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };
    cv::Mat sobelDikeyKernel(3, 3, CV_32F, dataSobelDikey);
    cv::Mat sobelDikey;
    tm.start();
    cv::filter2D(gauss_img, sobelDikey, CV_8U, sobelDikeyKernel);
    tm.stop();
    double timeSobelDikey = tm.getTimeMilli();
    tm.reset();

    // =========================================================
    // 6. Sobel Yatay
    // =========================================================
    float dataSobelYatay[] = {
        -1, -2, -1,
         0,  0,  0,
         1,  2,  1
    };
    cv::Mat sobelYatayKernel(3, 3, CV_32F, dataSobelYatay);
    cv::Mat sobelYatay;
    tm.start();
    cv::filter2D(gauss_img, sobelYatay, CV_8U, sobelYatayKernel);
    tm.stop();
    double timeSobelYatay = tm.getTimeMilli();
    tm.reset();

    // =========================================================
    // 7. Laplacian 3x3
    // =========================================================
    float dataLaplacian[] = {
        -1, -1, -1,
        -1,  8, -1,
        -1, -1, -1
    };
    cv::Mat kernelLaplacian(3, 3, CV_32F, dataLaplacian);
    cv::Mat laplacianHam, laplacianSonuc;
    tm.start();
    cv::filter2D(gauss_img, laplacianHam, CV_32F, kernelLaplacian);
    laplacianHam = cv::abs(laplacianHam);
    cv::normalize(laplacianHam, laplacianSonuc, 0, 255, cv::NORM_MINMAX, CV_8U);
    tm.stop();
    double timeLaplacian = tm.getTimeMilli();
    tm.reset();

    // =========================================================
    // 8. 7x7 Laplacian of Gaussian (LoG)
    // =========================================================
    float dataLoG7x7[] = {
         0,  0, -1, -1, -1,  0,  0,
         0, -2, -3, -3, -3, -2,  0,
        -1, -3,  5, 16,  5, -3, -1,
        -1, -3, 16, 40, 16, -3, -1,
        -1, -3,  5, 16,  5, -3, -1,
         0, -2, -3, -3, -3, -2,  0,
         0,  0, -1, -1, -1,  0,  0
    };
    cv::Mat kernelLoG7x7(7, 7, CV_32F, dataLoG7x7);
    cv::Mat logHam, logSonuc;
    tm.start();
    cv::filter2D(img, logHam, CV_32F, kernelLoG7x7);
    logHam = cv::abs(logHam);
    cv::normalize(logHam, logSonuc, 0, 255, cv::NORM_MINMAX, CV_8U);
    tm.stop();
    double timeLoG = tm.getTimeMilli();
    tm.reset();

    // =========================================================
    // 9. Canny Kenar Tespiti Algoritması
    // =========================================================
    cv::Mat cannySonuc;
    tm.start();
    // Alt eşik: 50, Üst eşik: 150 (Tuğla derzleri için ideal Hysteresis eşikleri)
    cv::Canny(gauss_img, cannySonuc, 50, 150, 3);
    tm.stop();
    double timeCanny = tm.getTimeMilli();
    tm.reset();

    // =========================================================
    // ÇALIŞMA SÜRELERİNİ KONSOLA YAZDIRMA
    // =========================================================
    std::cout << "\n================ CALISMA SURELERI (ms) ================" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "3. Prewitt Dikey        : " << timePrewittDikey << " ms" << std::endl;
    std::cout << "4. Prewitt Yatay        : " << timePrewittYatay << " ms" << std::endl;
    std::cout << "5. Sobel Dikey          : " << timeSobelDikey << " ms" << std::endl;
    std::cout << "6. Sobel Yatay          : " << timeSobelYatay << " ms" << std::endl;
    std::cout << "7. Laplacian 3x3        : " << timeLaplacian << " ms" << std::endl;
    std::cout << "8. LoG 7x7              : " << timeLoG << " ms" << std::endl;
    std::cout << "9. Canny Algoritmasi    : " << timeCanny << " ms" << std::endl;
    std::cout << "=======================================================\n" << std::endl;

    // Çıktıları diske kaydetme
    cv::imwrite("prewitt_dikey.jpg", dikeySonuc);
    cv::imwrite("prewitt_yatay.jpg", yataySonuc);
    cv::imwrite("sobel_dikey.jpg", sobelDikey);
    cv::imwrite("sobel_yatay.jpg", sobelYatay);
    cv::imwrite("laplacian_cikti.jpg", laplacianSonuc);
    cv::imwrite("log_7x7_cikti.jpg", logSonuc);
    cv::imwrite("canny_cikti.jpg", cannySonuc);

    // =========================================================
    // Pencereleri Oluşturma ve Grid Düzeni (3x3 Düzen)
    // =========================================================
    cv::namedWindow("1. Orijinal Gri", cv::WINDOW_NORMAL);
    cv::namedWindow("2. Gauss Yumusatilmis", cv::WINDOW_NORMAL);
    cv::namedWindow("3. Prewitt Dikey", cv::WINDOW_NORMAL);
    cv::namedWindow("4. Prewitt Yatay", cv::WINDOW_NORMAL);
    cv::namedWindow("5. Sobel Dikey", cv::WINDOW_NORMAL);
    cv::namedWindow("6. Sobel Yatay", cv::WINDOW_NORMAL);
    cv::namedWindow("7. Laplacian 3x3", cv::WINDOW_NORMAL);
    cv::namedWindow("8. LoG 7x7", cv::WINDOW_NORMAL);
    cv::namedWindow("9. Canny Kenar", cv::WINDOW_NORMAL);

    int w = 310, h = 200;
    cv::resizeWindow("1. Orijinal Gri", w, h);
    cv::resizeWindow("2. Gauss Yumusatilmis", w, h);
    cv::resizeWindow("3. Prewitt Dikey", w, h);
    cv::resizeWindow("4. Prewitt Yatay", w, h);
    cv::resizeWindow("5. Sobel Dikey", w, h);
    cv::resizeWindow("6. Sobel Yatay", w, h);
    cv::resizeWindow("7. Laplacian 3x3", w, h);
    cv::resizeWindow("8. LoG 7x7", w, h);
    cv::resizeWindow("9. Canny Kenar", w, h);

    // 1. Sıra (Y = 20)
    cv::moveWindow("1. Orijinal Gri", 20, 20);
    cv::moveWindow("2. Gauss Yumusatilmis", 350, 20);
    cv::moveWindow("3. Prewitt Dikey", 680, 20);

    // 2. Sıra (Y = 260)
    cv::moveWindow("4. Prewitt Yatay", 20, 260);
    cv::moveWindow("5. Sobel Dikey", 350, 260);
    cv::moveWindow("6. Sobel Yatay", 680, 260);

    // 3. Sıra (Y = 500)
    cv::moveWindow("7. Laplacian 3x3", 20, 500);
    cv::moveWindow("8. LoG 7x7", 350, 500);
    cv::moveWindow("9. Canny Kenar", 680, 500);

    // Görüntüleri aktarma
    cv::imshow("1. Orijinal Gri", img);
    cv::imshow("2. Gauss Yumusatilmis", gauss_img);
    cv::imshow("3. Prewitt Dikey", dikeySonuc);
    cv::imshow("4. Prewitt Yatay", yataySonuc);
    cv::imshow("5. Sobel Dikey", sobelDikey);
    cv::imshow("6. Sobel Yatay", sobelYatay);
    cv::imshow("7. Laplacian 3x3", laplacianSonuc);
    cv::imshow("8. LoG 7x7", logSonuc);
    cv::imshow("9. Canny Kenar", cannySonuc);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}