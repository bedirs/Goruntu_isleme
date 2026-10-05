#include <iostream>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

int main() {
    // 1. Resmi gri tonlamalı (tek kanallı) olarak oku
    // "resim.jpg" dosyasının Main.cpp ile aynı proje klasöründe olduğundan emin olun
    std::string dosyaYolu = "resim.jpg";
    cv::Mat img = cv::imread(dosyaYolu, cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cerr << "Hata: '" << dosyaYolu << "' dosyasi bulunamadi!" << std::endl;
        std::cerr << "Lutfen resmi proje klasorune koydugunuzdan emin olun." << std::endl;
        std::cin.get();
        return -1;
    }

    // 2. Resmin tamamını gezerek min ve max piksel değerlerini bul
    uchar min_val = 255;
    uchar max_val = 0;

    for (int r = 0; r < img.rows; ++r) {
        const uchar* row_ptr = img.ptr<uchar>(r);
        for (int c = 0; c < img.cols; ++c) {
            uchar pixel = row_ptr[c];
            if (pixel < min_val) min_val = pixel;
            if (pixel > max_val) max_val = pixel;
        }
    }

    std::cout << "==========================================" << std::endl;
    std::cout << "Gorsel Boyutu : " << img.cols << "x" << img.rows << std::endl;
    std::cout << "Orijinal Min Deger (I_min) : " << static_cast<int>(min_val) << std::endl;
    std::cout << "Orijinal Max Deger (I_max) : " << static_cast<int>(max_val) << std::endl;
    std::cout << "Mevcut Dinamik Aralık     : " << static_cast<int>(max_val - min_val) << std::endl;
    std::cout << "Hedef Dinamik Aralik       : [0 - 255]" << std::endl;
    std::cout << "==========================================" << std::endl;

    // Tüm pikseller aynıysa (örneğin tamamen gri bir resimse) sıfıra bölme hatasını engelle
    if (max_val == min_val) {
        std::cout << "Uyari: Tum pikseller ayni tonda, kontrast germe uygulanamaz." << std::endl;
        return 0;
    }

    // 3. Doğrusal Kontrast Germe İşlemi
    cv::Mat gerilmis = cv::Mat::zeros(img.size(), img.type());
    float range = static_cast<float>(max_val - min_val);

    for (int r = 0; r < img.rows; ++r) {
        const uchar* src_row = img.ptr<uchar>(r);
        uchar* dst_row = gerilmis.ptr<uchar>(r);

        for (int c = 0; c < img.cols; ++c) {
            // Formül: ((I - I_min) / (I_max - I_min)) * 255.0
            float normalDeger = (static_cast<float>(src_row[c] - min_val) / range) * 255.0f;

            // Yuvarlama ve 0-255 aralığına güvenli dönüştürme
            dst_row[c] = cv::saturate_cast<uchar>(std::round(normalDeger));
        }
    }

    // Gerilmiş resmin yeni min-max kontrolü (Doğrulama)
    uchar yeni_min = 255, yeni_max = 0;
    for (int r = 0; r < gerilmis.rows; ++r) {
        const uchar* row = gerilmis.ptr<uchar>(r);
        for (int c = 0; c < gerilmis.cols; ++c) {
            if (row[c] < yeni_min) yeni_min = row[c];
            if (row[c] > yeni_max) yeni_max = row[c];
        }
    }
    std::cout << "Gerilmis Yeni Min Deger: " << static_cast<int>(yeni_min) << std::endl;
    std::cout << "Gerilmis Yeni Max Deger: " << static_cast<int>(yeni_max) << std::endl;

    // 4. Pencereleri ekrana sığacak boyutta yan yana göster
    int winW = 380;
    int winH = (380 * img.rows) / img.cols;

    cv::namedWindow("1. Orijinal Dar Aralik", cv::WINDOW_NORMAL);
    cv::resizeWindow("1. Orijinal Dar Aralik", winW, winH);
    cv::moveWindow("1. Orijinal Dar Aralik", 100, 100);
    cv::imshow("1. Orijinal Dar Aralik", img);

    cv::namedWindow("2. Kontrasti Gerilmis [0-255]", cv::WINDOW_NORMAL);
    cv::resizeWindow("2. Kontrasti Gerilmis [0-255]", winW, winH);
    cv::moveWindow("2. Kontrasti Gerilmis [0-255]", 100 + winW + 30, 100);
    cv::imshow("2. Kontrasti Gerilmis [0-255]", gerilmis);

    // Sonucu diske kaydet
    cv::imwrite("kontrast_gerilmis_sonuc.jpg", gerilmis);

    std::cout << "\nPencereleri kapatmak icin herhangi bir tusa basin..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}