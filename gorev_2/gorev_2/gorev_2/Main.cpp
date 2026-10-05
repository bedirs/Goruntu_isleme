#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

int main() {
    // 1. Düşük kontrastlı görüntüyü tek kanallı (gri) olarak aç
    std::string dosyaYolu = "resim.jpg";
    cv::Mat img = cv::imread(dosyaYolu, cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cerr << "Hata: '" << dosyaYolu << "' dosyasi bulunamadi!" << std::endl;
        std::cerr << "Lutfen resmi Main.cpp dosyasinin yanina koydugunuzdan emin olun." << std::endl;
        std::cin.get();
        return -1;
    }

    int rows = img.rows;
    int cols = img.cols;
    int totalPixels = rows * cols;

    // 2. ADIM: 256 Elemanlı Histogramı Çıkar (Hazır fonksiyon yok)
    std::vector<int> histogram(256, 0);

    for (int r = 0; r < rows; ++r) {
        const uchar* row_ptr = img.ptr<uchar>(r);
        for (int c = 0; c < cols; ++c) {
            uchar val = row_ptr[c];
            histogram[val]++;
        }
    }

    // 3. ADIM: Kümülatif Dağılım Fonksiyonunu (CDF) Hesapla (Hazır fonksiyon yok)
    std::vector<int> cdf(256, 0);
    cdf[0] = histogram[0];
    for (int i = 1; i < 256; ++i) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    // CDF_min değerini bul (sıfırdan büyük olan ilk kümülatif değer)
    int cdf_min = 0;
    for (int i = 0; i < 256; ++i) {
        if (cdf[i] > 0) {
            cdf_min = cdf[i];
            break;
        }
    }

    std::cout << "==========================================" << std::endl;
    std::cout << "Goruntu Boyutu       : " << cols << "x" << rows << " (" << totalPixels << " piksel)" << std::endl;
    std::cout << "Ilk Sifir Olmayan CDF: " << cdf_min << std::endl;
    std::cout << "Son CDF Degeri       : " << cdf[255] << " (Toplam pikselle ayni olmali)" << std::endl;
    std::cout << "==========================================" << std::endl;

    // 4. ADIM: Eşitleme Dönüşüm Tablosu (Lookup Table - LUT) Oluştur
    // Formül: round( ((CDF[k] - CDF_min) / (Total - CDF_min)) * 255 )
    std::vector<uchar> lut(256, 0);
    float payda = static_cast<float>(totalPixels - cdf_min);

    if (payda <= 0.0f) {
        std::cout << "Tum pikseller ayni renkte, histogram esitleme yapilamaz." << std::endl;
        return 0;
    }

    for (int i = 0; i < 256; ++i) {
        if (cdf[i] < cdf_min) {
            lut[i] = 0;
        }
        else {
            float normalized = ((cdf[i] - cdf_min) / payda) * 255.0f;
            lut[i] = cv::saturate_cast<uchar>(std::round(normalized));
        }
    }

    // 5. ADIM: Pikselleri Dönüştürerek Yeni Resmi Oluştur (Hazır fonksiyon yok)
    cv::Mat equalizedImg = cv::Mat::zeros(img.size(), img.type());

    for (int r = 0; r < rows; ++r) {
        const uchar* src_row = img.ptr<uchar>(r);
        uchar* dst_row = equalizedImg.ptr<uchar>(r);

        for (int c = 0; c < cols; ++c) {
            dst_row[c] = lut[src_row[c]];
        }
    }

    // 6. ADIM: Çıktı Resmini Kaydet
    std::string ciktiYolu = "histogram_esitlenmis_sonuc.jpg";
    cv::imwrite(ciktiYolu, equalizedImg);
    std::cout << "Sonuc resmi basariyla kaydedildi: " << ciktiYolu << std::endl;

    // 7. ADIM: Sonuçları Ekranda Göster
    int winW = 380;
    int winH = (380 * rows) / cols;

    cv::namedWindow("1. Orijinal Dusuk Kontrast", cv::WINDOW_NORMAL);
    cv::resizeWindow("1. Orijinal Dusuk Kontrast", winW, winH);
    cv::moveWindow("1. Orijinal Dusuk Kontrast", 100, 100);
    cv::imshow("1. Orijinal Dusuk Kontrast", img);

    cv::namedWindow("2. Manuel Histogram Esitlenmis", cv::WINDOW_NORMAL);
    cv::resizeWindow("2. Manuel Histogram Esitlenmis", winW, winH);
    cv::moveWindow("2. Manuel Histogram Esitlenmis", 100 + winW + 30, 100);
    cv::imshow("2. Manuel Histogram Esitlenmis", equalizedImg);

    std::cout << "\nPencereleri kapatmak icin herhangi bir tusa basin..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}