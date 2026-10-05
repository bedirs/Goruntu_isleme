#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

// ============================================================================
// 1. MANUEL 3x3 KONVOLÜSYON ORTALAMA (MEAN / BOX) FİLTRESİ
// ============================================================================
cv::Mat manualBoxFilter3x3(const cv::Mat& src) {
    int rows = src.rows;
    int cols = src.cols;
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC1);

    // 3x3 Çekirdek katsayıları (tüm elemanlar 1, toplam ağırlık 9)
    // Sınır taşmalarını önlemek için kenarları clamp (en yakın geçerli piksel) ile koruyoruz
    for (int r = 0; r < rows; ++r) {
        uchar* dstRow = dst.ptr<uchar>(r);

        for (int c = 0; c < cols; ++c) {
            int sum = 0;

            // 3x3 çekirdeği (kernel) piksel etrafında gezdir
            for (int kr = -1; kr <= 1; ++kr) {
                int neighborR = std::clamp(r + kr, 0, rows - 1);
                const uchar* srcRow = src.ptr<uchar>(neighborR);

                for (int kc = -1; kc <= 1; ++kc) {
                    int neighborC = std::clamp(c + kc, 0, cols - 1);
                    sum += srcRow[neighborC];
                }
            }

            // 9 komşunun aritmetik ortalamasını al ve yuvarla
            dstRow[c] = cv::saturate_cast<uchar>(std::round(sum / 9.0f));
        }
    }

    return dst;
}

// ============================================================================
// 2. GÖRÜNTÜYE SENTETİK GÜRÜLTÜ EKLEME (Tuz-Biber & Gauss)
// ============================================================================
cv::Mat addNoise(const cv::Mat& src, double saltPepperRatio = 0.05) {
    cv::Mat noisy = src.clone();
    std::mt19937 gen(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int r = 0; r < noisy.rows; ++r) {
        uchar* row = noisy.ptr<uchar>(r);
        for (int c = 0; c < noisy.cols; ++c) {
            double rnd = dist(gen);
            if (rnd < saltPepperRatio / 2.0) {
                row[c] = 0;   // Biber (siyah gürültü)
            }
            else if (rnd < saltPepperRatio) {
                row[c] = 255; // Tuz (beyaz gürültü)
            }
        }
    }
    return noisy;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    // Proje klasöründeki resmi oku (bulunamazsa otomatik test görüntüsü oluşturur)
    std::string dosyaYolu = "resim.jpg";
    cv::Mat original = cv::imread(dosyaYolu, cv::IMREAD_GRAYSCALE);

    if (original.empty()) {
        std::cout << "'" << dosyaYolu << "' bulunamadi, sentetik test goruntusu olusturuluyor...\n";
        original = cv::Mat(400, 400, CV_8UC1);
        for (int r = 0; r < original.rows; ++r) {
            uchar* row = original.ptr<uchar>(r);
            for (int c = 0; c < original.cols; ++c) {
                row[c] = static_cast<uchar>((r / 40 % 2 == c / 40 % 2) ? 190 : 60);
            }
        }
    }

    // 1. Resme gürültü bulaştır
    cv::Mat noisy = addNoise(original, 0.08); // %8 oranında tuz-biber gürültüsü

    // 2. Kendi yazdığımız manuel 3x3 konvolüsyon filtresini uygula
    cv::Mat filteredManual = manualBoxFilter3x3(noisy);

    // 3. Doğrulama için OpenCV'nin hazır fonksiyonu (cv::blur)
    cv::Mat filteredOpenCV;
    cv::blur(noisy, filteredOpenCV, cv::Size(3, 3), cv::Point(-1, -1), cv::BORDER_REPLICATE);

    // 4. İki filtreleme arasındaki farkı hesapla
    cv::Mat diff;
    cv::absdiff(filteredManual, filteredOpenCV, diff);
    cv::Scalar meanDiff = cv::mean(diff);

    std::cout << "========================================\n";
    std::cout << "Gorsel Boyutu: " << original.cols << "x" << original.rows << "\n";
    std::cout << "Manuel Konvolusyon ile cv::blur() farki (MAE): " << meanDiff[0] << " / 255\n";
    std::cout << "========================================\n";

    // 5. Pencereleri ekrana düzgün yerleştir
    int winW = 340, winH = (340 * original.rows) / original.cols;
    auto goster = [&](const std::string& title, const cv::Mat& m, int x, int y) {
        cv::namedWindow(title, cv::WINDOW_NORMAL);
        cv::resizeWindow(title, winW, winH);
        cv::moveWindow(title, x, y);
        cv::imshow(title, m);
        };

    goster("1. Orijinal Goruntu", original, 40, 50);
    goster("2. Gurultulu Goruntu (Tuz-Biber)", noisy, 400, 50);
    goster("3. Manuel 3x3 Filtre (Bastirilmis)", filteredManual, 760, 50);
    goster("4. OpenCV blur() Sonucu", filteredOpenCV, 1120, 50);

    // İsteğe bağlı sonuçları kaydet
    cv::imwrite("gurultulu.jpg", noisy);
    cv::imwrite("manuel_filtre_sonuc.jpg", filteredManual);

    std::cout << "\nPencereleri kapatmak icin herhangi bir tusa basin...\n";
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}