#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

// ============================================================================
// 1. MANUEL 5x5 MEDYAN FİLTRESİ (Hazır fonksiyon kullanılmadan)
// ============================================================================
cv::Mat manualMedianFilter5x5(const cv::Mat& src) {
    int rows = src.rows;
    int cols = src.cols;
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC1);

    // 5x5 pencere için 25 elemanlı tampon dizi
    std::vector<uchar> window(25);

    // Yarıçap (ksize = 5 ise radius = 2)
    const int r_offset = 2;

    for (int r = 0; r < rows; ++r) {
        uchar* dstRow = dst.ptr<uchar>(r);

        for (int c = 0; c < cols; ++c) {
            int idx = 0;

            // 5x5 komşuluktaki pikselleri topla
            for (int kr = -r_offset; kr <= r_offset; ++kr) {
                // Sınır taşmalarını önlemek için kenarları yansıt/sınırla (clamp)
                int neighborR = std::clamp(r + kr, 0, rows - 1);
                const uchar* srcRow = src.ptr<uchar>(neighborR);

                for (int kc = -r_offset; kc <= r_offset; ++kc) {
                    int neighborC = std::clamp(c + kc, 0, cols - 1);
                    window[idx++] = srcRow[neighborC];
                }
            }

            // 25 elemanı sırala ve ortadaki elemanı (indis 12) medyan olarak al
            // std::nth_element tam sıralamadan çok daha hızlıdır (O(N) karmaşıklık)
            std::nth_element(window.begin(), window.begin() + 12, window.end());
            dstRow[c] = window[12];
        }
    }

    return dst;
}

// ============================================================================
// 2. RESİMDE GÜRÜLTÜ YOKSA SENTETİK SALT-AND-PEPPER EKLEME FONKSİYONU
// ============================================================================
cv::Mat addSaltAndPepperNoise(const cv::Mat& src, double noiseRatio = 0.10) {
    cv::Mat noisy = src.clone();
    std::mt19937 gen(1337);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int r = 0; r < noisy.rows; ++r) {
        uchar* row = noisy.ptr<uchar>(r);
        for (int c = 0; c < noisy.cols; ++c) {
            double rnd = dist(gen);
            if (rnd < noiseRatio / 2.0) {
                row[c] = 0;   // Biber (siyah nokta)
            }
            else if (rnd < noiseRatio) {
                row[c] = 255; // Tuz (beyaz nokta)
            }
        }
    }
    return noisy;
}

// ============================================================================
// MAIN FONKSİYONU
// ============================================================================
int main() {
    std::string dosyaYolu = "resim.jpg";
    cv::Mat inputImg = cv::imread(dosyaYolu, cv::IMREAD_GRAYSCALE);

    // Eğer resim bulunamazsa sentetik test resmi oluştur
    if (inputImg.empty()) {
        std::cout << "'" << dosyaYolu << "' bulunamadi, sentetik test resmi olusturuluyor...\n";
        inputImg = cv::Mat(350, 350, CV_8UC1);
        for (int r = 0; r < inputImg.rows; ++r) {
            uchar* row = inputImg.ptr<uchar>(r);
            for (int c = 0; c < inputImg.cols; ++c) {
                row[c] = static_cast<uchar>(80 + (r * 100) / inputImg.rows);
            }
        }
    }

    // 1. Görüntüye Salt-and-Pepper (Tuz ve Biber) gürültüsü ekle
    cv::Mat noisyImg = addSaltAndPepperNoise(inputImg, 0.12); // %12 oranında gürültü

    // 2. OpenCV hazır fonksiyonu ile 5x5 medyan filtreleme
    cv::Mat opencvMedian;
    cv::medianBlur(noisyImg, opencvMedian, 5);

    // 3. Kendi yazdığımız sıfırdan manuel 5x5 medyan filtreleme
    cv::Mat manualMedian = manualMedianFilter5x5(noisyImg);

    // 4. İki filtre arasındaki farkı test et
    cv::Mat diff;
    cv::absdiff(opencvMedian, manualMedian, diff);
    cv::Scalar meanDiff = cv::mean(diff);

    std::cout << "==================================================" << std::endl;
    std::cout << "Filtre Boyutu: 5x5 (25 piksel siralama)" << std::endl;
    std::cout << "OpenCV medianBlur ile Manuel Medyan farki (MAE): "
        << meanDiff[0] << " / 255" << std::endl;
    std::cout << "==================================================" << std::endl;

    // 5. Pencereleri ekrana sığacak şekilde düzenli yerleştir
    int winW = 340;
    int winH = (340 * noisyImg.rows) / noisyImg.cols;

    auto goster = [&](const std::string& baslik, const cv::Mat& m, int x, int y) {
        cv::namedWindow(baslik, cv::WINDOW_NORMAL);
        cv::resizeWindow(baslik, winW, winH);
        cv::moveWindow(baslik, x, y);
        cv::imshow(baslik, m);
        };

    goster("1. Gurultulu Resim (Salt-Pepper)", noisyImg, 40, 50);
    goster("2. OpenCV medianBlur (5x5)", opencvMedian, 400, 50);
    goster("3. Manuel Medyan Filtre (5x5)", manualMedian, 760, 50);
    goster("4. Fark Haritasi (x10)", diff * 10, 1120, 50);

    // Çıktıyı diske kaydet
    cv::imwrite("gurultulu_resim.jpg", noisyImg);
    cv::imwrite("manuel_medyan_sonuc.jpg", manualMedian);

    std::cout << "\nPencereleri kapatmak icin herhangi bir tusa basin..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}