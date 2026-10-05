#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>

// ============================================================================
// MANUEL CLAHE (Contrast Limited Adaptive Histogram Equalization) ALGORİTMASI
// ============================================================================
cv::Mat manualCLAHE(const cv::Mat& src, double clipLimitVal, int gridX, int gridY) {
    int width = src.cols;
    int height = src.rows;

    int tileW = width / gridX;
    int tileH = height / gridY;

    // Her blok için CDF (kümülatif dağılım) tablosu: gridY x gridX x 256
    std::vector<std::vector<std::vector<uchar>>> cdfTables(
        gridY, std::vector<std::vector<uchar>>(gridX, std::vector<uchar>(256, 0))
    );

    // 1. ADIM: Her karonun histogramını çıkar, Clip Limit uygula ve CDF hesapla
    for (int gy = 0; gy < gridY; ++gy) {
        for (int gx = 0; gx < gridX; ++gx) {
            int startX = gx * tileW;
            int startY = gy * tileH;
            int actualTileW = (gx == gridX - 1) ? (width - startX) : tileW;
            int actualTileH = (gy == gridY - 1) ? (height - startY) : tileH;
            int totalPixels = actualTileW * actualTileH;

            // Blok histogramını hesapla
            std::vector<int> hist(256, 0);
            for (int y = 0; y < actualTileH; ++y) {
                const uchar* ptr = src.ptr<uchar>(startY + y);
                for (int x = 0; x < actualTileW; ++x) {
                    hist[ptr[startX + x]]++;
                }
            }

            // Clip Limit eşiğini aşan pikselleri kırp ve kalanlara eşit dağıt
            int clipLimit = static_cast<int>(clipLimitVal * (totalPixels / 256.0));
            if (clipLimit < 1) clipLimit = 1;

            int clipped = 0;
            for (int i = 0; i < 256; ++i) {
                if (hist[i] > clipLimit) {
                    clipped += (hist[i] - clipLimit);
                    hist[i] = clipLimit;
                }
            }

            int redist = clipped / 256;
            int residual = clipped % 256;
            for (int i = 0; i < 256; ++i) {
                hist[i] += redist;
            }
            for (int i = 0; i < residual; ++i) {
                hist[i]++;
            }

            // CDF (Kümülatif Dağılım Fonksiyonu) ve [0, 255] normalizasyonu
            int sum = 0;
            for (int i = 0; i < 256; ++i) {
                sum += hist[i];
                float val = (static_cast<float>(sum) / totalPixels) * 255.0f;
                cdfTables[gy][gx][i] = cv::saturate_cast<uchar>(std::round(val));
            }
        }
    }

    // 2. ADIM: Blok sınırlarını yumuşatmak için Çift Doğrusal İnterpolasyon (Bilinear)
    cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC1);

    for (int y = 0; y < height; ++y) {
        uchar* dstPtr = dst.ptr<uchar>(y);
        const uchar* srcPtr = src.ptr<uchar>(y);

        float fy = (y - tileH / 2.0f) / static_cast<float>(tileH);
        int y1 = static_cast<int>(std::floor(fy));
        int y2 = y1 + 1;
        float ay = fy - y1;

        y1 = std::clamp(y1, 0, gridY - 1);
        y2 = std::clamp(y2, 0, gridY - 1);

        for (int x = 0; x < width; ++x) {
            float fx = (x - tileW / 2.0f) / static_cast<float>(tileW);
            int x1 = static_cast<int>(std::floor(fx));
            int x2 = x1 + 1;
            float ax = fx - x1;

            x1 = std::clamp(x1, 0, gridX - 1);
            x2 = std::clamp(x2, 0, gridX - 1);

            uchar val = srcPtr[x];

            uchar v11 = cdfTables[y1][x1][val];
            uchar v12 = cdfTables[y1][x2][val];
            uchar v21 = cdfTables[y2][x1][val];
            uchar v22 = cdfTables[y2][x2][val];

            float top = (1.0f - ax) * v11 + ax * v12;
            float bot = (1.0f - ax) * v21 + ax * v22;
            float res = (1.0f - ay) * top + ay * bot;

            dstPtr[x] = cv::saturate_cast<uchar>(std::round(res));
        }
    }

    return dst;
}

// ============================================================================
// MAIN
// ============================================================================
int main() {
    // Görsel dosya adı (proje klasörünüzdeki dosya adı ile aynı olmalı)
    std::string dosyaAdi = "resim.jpg";

    // 1. Resmi tek kanallı (gri) olarak aç
    cv::Mat img = cv::imread(dosyaAdi, cv::IMREAD_GRAYSCALE);

    if (img.empty()) {
        std::cerr << "Hata: '" << dosyaAdi << "' okunamadi!" << std::endl;
        std::cerr << "Lutfen resmin .cpp dosyasinin yaninda oldugundan emin olun." << std::endl;
        std::cin.get();
        return -1;
    }

    const double clipLimit = 2.0;
    const cv::Size gridSize(8, 8);

    // 2. OpenCV Hazır CLAHE
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, gridSize);
    cv::Mat opencvSonuc;
    clahe->apply(img, opencvSonuc);

    // 3. Sıfırdan Kodladığımız Manuel CLAHE
    cv::Mat manualSonuc = manualCLAHE(img, clipLimit, gridSize.width, gridSize.height);

    // 4. İki Yöntem Arasındaki Fark Haritası
    cv::Mat fark;
    cv::absdiff(opencvSonuc, manualSonuc, fark);
    cv::Scalar ortalamaFark = cv::mean(fark);

    std::cout << "========================================" << std::endl;
    std::cout << "Gorsel: " << dosyaAdi << " (" << img.cols << "x" << img.rows << ")" << std::endl;
    std::cout << "OpenCV CLAHE ile Manuel CLAHE ortalama piksel farki: "
        << ortalamaFark[0] << " / 255" << std::endl;
    std::cout << "========================================" << std::endl;

    // 5. Pencereleri ekrana sığacak boyutta yan yana yerleştir
    int winW = 340;
    int winH = 450; // Dikey formatta bir resim olduğu için 340x450 oranı uygundur

    auto pencereYap = [&](const std::string& baslik, const cv::Mat& m, int x, int y) {
        cv::namedWindow(baslik, cv::WINDOW_NORMAL);
        cv::resizeWindow(baslik, winW, winH);
        cv::moveWindow(baslik, x, y);
        cv::imshow(baslik, m);
        };

    pencereYap("1. Orijinal Sisli Yol (Gri)", img, 40, 50);
    pencereYap("2. OpenCV Hazir CLAHE", opencvSonuc, 400, 50);
    pencereYap("3. Manuel Kodlanan CLAHE", manualSonuc, 760, 50);
    pencereYap("4. Fark (x10 Guclendirilmis)", fark * 10, 1120, 50);

    // İsteğe bağlı sonuçları kaydet
    cv::imwrite("opencv_clahe_sonuc.jpg", opencvSonuc);
    cv::imwrite("manuel_clahe_sonuc.jpg", manualSonuc);

    std::cout << "\nPencereleri kapatmak icin herhangi bir tusa basin..." << std::endl;
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}