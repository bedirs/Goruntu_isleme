#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <chrono>
#include <vector>

// =========================================================
// GÖREV 2: 1000x1000 BMP -> Grayscale ve Boyut Kıyaslama
// Gerekli Dosya: resim.bmp
// =========================================================
void gorev2_GrayscaleKiyasla() {
    cv::Mat renkli = cv::imread("resim.bmp", cv::IMREAD_COLOR);
    if (renkli.empty()) {
        std::cout << "HATA: resim.bmp bulunamadi!" << std::endl;
        return;
    }

    cv::Mat gri;
    cv::cvtColor(renkli, gri, cv::COLOR_BGR2GRAY);
    cv::imwrite("gri_sonuc.bmp", gri);

    long piksel = renkli.cols * renkli.rows;
    long renkliBoyut = (piksel * renkli.channels()) + 54;
    long griBoyut = (piksel * gri.channels()) + 54;

    std::cout << "Cozunurluk: " << renkli.cols << " x " << renkli.rows << std::endl;
    std::cout << "Renkli BMP (3 kanal): " << renkliBoyut << " Byte" << std::endl;
    std::cout << "Gri BMP    (1 kanal): " << griBoyut << " Byte" << std::endl;

    cv::imshow("Renkli", renkli);
    cv::imshow("Grayscale", gri);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

// =========================================================
// GÖREV 3: 8K BMP Oluşturma, Piksel / 2 ve Hassas Süre Ölçümü
// Gerekli Dosya: Otomatik olusturur (orijinal_8k.bmp)
// =========================================================
void gorev3_8K() {
    cv::Mat uretilen8K(4320, 7680, CV_8UC3, cv::Scalar(200, 160, 120));
    cv::imwrite("orijinal_8k.bmp", uretilen8K);
    std::cout << "8K BMP olusturuldu (~95 MB)." << std::endl;

    cv::Mat goruntu = cv::imread("orijinal_8k.bmp", cv::IMREAD_COLOR);
    if (goruntu.empty()) {
        std::cout << "HATA: orijinal_8k.bmp acilamadi!" << std::endl;
        return;
    }

    auto baslangic = std::chrono::high_resolution_clock::now();

    int toplamSutun = goruntu.cols * goruntu.channels();
    for (int y = 0; y < goruntu.rows; ++y) {
        uchar* satir = goruntu.ptr<uchar>(y);
        for (int x = 0; x < toplamSutun; ++x) {
            satir[x] = satir[x] / 2;
        }
    }

    auto bitis = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gecenSure = bitis - baslangic;

    cv::imwrite("duzenlenmis_8k.bmp", goruntu);
    std::cout << "Sadece piksel duzenleme suresi: " << gecenSure.count() << " ms" << std::endl;
}

// =========================================================
// GÖREV 4: RGB -> HSV, HSI (HLS), YUV, CIE Lab Dönüşümü
// Gerekli Dosya: goruntu.bmp
// =========================================================
void gorev4_RenkUzaylari() {
    cv::Mat goruntu = cv::imread("goruntu.bmp", cv::IMREAD_COLOR);
    if (goruntu.empty()) {
        std::cout << "HATA: goruntu.bmp bulunamadi!" << std::endl;
        return;
    }

    cv::Mat hsv, hsi, yuv, lab;
    cv::cvtColor(goruntu, hsv, cv::COLOR_BGR2HSV);
    cv::cvtColor(goruntu, hsi, cv::COLOR_BGR2HLS);
    cv::cvtColor(goruntu, yuv, cv::COLOR_BGR2YUV);
    cv::cvtColor(goruntu, lab, cv::COLOR_BGR2Lab);

    cv::imwrite("sonuc_hsv.bmp", hsv);
    cv::imwrite("sonuc_hsi.bmp", hsi);
    cv::imwrite("sonuc_yuv.bmp", yuv);
    cv::imwrite("sonuc_lab.bmp", lab);

    cv::imshow("Orijinal", goruntu);
    cv::imshow("HSV", hsv);
    cv::imshow("HSI / HLS", hsi);
    cv::imshow("YUV", yuv);
    cv::imshow("CIE Lab", lab);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

// =========================================================
// GÖREV 5: Ten Rengi Tespiti (Ten Dışı Kırmızı) - RGB, YCrCb, HSV
// Gerekli Dosya: deneme2.jpg
// =========================================================
void gorev5_TenRengiKirmizi() {
    cv::Mat goruntu = cv::imread("deneme2.jpg", cv::IMREAD_COLOR);
    if (goruntu.empty()) {
        std::cout << "HATA: deneme2.jpg bulunamadi!" << std::endl;
        return;
    }

    cv::Mat kirmiziZemin(goruntu.size(), goruntu.type(), cv::Scalar(0, 0, 255));

    // RGB (BGR)
    cv::Mat bgrMaske, bgrSonuc = kirmiziZemin.clone();
    cv::inRange(goruntu, cv::Scalar(0, 40, 90), cv::Scalar(50, 150, 255), bgrMaske);
    goruntu.copyTo(bgrSonuc, bgrMaske);

    // YCrCb
    cv::Mat ycrcb, ycrcbMaske, ycrcbSonuc = kirmiziZemin.clone();
    cv::cvtColor(goruntu, ycrcb, cv::COLOR_BGR2YCrCb);
    cv::inRange(ycrcb, cv::Scalar(0, 133, 77), cv::Scalar(255, 173, 127), ycrcbMaske);
    goruntu.copyTo(ycrcbSonuc, ycrcbMaske);

    // HSV
    cv::Mat hsv, hsvMaske, hsvSonuc = kirmiziZemin.clone();
    cv::cvtColor(goruntu, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, cv::Scalar(0, 20, 70), cv::Scalar(20, 255, 255), hsvMaske);
    goruntu.copyTo(hsvSonuc, hsvMaske);

    cv::namedWindow("BGR", cv::WINDOW_NORMAL);
    cv::namedWindow("YCrCb", cv::WINDOW_NORMAL);
    cv::namedWindow("HSV", cv::WINDOW_NORMAL);

    cv::resizeWindow("BGR", 450, 350);
    cv::resizeWindow("YCrCb", 450, 350);
    cv::resizeWindow("HSV", 450, 350);

    cv::imshow("BGR", bgrSonuc);
    cv::imshow("YCrCb", ycrcbSonuc);
    cv::imshow("HSV", hsvSonuc);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

// =========================================================
// HAFTALIK GÖREV 1: Web Kamerası + Hareketli ROI + Ten Rengi
// =========================================================
void haftalikGorev1_KameraROI() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "HATA: Web kamerasi acilamadi!" << std::endl;
        return;
    }

    cv::Mat kare, oncekiGri, simdikiGri, fark, hareketMaskesi;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

    cap >> kare;
    if (kare.empty()) return;
    cv::cvtColor(kare, oncekiGri, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(oncekiGri, oncekiGri, cv::Size(5, 5), 0);

    std::cout << "Kamera acildi. Cikmak icin ESC veya 'q' tusuna basin..." << std::endl;

    while (true) {
        cap >> kare;
        if (kare.empty()) break;

        cv::Mat sonuc = kare.clone();

        cv::cvtColor(kare, simdikiGri, cv::COLOR_BGR2GRAY);
        cv::GaussianBlur(simdikiGri, simdikiGri, cv::Size(5, 5), 0);

        cv::absdiff(oncekiGri, simdikiGri, fark);
        cv::threshold(fark, hareketMaskesi, 25, 255, cv::THRESH_BINARY);

        cv::morphologyEx(hareketMaskesi, hareketMaskesi, cv::MORPH_OPEN, kernel);
        cv::dilate(hareketMaskesi, hareketMaskesi, kernel, cv::Point(-1, -1), 2);

        if (cv::countNonZero(hareketMaskesi) > 1500) {
            // Hareketli pikselleri kapsayan ROI sınırlarını manuel hesapla
            int minX = hareketMaskesi.cols, minY = hareketMaskesi.rows, maxX = 0, maxY = 0;
            for (int y = 0; y < hareketMaskesi.rows; ++y) {
                const uchar* satir = hareketMaskesi.ptr<uchar>(y);
                for (int x = 0; x < hareketMaskesi.cols; ++x) {
                    if (satir[x] > 0) {
                        if (x < minX) minX = x;
                        if (x > maxX) maxX = x;
                        if (y < minY) minY = y;
                        if (y > maxY) maxY = y;
                    }
                }
            }

            if (maxX > minX && maxY > minY) {
                cv::Rect roiKutu(minX, minY, maxX - minX + 1, maxY - minY + 1);
                cv::Mat roiGoruntu = kare(roiKutu);
                cv::Mat roiYCrCb, roiTenMaskesi;

                cv::cvtColor(roiGoruntu, roiYCrCb, cv::COLOR_BGR2YCrCb);
                cv::inRange(roiYCrCb, cv::Scalar(0, 133, 77), cv::Scalar(255, 173, 127), roiTenMaskesi);

                sonuc(roiKutu).setTo(cv::Scalar(0, 255, 0), roiTenMaskesi);
                cv::rectangle(sonuc, roiKutu, cv::Scalar(255, 0, 0), 2);
            }
        }

        oncekiGri = simdikiGri.clone();

        cv::imshow("Hareket Maskesi", hareketMaskesi);
        cv::imshow("ROI Icinde Ten Tespiti", sonuc);

        char tus = (char)cv::waitKey(30);
        if (tus == 27 || tus == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
}

// =========================================================
// HAFTALIK GÖREV 2: Trafik İşaretleri (HSV + YCrCb + Hough)
// Gerekli Klasor: trafic_test
// =========================================================
void haftalikGorev2_TrafikIsaretleri() {
    std::vector<cv::String> dosyalar;
    cv::glob("trafic_test/*.*", dosyalar, false);

    if (dosyalar.empty()) {
        std::cout << "HATA: trafic_test klasoru bos veya bulunamadi!" << std::endl;
        return;
    }

    for (const auto& dosyaYolu : dosyalar) {
        cv::Mat goruntu = cv::imread(dosyaYolu, cv::IMREAD_COLOR);
        if (goruntu.empty()) continue;

        cv::Mat sonuc = goruntu.clone();

        // 1. HSV Kırmızı Maske
        cv::Mat hsv, maskeAlt, maskeUst, hsvKirmiziMaske;
        cv::cvtColor(goruntu, hsv, cv::COLOR_BGR2HSV);
        cv::inRange(hsv, cv::Scalar(0, 70, 50), cv::Scalar(10, 255, 255), maskeAlt);
        cv::inRange(hsv, cv::Scalar(170, 70, 50), cv::Scalar(180, 255, 255), maskeUst);
        cv::bitwise_or(maskeAlt, maskeUst, hsvKirmiziMaske);

        // 2. YCrCb Kırmızı Maske
        cv::Mat ycrcb, ycrcbKirmiziMaske;
        cv::cvtColor(goruntu, ycrcb, cv::COLOR_BGR2YCrCb);
        cv::inRange(ycrcb, cv::Scalar(0, 155, 0), cv::Scalar(255, 255, 135), ycrcbKirmiziMaske);

        // Kesişim ve Yumuşatma
        cv::Mat ortakKirmiziMaske;
        cv::bitwise_and(hsvKirmiziMaske, ycrcbKirmiziMaske, ortakKirmiziMaske);
        cv::GaussianBlur(ortakKirmiziMaske, ortakKirmiziMaske, cv::Size(9, 9), 2, 2);

        // 3. Hough Çember Dönüşümü
        std::vector<cv::Vec3f> cemberler;
        cv::HoughCircles(
            ortakKirmiziMaske, cemberler, cv::HOUGH_GRADIENT,
            1, ortakKirmiziMaske.rows / 8, 100, 30, 15, 200
        );

        std::cout << dosyaYolu << " -> Tespit edilen cember: " << cemberler.size() << std::endl;

        for (size_t i = 0; i < cemberler.size(); ++i) {
            cv::Point merkez(cvRound(cemberler[i][0]), cvRound(cemberler[i][1]));
            int yaricap = cvRound(cemberler[i][2]);
            cv::circle(sonuc, merkez, 3, cv::Scalar(255, 0, 0), -1);
            cv::circle(sonuc, merkez, yaricap, cv::Scalar(0, 255, 0), 3);
        }

        cv::imshow("1 - Kirmizi Maske", ortakKirmiziMaske);
        cv::imshow("2 - Hough Sonuc", sonuc);

        if (cv::waitKey(0) == 27) break;
    }

    cv::destroyAllWindows();
}

// =========================================================
// ANA MENÜ
// =========================================================
int main() {
    int secim = 0;

    while (true) {
        std::cout << "\n=== GORUNTU ISLEME ODEVLERI ===" << std::endl;
        std::cout << "2 - Gorev 2: 1000x1000 BMP -> Grayscale Boyut Kiyaslama" << std::endl;
        std::cout << "3 - Gorev 3: 8K BMP Olusturma, Piksel/2 ve Sure Olcumu" << std::endl;
        std::cout << "4 - Gorev 4: Renk Uzaylari Donusumu (HSV, HSI, YUV, Lab)" << std::endl;
        std::cout << "5 - Gorev 5: Ten Rengi Tespiti (RGB, YCrCb, HSV)" << std::endl;
        std::cout << "6 - Haftalik Gorev 1: Kamera + Hareketli ROI + Ten Rengi" << std::endl;
        std::cout << "7 - Haftalik Gorev 2: Trafik Isaretleri (Kirmizi Cember + Hough)" << std::endl;
        std::cout << "0 - Cikis" << std::endl;
        std::cout << "Seciminiz: ";
        std::cin >> secim;

        if (secim == 0) break;

        switch (secim) {
        case 2: gorev2_GrayscaleKiyasla(); break;
        case 3: gorev3_8K(); break;
        case 4: gorev4_RenkUzaylari(); break;
        case 5: gorev5_TenRengiKirmizi(); break;
        case 6: haftalikGorev1_KameraROI(); break;
        case 7: haftalikGorev2_TrafikIsaretleri(); break;
        default: std::cout << "Gecersiz secim!" << std::endl; break;
        }
    }

    return 0;
}