#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

int main() {
    std::vector<std::string> resimler = {
        "img1.jpg",
        "img2.jpg",
        "img3.jpg",
        "img4.jpg",
        "img5.jpg"
    };

    std::string winName = "X-Ray: Orijinal | Gri | Sahte Renklendirme (Jet)";
    cv::namedWindow(winName, cv::WINDOW_NORMAL);
    cv::resizeWindow(winName, 1200, 400);

    for (size_t i = 0; i < resimler.size(); ++i) {
        cv::Mat orijinal = cv::imread(resimler[i]);
        if (orijinal.empty()) {
            std::cerr << "Uyari: " << resimler[i] << " okunamadi, atlaniyor." << std::endl;
            continue;
        }

        cv::Mat gri;
        cv::cvtColor(orijinal, gri, cv::COLOR_BGR2GRAY);

        cv::Mat renklendirilmis;
        cv::applyColorMap(gri, renklendirilmis, cv::COLORMAP_JET);

        cv::Mat gri3Ch;
        cv::cvtColor(gri, gri3Ch, cv::COLOR_GRAY2BGR);

        cv::Size panelBoyut(400, 350);
        cv::Mat r1, r2, r3;
        cv::resize(orijinal, r1, panelBoyut);
        cv::resize(gri3Ch, r2, panelBoyut);
        cv::resize(renklendirilmis, r3, panelBoyut);

        cv::Mat birlesik;
        cv::hconcat(std::vector<cv::Mat>{r1, r2, r3}, birlesik);

        cv::putText(birlesik, "Orijinal ThreatsRGB", cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(birlesik, "Gri Ton", cv::Point(420, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
        cv::putText(birlesik, "Guvenlik Paleti (Jet)", cv::Point(820, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        cv::imshow(winName, birlesik);
        std::cout << "[" << (i + 1) << "/5] " << resimler[i] << " gosteriliyor. Sonraki resim icin bir tusa basin..." << std::endl;
        cv::waitKey(0);
    }

    cv::destroyAllWindows();
    return 0;
}