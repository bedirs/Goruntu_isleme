#include <opencv2/opencv.hpp>
#include <iostream>
#include <climits>
#include <cmath>
#include <chrono>

int hesaplaSAD(const cv::Mat& sol, const cv::Mat& sag, int x, int y, int d, int r) {
    int sad = 0;
    for (int dy = -r; dy <= r; ++dy) {
        const uchar* s1 = sol.ptr<uchar>(y + dy);
        const uchar* s2 = sag.ptr<uchar>(y + dy);
        for (int dx = -r; dx <= r; ++dx) {
            sad += std::abs(s1[x + dx] - s2[x - d + dx]);
        }
    }
    return sad;
}

cv::Mat seyrekStereo(const cv::Mat& sol, const cv::Mat& sag, const cv::Mat& kenarlar, int blokBoyutu, int maxDisparity, int& islenenPiksel) {
    int H = sol.rows;
    int W = sol.cols;
    int r = blokBoyutu / 2;
    islenenPiksel = 0;

    cv::Mat disparityMap = cv::Mat::zeros(H, W, CV_8UC1);

    for (int y = r; y < H - r; ++y) {
        const uchar* kenarSatir = kenarlar.ptr<uchar>(y);
        for (int x = r; x < W - r; ++x) {
            if (kenarSatir[x] == 0) continue;

            islenenPiksel++;
            int enKucukSad = INT_MAX;
            int enIyiD = 0;

            for (int d = 0; d < maxDisparity; ++d) {
                if (x - d - r < 0) break;

                int sad = hesaplaSAD(sol, sag, x, y, d, r);
                if (sad < enKucukSad) {
                    enKucukSad = sad;
                    enIyiD = d;
                }
            }
            disparityMap.at<uchar>(y, x) = static_cast<uchar>(enIyiD);
        }
    }
    return disparityMap;
}

cv::Mat yogunStereo(const cv::Mat& sol, const cv::Mat& sag, int blokBoyutu, int maxDisparity) {
    int H = sol.rows;
    int W = sol.cols;
    int r = blokBoyutu / 2;

    cv::Mat disparityMap = cv::Mat::zeros(H, W, CV_8UC1);

    for (int y = r; y < H - r; ++y) {
        for (int x = r; x < W - r; ++x) {
            int enKucukSad = INT_MAX;
            int enIyiD = 0;

            for (int d = 0; d < maxDisparity; ++d) {
                if (x - d - r < 0) break;

                int sad = hesaplaSAD(sol, sag, x, y, d, r);
                if (sad < enKucukSad) {
                    enKucukSad = sad;
                    enIyiD = d;
                }
            }
            disparityMap.at<uchar>(y, x) = static_cast<uchar>(enIyiD);
        }
    }
    return disparityMap;
}

int main() {
    cv::Mat solResim = cv::imread("scene1.row3.col1.ppm", cv::IMREAD_GRAYSCALE);
    cv::Mat sagResim = cv::imread("scene1.row3.col3.ppm", cv::IMREAD_GRAYSCALE);

    if (solResim.empty() || sagResim.empty()) {
        std::cout << "HATA: Goruntuler bulunamadi!" << std::endl;
        return -1;
    }

    int blokBoyutu = 7;
    int maxDisparity = 16;

    cv::Mat kenarlar;
    cv::Canny(solResim, kenarlar, 50, 150);

    int islenenKenar = 0;
    auto t1 = std::chrono::high_resolution_clock::now();
    cv::Mat hamSeyrek = seyrekStereo(solResim, sagResim, kenarlar, blokBoyutu, maxDisparity, islenenKenar);
    auto t2 = std::chrono::high_resolution_clock::now();
    double seyrekSure = std::chrono::duration<double, std::milli>(t2 - t1).count();

    t1 = std::chrono::high_resolution_clock::now();
    cv::Mat hamYogun = yogunStereo(solResim, sagResim, blokBoyutu, maxDisparity);
    t2 = std::chrono::high_resolution_clock::now();
    double yogunSure = std::chrono::duration<double, std::milli>(t2 - t1).count();

    int toplamPiksel = solResim.rows * solResim.cols;
    std::cout << "Toplam Piksel : " << toplamPiksel << std::endl;
    std::cout << "Kenar Pikseli : " << islenenKenar << " (%" << (100.0 * islenenKenar / toplamPiksel) << ")" << std::endl;
    std::cout << "Seyrek Sure   : " << seyrekSure << " ms" << std::endl;
    std::cout << "Yogun Sure    : " << yogunSure << " ms" << std::endl;
    std::cout << "Hiz Kazanci   : " << (yogunSure / seyrekSure) << "x daha hizli" << std::endl;

    cv::Mat normSeyrek, normYogun, renkSeyrek, renkYogun;
    cv::normalize(hamSeyrek, normSeyrek, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::normalize(hamYogun, normYogun, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::applyColorMap(normSeyrek, renkSeyrek, cv::COLORMAP_JET);
    cv::applyColorMap(normYogun, renkYogun, cv::COLORMAP_JET);

    renkSeyrek.setTo(cv::Scalar(0, 0, 0), kenarlar == 0);

    cv::imwrite("canny_kenarlar.png", kenarlar);
    cv::imwrite("seyrek_disparity.png", renkSeyrek);
    cv::imwrite("yogun_disparity.png", renkYogun);

    std::string p1 = "1 - Canny Kenarlar";
    std::string p2 = "2 - Seyrek (Sparse) Stereo";
    std::string p3 = "3 - Yogun (Dense) Stereo";

    cv::namedWindow(p1, cv::WINDOW_NORMAL);
    cv::namedWindow(p2, cv::WINDOW_NORMAL);
    cv::namedWindow(p3, cv::WINDOW_NORMAL);

    int w = 400, h = 300;
    cv::resizeWindow(p1, w, h); cv::moveWindow(p1, 40, 100);
    cv::resizeWindow(p2, w, h); cv::moveWindow(p2, 40 + w + 20, 100);
    cv::resizeWindow(p3, w, h); cv::moveWindow(p3, 40 + (w + 20) * 2, 100);

    cv::imshow(p1, kenarlar);
    cv::imshow(p2, renkSeyrek);
    cv::imshow(p3, renkYogun);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}