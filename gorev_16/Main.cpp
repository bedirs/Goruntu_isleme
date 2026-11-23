#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
#include <cmath>
#include <chrono>

int idx(int y, int x, int d, int W, int D) {
    return (y * W + x) * D + d;
}

std::vector<int> maliyetHacmiOlustur(const cv::Mat& sol, const cv::Mat& sag, int blokBoyutu, int D) {
    int H = sol.rows;
    int W = sol.cols;
    int r = blokBoyutu / 2;
    std::vector<int> C(H * W * D, 0);

    for (int y = r; y < H - r; ++y) {
        for (int x = r; x < W - r; ++x) {
            for (int d = 0; d < D; ++d) {
                if (x - d - r < 0) {
                    C[idx(y, x, d, W, D)] = 100000;
                    continue;
                }
                int sad = 0;
                for (int dy = -r; dy <= r; ++dy) {
                    const uchar* s1 = sol.ptr<uchar>(y + dy);
                    const uchar* s2 = sag.ptr<uchar>(y + dy);
                    for (int dx = -r; dx <= r; ++dx) {
                        sad += std::abs(s1[x + dx] - s2[x - d + dx]);
                    }
                }
                C[idx(y, x, d, W, D)] = sad;
            }
        }
    }
    return C;
}

void yolTopla(const std::vector<int>& C, std::vector<int>& S, int H, int W, int D, int dy, int dx, int P1, int P2) {
    std::vector<int> L(H * W * D, 0);

    int yBas = (dy >= 0) ? 0 : H - 1;
    int yBit = (dy >= 0) ? H : -1;
    int yAdim = (dy >= 0) ? 1 : -1;

    int xBas = (dx >= 0) ? 0 : W - 1;
    int xBit = (dx >= 0) ? W : -1;
    int xAdim = (dx >= 0) ? 1 : -1;

    for (int y = yBas; y != yBit; y += yAdim) {
        for (int x = xBas; x != xBit; x += xAdim) {
            int py = y - dy;
            int px = x - dx;

            if (py < 0 || py >= H || px < 0 || px >= W) {
                for (int d = 0; d < D; ++d) {
                    int val = C[idx(y, x, d, W, D)];
                    L[idx(y, x, d, W, D)] = val;
                    S[idx(y, x, d, W, D)] += val;
                }
                continue;
            }

            int minOnceki = INT_MAX;
            for (int d = 0; d < D; ++d) {
                minOnceki = std::min(minOnceki, L[idx(py, px, d, W, D)]);
            }

            for (int d = 0; d < D; ++d) {
                int c0 = L[idx(py, px, d, W, D)];
                int c1 = (d > 0) ? L[idx(py, px, d - 1, W, D)] + P1 : INT_MAX;
                int c2 = (d < D - 1) ? L[idx(py, px, d + 1, W, D)] + P1 : INT_MAX;
                int c3 = minOnceki + P2;

                int enKucuk = std::min({ c0, c1, c2, c3 });
                int val = C[idx(y, x, d, W, D)] + enKucuk - minOnceki;

                L[idx(y, x, d, W, D)] = val;
                S[idx(y, x, d, W, D)] += val;
            }
        }
    }
}

cv::Mat kendiSGBM(const cv::Mat& sol, const cv::Mat& sag, int blokBoyutu, int D, int P1, int P2) {
    int H = sol.rows;
    int W = sol.cols;

    std::vector<int> C = maliyetHacmiOlustur(sol, sag, blokBoyutu, D);
    std::vector<int> S(H * W * D, 0);

    yolTopla(C, S, H, W, D, 0, 1, P1, P2);
    yolTopla(C, S, H, W, D, 0, -1, P1, P2);
    yolTopla(C, S, H, W, D, 1, 0, P1, P2);
    yolTopla(C, S, H, W, D, -1, 0, P1, P2);

    cv::Mat disparityMap = cv::Mat::zeros(H, W, CV_8UC1);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            int enKucuk = INT_MAX;
            int enIyiD = 0;
            for (int d = 0; d < D; ++d) {
                int val = S[idx(y, x, d, W, D)];
                if (val < enKucuk) {
                    enKucuk = val;
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

    int blokBoyutu = 5;
    int maxDisparity = 16;
    int P1 = 8 * blokBoyutu * blokBoyutu;
    int P2 = 32 * blokBoyutu * blokBoyutu;

    auto t1 = std::chrono::high_resolution_clock::now();
    cv::Mat hamDisparity = kendiSGBM(solResim, sagResim, blokBoyutu, maxDisparity, P1, P2);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::cout << "Kendi SGBM suresi: "
        << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms" << std::endl;

    cv::Mat disparity8U, renkliDisparity;
    cv::normalize(hamDisparity, disparity8U, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(disparity8U, renkliDisparity, cv::COLORMAP_JET);

    cv::imwrite("kendi_sgbm_gri.png", disparity8U);
    cv::imwrite("kendi_sgbm_renkli.png", renkliDisparity);

    std::string p1 = "1 - Kendi SGBM (Gri)";
    std::string p2 = "2 - Kendi SGBM (Renkli)";

    cv::namedWindow(p1, cv::WINDOW_NORMAL);
    cv::namedWindow(p2, cv::WINDOW_NORMAL);

    cv::resizeWindow(p1, 450, 340); cv::moveWindow(p1, 100, 100);
    cv::resizeWindow(p2, 450, 340); cv::moveWindow(p2, 570, 100);

    cv::imshow(p1, disparity8U);
    cv::imshow(p2, renkliDisparity);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}