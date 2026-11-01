#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

const double PI = 3.14159265358979323846;

int main() {
    cv::Mat src = cv::imread("resim.jpg");
    if (src.empty()) {
        std::cerr << "resim.jpg bulunamadi!" << std::endl;
        return -1;
    }

    int w = src.cols;
    int h = src.rows;

    cv::Mat gray, edges, edgesFiltered;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 50, 150, 3);
    cv::medianBlur(edges, edgesFiltered, 3);

    cv::Mat imgHoughLines = src.clone();
    cv::Mat imgHoughLinesP = src.clone();

    std::vector<cv::Vec2f> lines;
    cv::HoughLines(edgesFiltered, lines, 1, CV_PI / 180, 120);
    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        double a = std::cos(theta), b = std::sin(theta);
        double x0 = a * rho, y0 = b * rho;
        cv::Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
        cv::Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
        cv::line(imgHoughLines, pt1, pt2, cv::Scalar(255, 0, 0), 2);
    }

    std::vector<cv::Vec4i> linesP;
    cv::HoughLinesP(edgesFiltered, linesP, 1, CV_PI / 180, 50, 40, 10);
    for (size_t i = 0; i < linesP.size(); i++) {
        cv::line(imgHoughLinesP, cv::Point(linesP[i][0], linesP[i][1]),
            cv::Point(linesP[i][2], linesP[i][3]), cv::Scalar(0, 0, 255), 2);
    }

    int maxRho = static_cast<int>(std::ceil(std::sqrt(w * w + h * h)));
    int numThetas = 180;

    std::vector<double> cosTable(numThetas);
    std::vector<double> sinTable(numThetas);
    for (int t = 0; t < numThetas; ++t) {
        double rad = t * PI / 180.0;
        cosTable[t] = std::cos(rad);
        sinTable[t] = std::sin(rad);
    }

    int rhoOffset = maxRho;
    std::vector<std::vector<int>> accumulator(2 * maxRho + 1, std::vector<int>(numThetas, 0));

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (edgesFiltered.at<uchar>(y, x) > 0) {
                for (int t = 0; t < numThetas; ++t) {
                    int rho = static_cast<int>(std::round(x * cosTable[t] + y * sinTable[t]));
                    accumulator[rho + rhoOffset][t]++;
                }
            }
        }
    }

    int maxVote = 0;
    for (int r = 0; r <= 2 * maxRho; ++r) {
        for (int t = 0; t < numThetas; ++t) {
            if (accumulator[r][t] > maxVote) maxVote = accumulator[r][t];
        }
    }

    cv::Mat accMat(2 * maxRho + 1, numThetas, CV_8UC1);
    for (int r = 0; r <= 2 * maxRho; ++r) {
        for (int t = 0; t < numThetas; ++t) {
            accMat.at<uchar>(r, t) = static_cast<uchar>((accumulator[r][t] * 255.0) / (maxVote == 0 ? 1 : maxVote));
        }
    }
    cv::imwrite("hough_line_oylama.png", accMat);

    cv::Mat imgManualLines = src.clone();
    int lineThreshold = 130;
    for (int r = 0; r <= 2 * maxRho; ++r) {
        for (int t = 0; t < numThetas; ++t) {
            if (accumulator[r][t] >= lineThreshold) {
                int realRho = r - rhoOffset;
                double a = cosTable[t], b = sinTable[t];
                double x0 = a * realRho, y0 = b * realRho;
                cv::Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
                cv::Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
                cv::line(imgManualLines, pt1, pt2, cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    cv::Mat grayBlur;
    cv::medianBlur(gray, grayBlur, 5);
    std::vector<cv::Vec3f> cvCircles;
    cv::HoughCircles(grayBlur, cvCircles, cv::HOUGH_GRADIENT, 1,
        grayBlur.rows / 8, 100, 30, 10, 150);

    cv::Mat imgCvCircles = src.clone();
    for (size_t i = 0; i < cvCircles.size(); i++) {
        cv::Point center(cvRound(cvCircles[i][0]), cvRound(cvCircles[i][1]));
        int radius = cvRound(cvCircles[i][2]);
        cv::circle(imgCvCircles, center, 3, cv::Scalar(0, 255, 0), -1);
        cv::circle(imgCvCircles, center, radius, cv::Scalar(0, 0, 255), 2);
    }

    int minR = 15;
    int maxR = 60;
    int numCircleThetas = 360;

    std::vector<double> circleCos(numCircleThetas);
    std::vector<double> circleSin(numCircleThetas);
    for (int t = 0; t < numCircleThetas; ++t) {
        double rad = t * PI / 180.0;
        circleCos[t] = std::cos(rad);
        circleSin[t] = std::sin(rad);
    }

    int rRange = maxR - minR + 1;
    std::vector<std::vector<std::vector<int>>> acc3D(
        h, std::vector<std::vector<int>>(w, std::vector<int>(rRange, 0)));

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            if (edgesFiltered.at<uchar>(y, x) > 0) {
                for (int r = minR; r <= maxR; ++r) {
                    int rIdx = r - minR;
                    for (int t = 0; t < numCircleThetas; t += 2) {
                        int a = static_cast<int>(std::round(x - r * circleCos[t]));
                        int b = static_cast<int>(std::round(y - r * circleSin[t]));
                        if (a >= 0 && a < w && b >= 0 && b < h) {
                            acc3D[b][a][rIdx]++;
                        }
                    }
                }
            }
        }
    }

    cv::Mat imgManualCircles = src.clone();
    cv::Mat imgConcentricCircles = src.clone();
    int circleThreshold = 140;
    int singleRadiusThreshold = 80;

    for (int b = 0; b < h; ++b) {
        for (int a = 0; a < w; ++a) {
            int esMerkezSayisi = 0;
            std::vector<int> esMerkezliYaricaplar;

            for (int rIdx = 0; rIdx < rRange; ++rIdx) {
                int vote = acc3D[b][a][rIdx];
                int r = rIdx + minR;

                if (vote >= circleThreshold) {
                    cv::circle(imgManualCircles, cv::Point(a, b), r, cv::Scalar(255, 0, 255), 2);
                    cv::circle(imgManualCircles, cv::Point(a, b), 2, cv::Scalar(0, 255, 0), -1);
                }

                if (vote >= singleRadiusThreshold) {
                    esMerkezSayisi++;
                    esMerkezliYaricaplar.push_back(r);
                }
            }

            if (esMerkezSayisi >= 2) {
                cv::circle(imgConcentricCircles, cv::Point(a, b), 3, cv::Scalar(0, 0, 255), -1);
                for (int r : esMerkezliYaricaplar) {
                    cv::circle(imgConcentricCircles, cv::Point(a, b), r, cv::Scalar(0, 255, 255), 2);
                }
            }
        }
    }

    const int winW = 380;
    const int winH = 260;
    const int gapX = 15;
    const int gapY = 40;
    const int startX = 20;
    const int startY = 20;

    std::vector<std::pair<std::string, cv::Mat>> pencereler = {
        { "1. Orijinal Resim", src },
        { "2. Canny ve Median Kenarlar", edgesFiltered },
        { "3. Hough Line Oylama Matrisi", accMat },
        { "4. Standart HoughLines", imgHoughLines },
        { "5. Probabilistic HoughLinesP", imgHoughLinesP },
        { "6. Manuel Hough Lines", imgManualLines },
        { "7. OpenCV Hough Circles", imgCvCircles },
        { "8. Manuel Hough Circles", imgManualCircles },
        { "9. Es Merkezli Cemberler", imgConcentricCircles }
    };

    for (size_t i = 0; i < pencereler.size(); ++i) {
        std::string baslik = pencereler[i].first;
        cv::Mat gosterilecekMatris = pencereler[i].second;

        int col = i % 3;
        int row = i / 3;

        int posX = startX + col * (winW + gapX);
        int posY = startY + row * (winH + gapY);

        cv::namedWindow(baslik, cv::WINDOW_NORMAL);
        cv::resizeWindow(baslik, winW, winH);
        cv::moveWindow(baslik, posX, posY);
        cv::imshow(baslik, gosterilecekMatris);
    }

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}