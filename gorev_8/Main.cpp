#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

int main() {
    cv::Mat img = cv::imread("resim.jpeg");
    if (img.empty()) img = cv::imread("resim.jpg");
    if (img.empty()) {
        std::cerr << "Hata: resim dosyasi bulunamadi!" << std::endl;
        return -1;
    }

    std::vector<cv::Mat> bgrPlanes;
    cv::split(img, bgrPlanes);

    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = { range };

    cv::Mat bHist, gHist, rHist;

    cv::calcHist(&bgrPlanes[0], 1, 0, cv::Mat(), bHist, 1, &histSize, &histRange);
    cv::calcHist(&bgrPlanes[1], 1, 0, cv::Mat(), gHist, 1, &histSize, &histRange);
    cv::calcHist(&bgrPlanes[2], 1, 0, cv::Mat(), rHist, 1, &histSize, &histRange);

    int histW = 512;
    int histH = 400;
    int adim = cvRound((double)histW / histSize);

    cv::Mat histTuvali(histH, histW, CV_8UC3, cv::Scalar(20, 20, 20));

    cv::normalize(bHist, bHist, 0, histH - 20, cv::NORM_MINMAX);
    cv::normalize(gHist, gHist, 0, histH - 20, cv::NORM_MINMAX);
    cv::normalize(rHist, rHist, 0, histH - 20, cv::NORM_MINMAX);

    for (int i = 1; i < histSize; ++i) {
        cv::line(histTuvali,
            cv::Point(adim * (i - 1), histH - cvRound(bHist.at<float>(i - 1))),
            cv::Point(adim * i, histH - cvRound(bHist.at<float>(i))),
            cv::Scalar(255, 0, 0), 2);

        cv::line(histTuvali,
            cv::Point(adim * (i - 1), histH - cvRound(gHist.at<float>(i - 1))),
            cv::Point(adim * i, histH - cvRound(gHist.at<float>(i))),
            cv::Scalar(0, 255, 0), 2);

        cv::line(histTuvali,
            cv::Point(adim * (i - 1), histH - cvRound(rHist.at<float>(i - 1))),
            cv::Point(adim * i, histH - cvRound(rHist.at<float>(i))),
            cv::Scalar(0, 0, 255), 2);
    }

    std::string winResim = "1. Orijinal Resim";
    std::string winGrafik = "2. RGB Histogram Grafigi";

    cv::namedWindow(winResim, cv::WINDOW_NORMAL);
    cv::namedWindow(winGrafik, cv::WINDOW_NORMAL);

    cv::resizeWindow(winResim, 480, 360);
    cv::resizeWindow(winGrafik, 512, 400);

    cv::moveWindow(winResim, 50, 60);
    cv::moveWindow(winGrafik, 560, 60);

    cv::imshow(winResim, img);
    cv::imshow(winGrafik, histTuvali);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}