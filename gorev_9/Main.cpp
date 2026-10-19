#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

cv::Mat myCalcHist(const cv::Mat& tekKanal) {
    int h = tekKanal.rows;
    int w = tekKanal.cols;

    cv::Mat hist = cv::Mat::zeros(256, 1, CV_32F);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            uchar val = tekKanal.at<uchar>(y, x);
            hist.at<float>(val) += 1.0f;
        }
    }

    return hist;
}

int main() {
    cv::Mat img = cv::imread("resim.jpeg");
    if (img.empty()) img = cv::imread("resim.jpg");
    if (img.empty()) {
        std::cerr << "Hata: resim bulunamadi!" << std::endl;
        return -1;
    }

    std::vector<cv::Mat> bgrPlanes;
    cv::split(img, bgrPlanes);

    cv::Mat bHist = myCalcHist(bgrPlanes[0]);
    cv::Mat gHist = myCalcHist(bgrPlanes[1]);
    cv::Mat rHist = myCalcHist(bgrPlanes[2]);

    int histW = 512;
    int histH = 400;
    int adim = cvRound((double)histW / 256);

    cv::Mat histTuvali(histH, histW, CV_8UC3, cv::Scalar(20, 20, 20));

    cv::normalize(bHist, bHist, 0, histH - 20, cv::NORM_MINMAX);
    cv::normalize(gHist, gHist, 0, histH - 20, cv::NORM_MINMAX);
    cv::normalize(rHist, rHist, 0, histH - 20, cv::NORM_MINMAX);

    for (int i = 1; i < 256; ++i) {
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

    cv::namedWindow("Orijinal Resim", cv::WINDOW_NORMAL);
    cv::namedWindow("myCalcHist Grafigi", cv::WINDOW_NORMAL);

    cv::resizeWindow("Orijinal Resim", 480, 360);
    cv::resizeWindow("myCalcHist Grafigi", 512, 400);

    cv::moveWindow("Orijinal Resim", 50, 60);
    cv::moveWindow("myCalcHist Grafigi", 560, 60);

    cv::imshow("Orijinal Resim", img);
    cv::imshow("myCalcHist Grafigi", histTuvali);

    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}