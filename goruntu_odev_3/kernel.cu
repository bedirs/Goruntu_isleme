#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

// warpers.hpp sorununu tamamen atlayan hafif OpenCV basliklari
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Hatasi: " << cudaGetErrorString(err) << " Satir: " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

__device__ inline uchar gpuClamp(float v) {
    float r = roundf(v);
    if (r > 255.0f) return 255;
    if (r < 0.0f) return 0;
    return (uchar)r;
}

// Gorev 1 & 2: Histogram CUDA Kernel
__global__ void histogramKernel(const uchar* d_channel, int* d_hist, int totalPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalPixels) {
        uchar val = d_channel[idx];
        atomicAdd(&d_hist[val], 1);
    }
}

// Gorev 3 & 4: BGR -> Gray CUDA Kernel
__global__ void bgrToGrayKernel(const uchar3* d_bgr, uchar* d_gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 p = d_bgr[idx];
        int yVal = (p.z * 4899 + p.y * 9617 + p.x * 1868 + (1 << 13)) >> 14;
        d_gray[idx] = (uchar)(yVal > 255 ? 255 : (yVal < 0 ? 0 : yVal));
    }
}

// Gorev 4: BGR -> YCrCb CUDA Kernel
__global__ void bgrToYCrCbKernel(const uchar3* d_bgr, uchar3* d_ycrcb, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 p = d_bgr[idx];

        float b = (float)p.x;
        float g = (float)p.y;
        float r = (float)p.z;

        float Y = 0.299f * r + 0.587f * g + 0.114f * b;
        float Cr = 128.0f + 0.5f * r - 0.418688f * g - 0.081312f * b;
        float Cb = 128.0f - 0.168736f * r - 0.331264f * g + 0.5f * b;

        d_ycrcb[idx] = make_uchar3(gpuClamp(Y), gpuClamp(Cr), gpuClamp(Cb));
    }
}

// Gorev 5: X-Ray Jet Sahte Renklendirme Kernel
__global__ void applyJetColorMapKernel(const uchar* d_gray, uchar3* d_colored, const uchar3* d_lut, int totalPixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalPixels) {
        uchar gVal = d_gray[idx];
        d_colored[idx] = d_lut[gVal];
    }
}

void runTask1And2(const cv::Mat& img) {
    std::cout << "\n=== GOREV 1 & 2: CUDA ile Histogram Hesabi ===" << std::endl;
    std::vector<cv::Mat> planes;
    cv::split(img, planes);

    int totalPixels = img.rows * img.cols;
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    uchar* d_channel = nullptr;
    int* d_hist = nullptr;

    CHECK_CUDA(cudaMalloc(&d_channel, totalPixels * sizeof(uchar)));
    CHECK_CUDA(cudaMalloc(&d_hist, 256 * sizeof(int)));

    for (int ch = 0; ch < 3; ++ch) {
        CHECK_CUDA(cudaMemcpy(d_channel, planes[ch].data, totalPixels * sizeof(uchar), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_hist, 0, 256 * sizeof(int)));

        histogramKernel << <blocksPerGrid, threadsPerBlock >> > (d_channel, d_hist, totalPixels);
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<int> h_hist(256, 0);
        CHECK_CUDA(cudaMemcpy(h_hist.data(), d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost));

        std::string kanalAdi = (ch == 0 ? "Mavi (B)" : (ch == 1 ? "Yesil (G)" : "Kirmizi (R)"));
        std::cout << kanalAdi << " Kanali Histogram Ilk 5 Ton: ";
        for (int i = 0; i < 5; ++i) std::cout << "[" << i << "]: " << h_hist[i] << " | ";
        std::cout << std::endl;
    }

    CHECK_CUDA(cudaFree(d_channel));
    CHECK_CUDA(cudaFree(d_hist));
}

void runTask3And4(const cv::Mat& img) {
    std::cout << "\n=== GOREV 3 & 4: CUDA Renk Donusumu ve Hata Analizi ===" << std::endl;
    int w = img.cols;
    int h = img.rows;
    int totalPixels = w * h;

    uchar3* d_bgr = nullptr;
    uchar* d_gray = nullptr;
    uchar3* d_ycrcb = nullptr;

    CHECK_CUDA(cudaMalloc(&d_bgr, totalPixels * sizeof(uchar3)));
    CHECK_CUDA(cudaMalloc(&d_gray, totalPixels * sizeof(uchar)));
    CHECK_CUDA(cudaMalloc(&d_ycrcb, totalPixels * sizeof(uchar3)));

    CHECK_CUDA(cudaMemcpy(d_bgr, img.data, totalPixels * sizeof(uchar3), cudaMemcpyHostToDevice));

    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);

    bgrToGrayKernel << <gridSize, blockSize >> > (d_bgr, d_gray, w, h);
    CHECK_CUDA(cudaDeviceSynchronize());

    cv::Mat cudaGray(h, w, CV_8UC1);
    CHECK_CUDA(cudaMemcpy(cudaGray.data, d_gray, totalPixels * sizeof(uchar), cudaMemcpyDeviceToHost));

    cv::Mat cvGray;
    cv::cvtColor(img, cvGray, cv::COLOR_BGR2GRAY);

    double grayError = 0.0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            grayError += std::abs((int)cudaGray.at<uchar>(y, x) - (int)cvGray.at<uchar>(y, x));
        }
    }
    std::cout << "BGR -> Gray Toplam Mutlak Hata (CUDA vs OpenCV): " << grayError << std::endl;

    bgrToYCrCbKernel << <gridSize, blockSize >> > (d_bgr, d_ycrcb, w, h);
    CHECK_CUDA(cudaDeviceSynchronize());

    cv::Mat cudaYCrCb(h, w, CV_8UC3);
    CHECK_CUDA(cudaMemcpy(cudaYCrCb.data, d_ycrcb, totalPixels * sizeof(uchar3), cudaMemcpyDeviceToHost));

    cv::Mat cvYCrCb;
    cv::cvtColor(img, cvYCrCb, cv::COLOR_BGR2YCrCb);

    double ycrcbError = 0.0;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            cv::Vec3b c1 = cudaYCrCb.at<cv::Vec3b>(y, x);
            cv::Vec3b c2 = cvYCrCb.at<cv::Vec3b>(y, x);
            ycrcbError += std::abs((int)c1[0] - (int)c2[0]) + std::abs((int)c1[1] - (int)c2[1]) + std::abs((int)c1[2] - (int)c2[2]);
        }
    }
    std::cout << "BGR -> YCrCb Toplam Mutlak Hata (CUDA vs OpenCV): " << ycrcbError << std::endl;

    CHECK_CUDA(cudaFree(d_bgr));
    CHECK_CUDA(cudaFree(d_gray));
    CHECK_CUDA(cudaFree(d_ycrcb));
}

void runTask5() {
    std::cout << "\n=== GOREV 5: CUDA ile X-Ray Sahte Renklendirme ===" << std::endl;

    std::vector<std::string> resimler = { "img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg" };

    cv::Mat dummy = cv::Mat::zeros(256, 1, CV_8UC1);
    for (int i = 0; i < 256; ++i) dummy.at<uchar>(i) = (uchar)i;
    cv::Mat lutMat;
    cv::applyColorMap(dummy, lutMat, cv::COLORMAP_JET);

    uchar3* d_lut = nullptr;
    CHECK_CUDA(cudaMalloc(&d_lut, 256 * sizeof(uchar3)));
    CHECK_CUDA(cudaMemcpy(d_lut, lutMat.data, 256 * sizeof(uchar3), cudaMemcpyHostToDevice));

    for (const auto& isim : resimler) {
        cv::Mat orijinal = cv::imread(isim);
        if (orijinal.empty()) continue;

        int w = orijinal.cols;
        int h = orijinal.rows;
        int totalPixels = w * h;

        uchar3* d_bgr = nullptr;
        uchar* d_gray = nullptr;
        uchar3* d_colored = nullptr;

        CHECK_CUDA(cudaMalloc(&d_bgr, totalPixels * sizeof(uchar3)));
        CHECK_CUDA(cudaMalloc(&d_gray, totalPixels * sizeof(uchar)));
        CHECK_CUDA(cudaMalloc(&d_colored, totalPixels * sizeof(uchar3)));

        CHECK_CUDA(cudaMemcpy(d_bgr, orijinal.data, totalPixels * sizeof(uchar3), cudaMemcpyHostToDevice));

        dim3 blockSize2D(16, 16);
        dim3 gridSize2D((w + blockSize2D.x - 1) / blockSize2D.x, (h + blockSize2D.y - 1) / blockSize2D.y);

        bgrToGrayKernel << <gridSize2D, blockSize2D >> > (d_bgr, d_gray, w, h);
        CHECK_CUDA(cudaDeviceSynchronize());

        int threads1D = 256;
        int blocks1D = (totalPixels + threads1D - 1) / threads1D;
        applyJetColorMapKernel << <blocks1D, threads1D >> > (d_gray, d_colored, d_lut, totalPixels);
        CHECK_CUDA(cudaDeviceSynchronize());

        cv::Mat gpuRenkli(h, w, CV_8UC3);
        CHECK_CUDA(cudaMemcpy(gpuRenkli.data, d_colored, totalPixels * sizeof(uchar3), cudaMemcpyDeviceToHost));

        cv::imshow("Orijinal X-Ray", orijinal);
        cv::imshow("CUDA Renklendirilmis (Jet)", gpuRenkli);
        std::cout << isim << " islendi. Sonraki resim icin bir tusa basin..." << std::endl;
        cv::waitKey(0);

        CHECK_CUDA(cudaFree(d_bgr));
        CHECK_CUDA(cudaFree(d_gray));
        CHECK_CUDA(cudaFree(d_colored));
    }

    CHECK_CUDA(cudaFree(d_lut));
    cv::destroyAllWindows();
}

int main() {
    cv::Mat testImg = cv::imread("resim.jpeg");
    if (testImg.empty()) testImg = cv::imread("resim.jpg");

    if (!testImg.empty()) {
        runTask1And2(testImg);
        runTask3And4(testImg);
    }
    else {
        std::cout << "resim.jpeg / resim.jpg bulunamadi, Gorev 1-4 icin test resmi ekleyin." << std::endl;
    }

    runTask5();

    return 0;
}