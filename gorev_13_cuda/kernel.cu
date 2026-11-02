#include <iostream>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Hatasi: " << cudaGetErrorString(err) \
                  << " Satir: " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

const double PI = 3.14159265358979323846;

__global__ void houghLineKernel(
    const unsigned char* d_edges,
    int* d_accumulator,
    const float* d_cosTable,
    const float* d_sinTable,
    int width,
    int height,
    int maxRho,
    int numThetas)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;

        if (d_edges[idx] > 0) {
            int rhoOffset = maxRho;

            for (int t = 0; t < numThetas; ++t) {
                float rhoVal = x * d_cosTable[t] + y * d_sinTable[t];
                int rho = __float2int_rn(rhoVal);

                int accIdx = (rho + rhoOffset) * numThetas + t;
                atomicAdd(&d_accumulator[accIdx], 1);
            }
        }
    }
}

int main() {
    // .jpeg, .jpg, .png gibi tum yaygin uzantilari sirayla kontrol eder
    std::vector<std::string> adayDosyalar = {
        "resim.jpeg", "resim.jpg", "resim.png",
        "resim.JPEG", "resim.JPG"
    };

    cv::Mat src;
    std::string yuklenenDosya = "";

    for (const auto& dosyaAdi : adayDosyalar) {
        src = cv::imread(dosyaAdi);
        if (!src.empty()) {
            yuklenenDosya = dosyaAdi;
            break;
        }
    }

    if (src.empty()) {
        std::cerr << "Hata: Resim bulunamadi! Denenen dosyalar: resim.jpeg, resim.jpg" << std::endl;
        return -1;
    }

    int width = src.cols;
    int height = src.rows;

    cv::Mat gray, edges, edgesFiltered;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 50, 150, 3);
    cv::medianBlur(edges, edgesFiltered, 3);

    int maxRho = static_cast<int>(std::ceil(std::sqrt(width * width + height * height)));
    int numThetas = 180;
    int accRows = 2 * maxRho + 1;
    int accCols = numThetas;
    size_t accSize = accRows * accCols * sizeof(int);

    std::vector<float> h_cosTable(numThetas);
    std::vector<float> h_sinTable(numThetas);
    for (int t = 0; t < numThetas; ++t) {
        float rad = static_cast<float>(t * PI / 180.0);
        h_cosTable[t] = std::cos(rad);
        h_sinTable[t] = std::sin(rad);
    }

    unsigned char* d_edges = nullptr;
    int* d_accumulator = nullptr;
    float* d_cosTable = nullptr;
    float* d_sinTable = nullptr;

    size_t imgSize = width * height * sizeof(unsigned char);
    CHECK_CUDA(cudaMalloc(&d_edges, imgSize));
    CHECK_CUDA(cudaMalloc(&d_accumulator, accSize));
    CHECK_CUDA(cudaMalloc(&d_cosTable, numThetas * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_sinTable, numThetas * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_edges, edgesFiltered.data, imgSize, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_cosTable, h_cosTable.data(), numThetas * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sinTable, h_sinTable.data(), numThetas * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemset(d_accumulator, 0, accSize));

    dim3 blockSize(16, 16);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    houghLineKernel << <gridSize, blockSize >> > (
        d_edges, d_accumulator, d_cosTable, d_sinTable,
        width, height, maxRho, numThetas
        );
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int> h_accumulator(accRows * accCols, 0);
    CHECK_CUDA(cudaMemcpy(h_accumulator.data(), d_accumulator, accSize, cudaMemcpyDeviceToHost));

    int maxVote = 0;
    for (int val : h_accumulator) {
        if (val > maxVote) maxVote = val;
    }

    cv::Mat accMat(accRows, accCols, CV_8UC1);
    for (int r = 0; r < accRows; ++r) {
        for (int t = 0; t < accCols; ++t) {
            int vote = h_accumulator[r * accCols + t];
            accMat.at<uchar>(r, t) = static_cast<uchar>((vote * 255.0) / (maxVote == 0 ? 1 : maxVote));
        }
    }
    cv::imwrite("cuda_hough_line_oylama.png", accMat);

    cv::Mat sonuc = src.clone();
    int threshold = 130;
    int rhoOffset = maxRho;

    for (int r = 0; r < accRows; ++r) {
        for (int t = 0; t < accCols; ++t) {
            int vote = h_accumulator[r * accCols + t];
            if (vote >= threshold) {
                int realRho = r - rhoOffset;
                double a = h_cosTable[t];
                double b = h_sinTable[t];
                double x0 = a * realRho;
                double y0 = b * realRho;

                cv::Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
                cv::Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));

                cv::line(sonuc, pt1, pt2, cv::Scalar(0, 255, 0), 2);
            }
        }
    }

    CHECK_CUDA(cudaFree(d_edges));
    CHECK_CUDA(cudaFree(d_accumulator));
    CHECK_CUDA(cudaFree(d_cosTable));
    CHECK_CUDA(cudaFree(d_sinTable));

    // Pencereleri yan yana yerlestirme
    const int winW = 420;
    const int winH = 320;
    const int startX = 30;
    const int startY = 40;
    const int gapX = 20;

    std::vector<std::pair<std::string, cv::Mat>> pencereler = {
        { "1. Canny ve Median Kenarlar", edgesFiltered },
        { "2. CUDA Hough Line Oylama Matrisi", accMat },
        { "3. CUDA ile Tespit Edilen Dogrular", sonuc }
    };

    for (size_t i = 0; i < pencereler.size(); ++i) {
        std::string baslik = pencereler[i].first;
        cv::Mat gosterilecekMatris = pencereler[i].second;

        cv::namedWindow(baslik, cv::WINDOW_NORMAL);
        cv::resizeWindow(baslik, winW, winH);
        cv::moveWindow(baslik, startX + static_cast<int>(i) * (winW + gapX), startY);
        cv::imshow(baslik, gosterilecekMatris);
    }

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}