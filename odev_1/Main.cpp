#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

int main() {
    std::string girisKlasoru = "odev_resimler";
    std::string cikisKlasoru = "islenen_resimler";

    if (!fs::exists(girisKlasoru)) {
        std::cerr << "Hata: '" << girisKlasoru << "' klasoru bulunamadi!" << std::endl;
        std::cerr << "Lutfen proje klasorunde '" << girisKlasoru << "' adinda bir klasor acip resimleri icine koyun." << std::endl;
        return -1;
    }

    if (!fs::exists(cikisKlasoru)) {
        fs::create_directory(cikisKlasoru);
        std::cout << "'" << cikisKlasoru << "' adli cikis klasoru otomatik olarak olusturuldu.\n" << std::endl;
    }

    int islenenSayisi = 0;

    for (const auto& entry : fs::directory_iterator(girisKlasoru)) {
        std::string dosyaYolu = entry.path().string();

        cv::Mat img = cv::imread(dosyaYolu);

        if (img.empty()) {
            continue;
        }

        cv::Mat resized_img;
        cv::resize(img, resized_img, cv::Size(1024, 768), 0, 0, cv::INTER_LINEAR);

        std::string cikisYolu = cikisKlasoru + "/" + entry.path().filename().string();
        cv::imwrite(cikisYolu, resized_img);

        std::cout << entry.path().filename().string() << " --> 1024x768 boyutuna getirildi." << std::endl;
        islenenSayisi++;
    }

    std::cout << "\nIslem basariyla bitti! Toplam " << islenenSayisi
        << " resim '" << cikisKlasoru << "' klasorune kaydedildi." << std::endl;

    return 0;
}