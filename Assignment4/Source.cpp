#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "mpi.h"

#define METADATA_COUNT  3

#define MASTER          0
#define FROM_MASTER     1
#define FROM_WORKER     2

#define RED     '\033[31m'
#define GREEN   '\033[32m'
#define BLUE    '\033[34m'

std::array<uchar, 3> average_pixel_RGB(const std::vector <uchar>& flat_image, size_t channels) {
    if (flat_image.empty() || channels < 3) {
        return { 0, 0, 0 };
    }

    uchar sumR = 0, sumG = 0, sumB = 0;
    std::array<uchar, 3> result = { sumR, sumG, sumB };

    size_t totalPixels = flat_image.size() / channels;

    for (size_t i = 0; i + 2 < flat_image.size(); i += channels) {
        sumR += flat_image[i];
        sumG += flat_image[i + 1];
        sumB += flat_image[i + 2];
    }

    return result;
}

void display_average_pixel_RGB(const std::array<uchar, 3> averages) {
    std::cout << "Red  : " << averages[0] << std::endl;
    std::cout << "Green: " << averages[1] << std::endl;
    std::cout << "Blue : " << averages[2] << std::endl;
}

void invert(int total_pixels, std::vector <uchar>& flat_image) {
    for (int i = 0; i < total_pixels; ++i) {
        flat_image[i] = 255 - flat_image[i];
    }
}

int main(int argc, char** argv) {
    
    // INITIALIZE 
    MPI_Status status{};
    int task_id, num_tasks, ip;
    int metadata[3] = {};
    int total_pixels;

    cv::Mat image;
    std::vector<uchar> flat_image;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);

    int num_workers = num_tasks - 1;

    // [MASTER] 
    // SETUP IMAGE & METADATA 
    if (task_id == MASTER) {
        image = cv::imread("input.jpg", cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Error loading image.\n";
            return 1;
        }

        metadata[0] = image.rows;
        metadata[1] = image.cols;
        metadata[2] = image.channels();
    }


    // [EVERYONE]
    // BROADCAST METADATA
    MPI_Bcast(&metadata, METADATA_COUNT, MPI_INT, MASTER, MPI_COMM_WORLD);

    total_pixels = metadata[0] * metadata[1] * metadata[2];

    // CHUNK & DISPLACE
    int chunk_size = total_pixels / num_tasks;
    int chunk_remainder = total_pixels % num_tasks;

    std::vector<int> chunk_sizes(num_tasks, chunk_size);

    for (int i = 0; i < chunk_remainder; i++)
        chunk_sizes[i]++;

    std::vector<int> displacements(num_tasks);
    displacements[0] = 0;
    for (int i = 1; i < num_tasks; i++)
        displacements[i] = displacements[i - 1] + chunk_sizes[i - 1];

    // SCATTER
    std::vector<uchar> flat_image(total_pixels);
    MPI_Scatter(image.data, chunk_sizes[task_id], MPI_UNSIGNED_CHAR,
        flat_image.data(), chunk_sizes[task_id], MPI_UNSIGNED_CHAR,
                MASTER, MPI_COMM_WORLD);

    // AVERAGE & INVERT
    average_pixel_RGB(flat_image, metadata[2]);

    invert(total_pixels, flat_image);
    
    // GATHER
    MPI_Gather(image.data, chunk_sizes[task_id], MPI_UNSIGNED_CHAR, 
               flat_image.data(), chunk_sizes[task_id], MPI_UNSIGNED_CHAR, 
               MASTER, MPI_COMM_WORLD);
 

    // [MASTER] RESULTS
    if (task_id == MASTER) {
        if (cv::imwrite("inverted_output.jpg", image))
            std::cout << "Completed - saved inverted_output.jpg" << std::endl;
        else
            std::cerr << "[ERROR] failed to write output image." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
