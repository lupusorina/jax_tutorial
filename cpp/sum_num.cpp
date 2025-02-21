#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>  // For std::accumulate

// Function to add two numbers
inline int add(int x, int y) {
    return x + y;
}

int main() {
    constexpr size_t SIZE = 100000000; // 100 thousand

    // Generate vectors
    std::vector<int> x_vector(SIZE), y_vector(SIZE);
    for (size_t i = 0; i < SIZE; ++i) {
        x_vector[i] = i;
        y_vector[i] = i;
    }

    // Measure time for element-wise addition using a loop
    auto start_1 = std::chrono::high_resolution_clock::now();
    std::vector<int> result_vectorized(SIZE);
    for (size_t i = 0; i < SIZE; ++i) {
        result_vectorized[i] = add(x_vector[i], y_vector[i]);
    }
    auto end_1 = std::chrono::high_resolution_clock::now();
    std::cout << "C++ Loop-based Summation Time taken: "
              << std::chrono::duration<double>(end_1 - start_1).count()
              << " seconds" << std::endl;

    // Measure time for standard library summation (std::accumulate)
    auto start_2 = std::chrono::high_resolution_clock::now();
    long long sum = 0;
    for (size_t i = 0; i < SIZE; ++i) {
        sum += x_vector[i] + y_vector[i];
    }
    auto end_2 = std::chrono::high_resolution_clock::now();
    std::cout << "C++ Standard Loop Summation Time taken: "
              << std::chrono::duration<double>(end_2 - start_2).count()
              << " seconds" << std::endl;

    return 0;
}
