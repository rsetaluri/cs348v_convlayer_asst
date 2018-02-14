#ifndef CONVOLUTION_KERNEL_HPP_
#define CONVOLUTION_KERNEL_HPP_

#include <cstring>
#include <memory>
#include <string>

template<typename T> class ConvolutionKernel {
 public:
  static std::unique_ptr<ConvolutionKernel> New(std::string filename) {
    FILE* f = fopen(filename.c_str(), "rb");
    if (not f) return nullptr;
    int width;
    int height;
    int m;
    int n;
    fread(&width, sizeof(width), 1, f);
    fread(&height, sizeof(height), 1, f);
    fread(&m, sizeof(m), 1, f);
    fread(&n, sizeof(n), 1, f);
    T* weights = new T[width * height * m * n];
    fread(weights, sizeof(T), width * height * m * n, f);
    return std::unique_ptr<ConvolutionKernel>(
	new ConvolutionKernel(width, height, m, n, weights));
  }
  ~ConvolutionKernel() { delete[] weights_; }

  int width() const { return width_; }
  int height() const { return height_; }
  int m() const { return m_; }
  int n() const { return n_; }

  // Weights is ordered in (i, j, k, l) major format.
  const T* weights() const { return weights_; }

 private:
  // Takes ownership of @weights.
  ConvolutionKernel(int width, int height, int m, int n, T* weights)
    : width_(width), height_(height), m_(m), n_(n), weights_(weights) {}

  int width_;
  int height_;
  int m_;
  int n_;
  T* weights_;  // owns.
};

#endif  // CONVOLUTION_KERNEL_HPP_
