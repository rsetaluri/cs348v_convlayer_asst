#include <iostream>
#include "convolution_kernel.hpp"
#include "feature_map.hpp"
#include "simple_convolution_layer.hpp"

int main(int argc, char** argv) {
  typedef double T;

  if (argc < 4) {
    std::cout << "usage: " << argv[0] << " activations depthwise_weights"
              << " pointwise_weights" << std::endl;
    return 0;
  }
  const std::string activations_filename = argv[1];
  const std::string dw_weights_filename = argv[2];
  const std::string pw_weights_filename = argv[3];

  auto activations = FeatureMap<T>::New(activations_filename);
  if (not activations) {
    std::cout << "Could not read activations from file "
              << activations_filename << std::endl;
    return 0;
  }
  auto dw_kernel = ConvolutionKernel<T>::New(dw_weights_filename);
  if (not dw_kernel) {
    std::cout << "Could not read depthwise weights from file "
              << dw_weights_filename << std::endl;
    return 0;
  }
  // Check that dw_kernel is really depthwise.
  if (dw_kernel->n() != 1) {
    std::cout << "Depthwise kernel is not actually depthwise" << std::endl;
    return 0;
  }
  auto pw_kernel = ConvolutionKernel<T>::New(pw_weights_filename);
  if (not pw_kernel) {
    std::cout << "Could not read pointwise weights from file "
              << pw_weights_filename << std::endl;
    return 0;
  }
  // Check that pw_kernel is really pointwise.
  if (pw_kernel->width() != 1 || pw_kernel->height() != 1) {
    std::cout << "Pointwise kernel is not actually pointwise" << std::endl;
    return 0;
  }
  std::unique_ptr<ConvolutionLayer<T>> layer(
      new SimpleConvolutionLayer<T>(
          std::move(dw_kernel), std::move(pw_kernel)));
  auto output = layer->Run(*activations);
  std::cout << "width = " << output->width() << std::endl;
  std::cout << "height = " << output->height() << std::endl;
  std::cout << "channels = " << output->channels() << std::endl;
}
