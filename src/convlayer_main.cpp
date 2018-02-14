#include <iostream>
#include "convolution_kernel.hpp"
#include "feature_map.hpp"
#include "simple_convolution_layer.hpp"

int main(int argc, char** argv) {
  typedef double T;

  if (argc < 3) {
    std::cout << "usage: " << argv[0] << " activations weights" << std::endl;
    return 0;
  }
  const std::string activations_filename = argv[1];
  const std::string weights_filename = argv[2];

  auto activations = FeatureMap<T>::New(activations_filename);
  if (not activations) {
    std::cout << "Could not read activations from file "
              << activations_filename << std::endl;
    return 0;
  }
  auto kernel = ConvolutionKernel<T>::New(weights_filename);
  if (not kernel) {
    std::cout << "Could not read weights from file "
              << weights_filename << std::endl;
    return 0;
  }
  std::unique_ptr<ConvolutionLayer<T>> layer(
      new SimpleConvolutionLayer<T>(std::move(kernel)));
  auto output = layer->Run(*activations);
  std::cout << "width = " << output->width() << std::endl;
  std::cout << "height = " << output->height() << std::endl;
  std::cout << "channels = " << output->channels() << std::endl;
}
