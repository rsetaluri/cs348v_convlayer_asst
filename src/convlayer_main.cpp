#include <iostream>
#include "convolution_kernel.hpp"
#include "feature_map.hpp"
#include "simple_convolution_layer.hpp"

int main(int argc, char** argv) {
  typedef double T;
  auto activations = FeatureMap<T>::New("activations.bin");
  if (not activations) {
    std::cout << "Could not read activations from file activations.bin" << std::endl;
    return 0;
  }
  auto kernel = ConvolutionKernel<T>::New("weights.bin");
  if (not kernel) {
    std::cout << "Could not read weights from file weights.bin" << std::endl;
    return 0;
  }
  std::unique_ptr<ConvolutionLayer<T>> layer(
      new SimpleConvolutionLayer<T>(std::move(kernel)));
  auto output = layer->Run(*activations);
  std::cout << "width = " << output->width() << std::endl;
  std::cout << "height = " << output->height() << std::endl;
  std::cout << "channels = " << output->channels() << std::endl;
}
