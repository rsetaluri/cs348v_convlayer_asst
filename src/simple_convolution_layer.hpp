#ifndef SIMPLE_CONVOLUTION_LAYER_HPP_
#define SIMPLE_CONVOLUTION_LAYER_HPP_

#include "convolution_layer.hpp"

template<typename T> class SimpleConvolutionLayer : public ConvolutionLayer<T> {
 public:
  SimpleConvolutionLayer(
      std::unique_ptr<ConvolutionKernel<T>> dw_kernel,
      std::unique_ptr<ConvolutionKernel<T>> pw_kernel)
    : ConvolutionLayer<T>(std::move(dw_kernel), std::move(pw_kernel)) {}

  std::unique_ptr<FeatureMap<T>> Run(const FeatureMap<T>& input) override {
    // TODO: Run convolution here.
    return input.Clone();
  }

 private:
  using ConvolutionLayer<T>::dw_kernel_;
  using ConvolutionLayer<T>::pw_kernel_;
};

#endif  // SIMPLE_CONVOLUTION_LAYER_HPP_
