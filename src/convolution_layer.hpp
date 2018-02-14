#ifndef CONVOLUTION_LAYER_HPP_
#define CONVOLUTION_LAYER_HPP_

#include "convolution_kernel.hpp"
#include "feature_map.hpp"

template<typename T> class ConvolutionLayer {
 public:
  ConvolutionLayer(std::unique_ptr<ConvolutionKernel<T>> kernel)
    : kernel_(std::move(kernel)) {}

  virtual std::unique_ptr<FeatureMap<T>> Run(const FeatureMap<T>& input) = 0;

 protected:
  std::unique_ptr<ConvolutionKernel<T>> kernel_;
};

#endif  // CONVOLUTION_LAYER_HPP_
