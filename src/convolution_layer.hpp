#ifndef CONVOLUTION_LAYER_HPP_
#define CONVOLUTION_LAYER_HPP_

#include "convolution_kernel.hpp"
#include "feature_map.hpp"

template<typename T> class ConvolutionLayer {
 public:
  ConvolutionLayer(
      std::unique_ptr<ConvolutionKernel<T>> dw_kernel,
      std::unique_ptr<ConvolutionKernel<T>> pw_kernel)
    : dw_kernel_(std::move(dw_kernel)),
      pw_kernel_(std::move(pw_kernel)) {}

  virtual std::unique_ptr<FeatureMap<T>> Run(const FeatureMap<T>& input) = 0;

 protected:
  std::unique_ptr<ConvolutionKernel<T>> dw_kernel_;
  std::unique_ptr<ConvolutionKernel<T>> pw_kernel_;
};

#endif  // CONVOLUTION_LAYER_HPP_
