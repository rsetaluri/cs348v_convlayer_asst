#ifndef HALIDE_CONVOLUTION_LAYER_HPP_
#define HALIDE_CONVOLUTION_LAYER_HPP_

#include "convolution_layer.hpp"

class HalideConvolutionLayer : public ConvolutionLayer {
 public:
  HalideConvolutionLayer() = default;

  void Init(Parameters params) override;
  void Run(Parameters params, Data data) override;
};

#endif  // HALIDE_CONVOLUTION_LAYER_HPP_
