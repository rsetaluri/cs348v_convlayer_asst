#ifndef SIMPLE_CONVOLUTION_LAYER_HPP_
#define SIMPLE_CONVOLUTION_LAYER_HPP_

#include "convolution_layer.hpp"

class SimpleConvolutionLayer : public ConvolutionLayer {
 public:
  SimpleConvolutionLayer() = default;

  void Init(Parameters params) override;
  void Run(Parameters params, Data data) override;
};

#endif  // SIMPLE_CONVOLUTION_LAYER_HPP_
