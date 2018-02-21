#ifndef CONVOLUTION_LAYER_HPP_
#define CONVOLUTION_LAYER_HPP_

class ConvolutionLayer {
 public:
  struct Data {
    float* input = nullptr;
    float* output = nullptr;
    float* depthwise_weights = nullptr;
    float* pointwise_weights = nullptr;
    float* depthwise_bias = nullptr;
    float* pointwise_bias = nullptr;
  };
  struct Parameters {
    int width = 0;
    int height = 0;
    int channels = 0;
    int k = 0;
    int f = 0;
  };
  
  virtual void Init(Parameters params) = 0;
  virtual void Run(Parameters params, Data data) = 0;  
};

#endif  // CONVOLUTION_LAYER_HPP_
