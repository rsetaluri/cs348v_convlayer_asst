#ifndef CONVOLUTION_LAYER_HPP_
#define CONVOLUTION_LAYER_HPP_

class ConvolutionLayer {
 public:
  struct Data {
    float* input = nullptr;
    float* output = nullptr;
    // Convolution kernel weights.
    float* depthwise_weights = nullptr;
    float* pointwise_weights = nullptr;
    // Depthwise batch norm learned parameters.
    float* depthwise_average = nullptr;
    float* depthwise_variance = nullptr;
    float* depthwise_beta = nullptr;
    float* depthwise_gamma = nullptr;
    // Pointwise batch norm learned parameters.
    float* pointwise_average = nullptr;
    float* pointwise_variance = nullptr;
    float* pointwise_beta = nullptr;
    float* pointwise_gamma = nullptr;
  };
  struct Parameters {
    int width = 0;
    int height = 0;
    int channels = 0;
    int k = 0;
    int f = 0;
    float epsilon = .00001f;  // used in batch norm.
  };

  virtual void Init(Parameters params) = 0;
  virtual void Run(Parameters params, Data data) = 0;
};

#endif  // CONVOLUTION_LAYER_HPP_
