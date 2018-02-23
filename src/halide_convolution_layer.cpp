#include "halide_convolution_layer.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include "Halide.h"

void HalideConvolutionLayer::Init(Parameters params) {
}

void HalideConvolutionLayer::Run(Parameters params, Data data) {
  Halide::Buffer<float> input;
  Halide::Func tmp;
  Halide::Var x, y, c;
  // (1) Depthwise convolution. Note that the spatial kernel is centered
  // around the output pixel, so we need an offset.
  {
    // TODO(raj): Populate w and b from @data.
    Halide::Buffer<float> w;  // convolution weights.
    Halide::Buffer<float> b;  // biases.
    Halide::RDom r(-1, 3, -1, 3);
    tmp(x, y, c) = 0.f;
    // Note the + 1 on the weights index, because r starts at -1.
    tmp(x, y, c) += input(x + r.x, y + r.y, c) * w(r.x + 1, r.y + 1) + b(c);
  }
  // (2) Batch norm.
  {
    // TODO(raj): Populate batch norm parameters from @data.
    Halide::Buffer<float> average;
    Halide::Buffer<float> variance;
    Halide::Buffer<float> beta;
    Halide::Buffer<float> gamma;
    Halide::Expr v = tmp(x, y, c);
    v = (v - average(c)) / Halide::sqrt(variance(c) + params.epsilon);
    tmp(x, y, c) = gamma(c) * v + beta(c);
  }
  // (3) ReLU.
  tmp(x, y, c) = Halide::max(tmp(x, y, c), 0.f);
  // (4) Pointwise convolution.
  Halide::Func output;
  {
    // TODO(raj): Populate w and b from @data.
    Halide::Buffer<float> w;  // convolution weights.
    Halide::Buffer<float> b;  // biases.
    Halide::RDom r(-1, 3, -1, 3);
    output(x, y, c) = 0.f;
    output(x, y, c) += tmp(x, y, r.x) * w(c, r.x) + b(c);
  }
  // (5) Batch norm.
  {
    // TODO(raj): Populate batch norm parameters from @data.
    Halide::Buffer<float> average;
    Halide::Buffer<float> variance;
    Halide::Buffer<float> beta;
    Halide::Buffer<float> gamma;
    Halide::Expr v = output(x, y, c);
    v = (v - average(c)) / Halide::sqrt(variance(c) + params.epsilon);
    output(x, y, c) = gamma(c) * v + beta(c);
  }
  // (6) ReLU.
  output(x, y, c) = Halide::max(output(x, y, c), 0.f);

  // Realize output buffer and copy to data.output pointer.
  Halide::Buffer<float> output_buffer =
      output.realize(input.width(), input.height(), input.channels());
  std::cout << "[DEBUG] output buffer size = "
            << output_buffer.get()->number_of_elements() << std::endl;
  const auto raw_ptr = output_buffer.get()->data();
  std::memcpy(data.output, raw_ptr, params.width * params.height * params.f);
}
