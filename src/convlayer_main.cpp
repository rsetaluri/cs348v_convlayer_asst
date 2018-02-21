#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include "convolution_layer.hpp"
#include "simple_convolution_layer.hpp"

int main(int argc, char** argv) {
  // Parse command line arguments.
  if (argc < 5) {
    std::cout << "usage: " << argv[0]
              << " activations data golden num_runs" << std::endl;
    return 0;
  }
  const std::string activations_filename = argv[1];
  const std::string data_filename = argv[2];
  const std::string golden_filename = argv[3];
  const int num_runs = atoi(argv[4]);

  ConvolutionLayer::Parameters params;
  ConvolutionLayer::Data data;
  std::unique_ptr<ConvolutionLayer> conv_layer(new SimpleConvolutionLayer);
  conv_layer->Init(params);

  // Run convolution layer implementation for num_runs which is specified on the
  // command line.
  double total_elapsed = 0.;
  for (int run = 0; run < num_runs; run++) {
    auto start = std::chrono::system_clock::now();
    conv_layer->Run(params, data);  
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    total_elapsed += elapsed.count();
    std::cout << "Convolution layer took " << elapsed.count()
              << " secconds" << std::endl;
  }
  const double average_elapsed = total_elapsed / (double)num_runs;
  std::cout << "Average time: " << average_elapsed << "s" << std::endl;

  return 0;
}
