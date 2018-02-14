#include <iostream>
#include "feature_map.hpp"

int main(int argc, char** argv) {
  auto feature_map = FeatureMap<double>::New("fm.bin");
  if (not feature_map) {
    std::cout << "Could not read feature map from file fm.bin" << std::endl;
    return 0;
  }
  std::cout << "width = " << feature_map->width() << std::endl;
  std::cout << "height = " << feature_map->height() << std::endl;
  std::cout << "channels = " << feature_map->channels() << std::endl;
}
