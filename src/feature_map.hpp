#ifndef FEATURE_MAP_HPP_
#define FEATURE_MAP_HPP_

#include <cstring>
#include <memory>
#include <string>

template<typename T> class FeatureMap {
 public:
  static std::unique_ptr<FeatureMap> New(int width, int height, int channels) {
    T* data = new T[width * height * channels];
    return std::unique_ptr<FeatureMap>(
	new FeatureMap(width, height, channels, data));
  }
  static std::unique_ptr<FeatureMap> New(std::string filename) {
    FILE* f = fopen(filename.c_str(), "rb");
    if (not f) return nullptr;
    int width;
    int height;
    int channels;
    fread(&width, sizeof(width), 1, f);
    fread(&height, sizeof(height), 1, f);
    fread(&channels, sizeof(channels), 1, f);
    T* data = new T[width * height * channels];
    fread(data, sizeof(T), width * height * channels, f);
    return std::unique_ptr<FeatureMap>(
	new FeatureMap(width, height, channels, data));
  }
  ~FeatureMap() { delete[] data_; }

  int width() const { return width_; }
  int height() const { return height_; }
  int channels() const { return channels_; }

  std::unique_ptr<FeatureMap> Clone() const {
    T* new_data = new T[width_ * height_ * channels_];
    std::memcpy(new_data, data_, sizeof(T) * width_ * height_ * channels_);
    return std::unique_ptr<FeatureMap>(
	new FeatureMap(width_, height_, channels_, new_data));
  }

 private:
  // Takes ownership of @data.
  FeatureMap(int width, int height, int channels, T* data)
    : width_(width), height_(height), channels_(channels), data_(data) {}

  int width_;
  int height_;
  int channels_;
  T* data_;  // owns.
};

#endif  // FEATURE_MAP_HPP_
