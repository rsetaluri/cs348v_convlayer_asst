SRC_DIR := src
HALIDE_DIR=/Users/setaluri/halide
BUILD_DIR := build
BIN_DIR := bin
SRC_FILES := $(wildcard $(SRC_DIR)/*.cpp)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRC_FILES))
INCLUDES=-I$(HALIDE_DIR)/include
LDFLAGS := -L$(HALIDE_DIR)/bin -lpthread -ldl -lHalide
CPPFLAGS :=
CXXFLAGS := -std=c++11

convlayer: $(OBJ_FILES)
	@mkdir -p $(BIN_DIR)
	g++ $(LDFLAGS) -o $(BIN_DIR)/$@ $^

clean:
	\rm -rf $(BUILD_DIR) $(BIN_DIR)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	g++ $(CPPFLAGS) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<
