# CS348V Assignment 3: Fast MobileNet Conv Layer evaluation #

In this assignment you will implement a simplified version of the MobileNet CNN. In particular, this assignment is restricted to the evaluation of a single convolutional layer of the network. See https://arxiv.org/abs/1704.04861 for more details about the full network. Also, unlike Assignment 2, this assignemnt is focused on efficiency. Your code will be evaluated on how fast it runs.

## Getting Started ##

Grab the assignment starter code.

    git clone < TODO: list repo >

To run the assignment, you will need to download the scene datasets, located at < TODO: data location >.

__Build Instructions__

The codebase uses a simple `Makefile` as the build system. However, there is a dependency on Halide. To get the code building right away, you can modify `Makefile`, and replace the lines

    DEFINES := -DUSE_HALIDE
    LDFLAGS := -L$(HALIDE_DIR)/bin -lHalide -ldl -lpthread

with

    DEFINES :=
    LDFLAGS := -ldl -lpthread

To build the starter code, run `make` from the top level directory. The assignment source code is in `src/`, and object files and binaries will be populated in `build/` and `bin/` respectively.

Once you decide to use Halide, follow the instructions at http://halide-lang.org/. In particular, you should [download a binary release of Halide](https://github.com/halide/Halide/releases). Once you've downloaded and untar'd the release, say into directory `halide_dir`, change the following line in `Makefile`

    HALIDE_DIR=/Users/setaluri/halide

to

    HALIDE_DIR=<halide_dir>

Then you can build the code using the instructions above.

__Running the starter code:__

Now you can run the camera. Just run:

    ./bin/convlayer DATA_DIR/activations.bin DATA_DIR/weights.bin DATA_DIR/golden.bin <num_runs>

This code will run your (initially empty) version of the convolution layer using the activations in `DATA_DIR/activations.bin` and weights in `DATA_DIR/weights.bin`. It will run for `num_runs` trials, and report the timings across all runs, as well as validate the output against the data contained in `DATA_DIR/golden.bin`. Note that if you are using Halide, the command will be slightly different. On OSX it will be

    DYLD_LIBRARY_PATH=<halide_dir>/bin ./bin/convlayer <args>

and on Linux it will be

    LD_LIBRARY_PATH=<halide_dir>/bin ./bin/convlayer <args>
