import numpy as np
import struct

def WriteBytesToFile(buff, filename):
    with open(filename, 'wb') as f:
        f.write(buff)

def AppendBytesToFile(buff, filename):
    with open(filename, 'ab') as f:
        f.write(buff)

def RunMain():
    activations_filename = 'activations.bin'
    weights_filename = 'weights.bin'
    golden_filename = 'golden.bin'
    width = 56
    height = 56
    channels = 128
    k = 3
    f = 128
    # Note the 2 additional elements in each dimension in the activations. This
    # is padding to make compute easier.
    activations = np.zeros((width + 2, height + 2, channels), dtype=np.float32)

    dw_weights = np.zeros((k, k, channels), dtype=np.float32)
    pw_weights = np.zeros((f, channels), dtype=np.float32)

    dw_bias = np.zeros((channels), dtype=np.float32)
    pw_bias = np.zeros((f), dtype=np.float32)

    dw_average = np.zeros((channels), dtype=np.float32)
    dw_variance = np.zeros((channels), dtype=np.float32)
    dw_beta = np.zeros((channels), dtype=np.float32)
    dw_gamma = np.zeros((channels), dtype=np.float32)

    pw_average = np.zeros((f), dtype=np.float32)
    pw_variance = np.zeros((f), dtype=np.float32)
    pw_beta = np.zeros((f), dtype=np.float32)
    pw_gamma = np.zeros((f), dtype=np.float32)

    golden = np.zeros((width, height, f), dtype=np.float32)

    # Write activations.
    WriteBytesToFile(struct.pack('i', int(width)), activations_filename)
    AppendBytesToFile(struct.pack('i', int(height)), activations_filename)
    AppendBytesToFile(struct.pack('i', int(channels)), activations_filename)
    AppendBytesToFile(activations.tostring('F'), activations_filename)

    # Write all weights to same file.
    # Write depthwise weights.
    WriteBytesToFile(struct.pack('i', int(k)), weights_filename)
    AppendBytesToFile(struct.pack('i', int(channels)), weights_filename)
    AppendBytesToFile(dw_weights.tostring('F'), weights_filename)
    # Write pointwise weights.
    AppendBytesToFile(struct.pack('i', int(f)), weights_filename)
    AppendBytesToFile(struct.pack('i', int(channels)), weights_filename)
    AppendBytesToFile(pw_weights.tostring('F'), weights_filename)
    # Write depthwise bias.
    AppendBytesToFile(dw_bias.tostring('F'), weights_filename)
    # Write pointwise bias.
    AppendBytesToFile(pw_bias.tostring('F'), weights_filename)
    # Write depthwise batch norm params.
    AppendBytesToFile(dw_average.tostring('F'), weights_filename)
    AppendBytesToFile(dw_variance.tostring('F'), weights_filename)
    AppendBytesToFile(dw_beta.tostring('F'), weights_filename)
    AppendBytesToFile(dw_gamma.tostring('F'), weights_filename)
    # Write pointwise batch norm params.
    AppendBytesToFile(pw_average.tostring('F'), weights_filename)
    AppendBytesToFile(pw_variance.tostring('F'), weights_filename)
    AppendBytesToFile(pw_beta.tostring('F'), weights_filename)
    AppendBytesToFile(pw_gamma.tostring('F'), weights_filename)

    # Write golden data.
    WriteBytesToFile(golden.tostring('F'), golden_filename)

if __name__ == '__main__':
    RunMain()
