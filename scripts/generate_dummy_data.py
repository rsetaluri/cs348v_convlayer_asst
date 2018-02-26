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
    activations = np.random.random_sample((width + 2, height + 2, channels)).astype(np.float32) * float(20.) - float(10.)

    dw_weights = np.random.random_sample((k, k, channels)).astype(np.float32) * float(20.) - float(10.)
    pw_weights = np.random.random_sample((f, channels)).astype(np.float32) * float(20.) - float(10.)

    dw_average = np.random.random_sample((channels)).astype(np.float32) * float(20.) - float(10.)
    dw_variance = np.random.random_sample((channels)).astype(np.float32) * float(10.)
    dw_beta = np.random.random_sample((channels)).astype(np.float32) * float(20.) - float(10.)
    dw_gamma = np.random.random_sample((channels)).astype(np.float32) * float(20.) - float(10.)

    pw_average = np.random.random_sample((f)).astype(np.float32) * float(20.) - float(10.)
    pw_variance = np.random.random_sample((f)).astype(np.float32) * float(10.)
    pw_beta = np.random.random_sample((f)).astype(np.float32) * float(20.) - float(10.)
    pw_gamma = np.random.random_sample((f)).astype(np.float32) * float(20.) - float(10.)

    golden = np.random.random_sample((width, height, f)).astype(np.float32) * float(20.) - float(10.)

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
