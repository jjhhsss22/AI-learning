import numpy as np

class MaxPool2:

    """
    Neighbouring pixels tend to have similar values, so conv layers will typically also produce similar values for
    pixels near each other.

    For example, if we use an edge-detecting filter and find a strong edge at a certain 3x3 kernel,
    we are also likely to find strong edges at locations 1 pixel shifted from the original one.
    These represent the same edge and do not add any new information.

    So we use a pooling layer to reduce the size.
    Pooling divides the input’s width and height by the pool size.
    So this pooling layer (2x2) will transform a 26x26x8 input into a 13x13x8 output
    """

    def iterate_regions(self, image):  # similar to conv layer but generates 2x2 kernels not 3x3
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield im_region, i, j


    def forward(self, input):
        self.last_input = input

        h, w, num_filters = input.shape
        output = np.zeros((h // 2, w // 2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output  # output: (h / 2, w / 2, num_filters)

    def backprop(self, d_L_d_out):

        """
        Each gradient value from the backprop is assigned to where the original max value was before the maxpool,
        and every other value/kernel is zero because they did not contribute in the output.
        """

        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

                            # only assign gradient change to the max value in each kernel and expand the input out back
                            # and expand the input back out (h/2, w/2) -> (h, w)

        return d_L_d_input



