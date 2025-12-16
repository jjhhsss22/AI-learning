import numpy as np

class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters

        # filter is a 3d array with dimensions (num_filters, 3, 3)
        # reduce variance by dividing by 9
        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iterate_regions(self, image):
        """
        Generates all possible 3x3 image regions using valid padding.
        image is a 2d numpy array for the pixels on the image

        It produces one valid 3×3 region at a time (along with its coordinates).
        You can loop through it, and each yield gives you the next region without keeping them all in memory.
        This is the same as sliding a kernel across the image
        """

        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i + 3), j:(j + 3)]
                yield im_region, i, j

    def forward(self, input):
        self.last_input = input

        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

        """
        returns a (h-2, w-2, # of filters) tensor/matrix where each 2D location stores
        the summed result of applying one filter to a 3x3 patch. 
        """
        return output

    def backprop(self, d_L_d_out, learning_rate):

        d_L_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region

        self.filters -= learning_rate * d_L_d_filters


        return None  # don't need to return anything as no more back propagation is need (conv is first layer)