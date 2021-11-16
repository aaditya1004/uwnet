#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <float.h>
#include "uwnet.h"


// Run a maxpool layer on input
// layer l: pointer to layer to run
// matrix in: input to layer
// returns: the result of running the layer
matrix forward_maxpool_layer(layer l, matrix in)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(in);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    matrix out = make_matrix(in.rows, outw*outh*l.channels);

    // TODO: 6.1 - iterate over the input and fill in the output with max values
    int out_c = 0;
    int padding = (l.size - 1) / 2;
    for (int im = 0; im < in.rows; im++) {

        float* c_im = in.data + im*in.cols;
        float* c_out = out.data + im*out.cols;

        for (int c = 0; c < l.channels; c++) {

            for (int y = 0; y < l.height; y += l.stride) {

                for (int x = 0; x < l.width; x += l.stride) {

                    float max = 0;

                    for (int f_y = 0; f_y < l.size; f_y++) {

                        for (int f_x = 0; f_x < l.size; f_x++) {

                            int o_y = y + f_y - padding;
                            int o_x = x + f_x - padding;

                            if (o_x >= 0 && o_x < l.width && o_y >= 0 && o_y < l.height) {
                                float curr = c_im[c * l.width * l.height + o_y * l.width + o_x];
                                if (curr > max) {
                                    max = curr;
                                }
                            }
                        }

                    }

                    c_out[outw * outh * c + out_c] = max;
                    out_c++;

                }

            }

            out_c = 0;

        }
        
    }
    return out;
}

// Run a maxpool layer backward
// layer l: layer to run
// matrix dy: error term for the previous layer
matrix backward_maxpool_layer(layer l, matrix dy)
{
    matrix in    = *l.x;
    matrix dx = make_matrix(dy.rows, l.width*l.height*l.channels);

    int outw = (l.width-1)/l.stride + 1;
    int outh = (l.height-1)/l.stride + 1;
    // TODO: 6.2 - find the max values in the input again and fill in the
    // corresponding delta with the delta from the output. This should be
    // similar to the forward method in structure.
    int out_c = 0;
    int padding = (l.size - 1) / 2;
    for (int im = 0; im < in.rows; im++) {
        float* layer_im = in.data + im*in.cols;
        float* dy_im = dy.data + im*dy.cols;
        float* dx_im = dx.data + im*dx.cols;
        for (int c = 0; c < l.channels; c++) {

            for (int y = 0; y < l.height; y += l.stride) {

                for (int x = 0; x < l.width; x += l.stride) {

                    int max_x = 0;
                    int max_y = 0;
                    float max = 0;

                    for (int f_y = 0; f_y < l.size; f_y++) {

                        for (int f_x = 0; f_x < l.size; f_x++) {
                            
                            int o_y = y + f_y - padding;
                            int o_x = x + f_x - padding;

                            if (o_x >= 0 && o_x < l.width && o_y >= 0 && o_y < l.height) {
                                float curr = layer_im[c * l.width * l.height + o_y * l.width + o_x];
                                if (curr > max) {
                                    max = curr;
                                    max_x = o_x;
                                    max_y = o_y;
                                }
                            }

                        }

                    }

                    dx_im[c * l.width * l.height + max_y * l.width + max_x] += dy_im[out_c + outw * outh * c];
                    out_c++;

                }

            }
            out_c = 0;

        }
    }
    return dx;
}

// Update maxpool layer
// Leave this blank since maxpool layers have no update
void update_maxpool_layer(layer l, float rate, float momentum, float decay){}

// Make a new maxpool layer
// int w: width of input image
// int h: height of input image
// int c: number of channels
// int size: size of maxpool filter to apply
// int stride: stride of operation
layer make_maxpool_layer(int w, int h, int c, int size, int stride)
{
    layer l = {0};
    l.width = w;
    l.height = h;
    l.channels = c;
    l.size = size;
    l.stride = stride;
    l.x = calloc(1, sizeof(matrix));
    l.forward  = forward_maxpool_layer;
    l.backward = backward_maxpool_layer;
    l.update   = update_maxpool_layer;
    return l;
}

