#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "uwnet.h"

#define EPS 0.00001

// Take mean of matrix x over rows and spatial dimension
// matrix x: matrix with data
// int groups: number of distinct means to take, usually equal to # outputs
// after connected layers or # channels after convolutional layers
// returns: (1 x groups) matrix with means
matrix mean(matrix x, int groups)
{
    assert(x.cols % groups == 0);
    matrix m = make_matrix(1, groups);
    int n = x.cols / groups;
    int i, j;
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            m.data[j/n] += x.data[i*x.cols + j];
        }
    }
    for(i = 0; i < m.cols; ++i){
        m.data[i] = m.data[i] / x.rows / n;
    }
    return m;
}

// Take variance over matrix x given mean m
matrix variance(matrix x, matrix m, int groups)
{
    matrix v = make_matrix(1, groups);
    // TODO: 7.1 - Calculate variance
    int i, j, n;
    n = x.cols/groups;

    for (i = 0; i < x.rows; i++) {
        for (j = 0; j < x.cols; j++) {
            int index = i*x.cols + j;
            int channel = j/n;

            float diff = x.data[index] - m.data[channel];

            v.data[channel] += pow(diff, 2.0);
        }
    }

    for(i = 0; i < v.cols; ++i){
        v.data[i] = v.data[i] / x.rows / n;
    }

    return v;
}

// Normalize x given mean m and variance v
// returns: y = (x-m)/sqrt(v + epsilon)
matrix normalize(matrix x, matrix m, matrix v, int groups)
{
    matrix norm = make_matrix(x.rows, x.cols);
    // TODO: 7.2 - Normalize x
    int i, j, n;
    float mean, var, data;
    n = x.cols/groups;
    for (i = 0; i < x.rows; i++) {
        for (j = 0; j < x.cols; j++) {
            mean = m.data[j/n];
            var = v.data[j/n];
            data = x.data[i*x.cols + j];
            norm.data[i*x.cols + j] = (data-mean)/sqrt(var + EPS);
        }
    }

    return norm;
}


// Run an batchnorm layer on input
// layer l: pointer to layer to run
// matrix x: input to layer
// returns: the result of running the layer y = (x - mu) / sigma
matrix forward_batchnorm_layer(layer l, matrix x)
{
    // Saving our input
    // Probably don't change this
    free_matrix(*l.x);
    *l.x = copy_matrix(x);

    if(x.rows == 1){
        return normalize(x, l.rolling_mean, l.rolling_variance, l.channels);
    }

    float s = 0.1;
    matrix m = mean(x, l.channels);
    matrix v = variance(x, m, l.channels);
    matrix y = normalize(x, m, v, l.channels);

    scal_matrix(1-s, l.rolling_mean);
    axpy_matrix(s, m, l.rolling_mean);
    scal_matrix(1-s, l.rolling_variance);
    axpy_matrix(s, v, l.rolling_variance);

    free_matrix(m);
    free_matrix(v);

    return y;
}

matrix delta_mean(matrix d, matrix v)
{
    int groups = v.cols;
    matrix dm = make_matrix(1, groups);
    // TODO 7.3 - Calculate dL/dm
    int i, j, n, index, channel; 
    n = d.cols/groups;
    for (i = 0; i < d.rows; i++) {
        for (j = 0; j < d.cols; j++) {
            index = i*d.cols + j;
            channel = j/n;
            dm.data[channel] += (d.data[index])*(-1.0/sqrt(v.data[channel] + EPS));
        }
    }
    return dm;
}


matrix delta_variance(matrix d, matrix x, matrix m, matrix v)
{
    int groups = m.cols;
    matrix dv = make_matrix(1, groups);
    // TODO 7.4 - Calculate dL/dv
    int i, j, n, index1, index2, channel;
    n = d.cols/groups;
    float dldy, dmean, dvariance;

    for (i = 0; i < d.rows; i++) {
        for (j = 0; j < d.cols; j++) {
            index1 = i*d.cols + j;
            index2 = i*x.cols + j;
            channel = j/n;

            dldy = d.data[index1];
            dmean = x.data[index2] - m.data[channel];
            dvariance = (-1.0/2)*pow(v.data[channel] + EPS, (-3.0/2.0));

            dv.data[channel] += (dldy * dmean * dvariance);
        }
    }
    return dv;
}

matrix delta_batch_norm(matrix d, matrix dm, matrix dv, matrix m, matrix v, matrix x)
{
    matrix dx = make_matrix(d.rows, d.cols);
    // TODO 7.5 - Calculate dL/dx
    int groups = m.cols;
    int i, j, n, index1, index2, index3, channel, nd;
    n = d.cols/groups;
    nd = n * dx.rows;
    for (i = 0; i < dx.rows; i++) {
        for (j = 0; j < dx.cols; j++) {

            index1 = i*d.cols + j;
            index2 = i*x.cols + j;
            index3 = i*dx.cols + j;
            channel = j/n;

            dx.data[index3] = d.data[index1]*(1.0/sqrt(v.data[channel] + EPS)) +    
                                dv.data[channel]*(2*(x.data[index2] - m.data[channel])/nd) +
                                dm.data[channel]*(1.0/nd);
            
        }
    }
    return dx;
}


// Run an batchnorm layer on input
// layer l: pointer to layer to run
// matrix dy: derivative of loss wrt output, dL/dy
// returns: derivative of loss wrt input, dL/dx
matrix backward_batchnorm_layer(layer l, matrix dy)
{
    matrix x = *l.x;

    matrix m = mean(x, l.channels);
    matrix v = variance(x, m, l.channels);

    matrix dm = delta_mean(dy, v);
    matrix dv = delta_variance(dy, x, m, v);
    matrix dx = delta_batch_norm(dy, dm, dv, m, v, x);

    free_matrix(m);
    free_matrix(v);
    free_matrix(dm);
    free_matrix(dv);

    return dx;
}

// Update batchnorm layer..... nothing happens tho
// layer l: layer to update
// float rate: SGD learning rate
// float momentum: SGD momentum term
// float decay: l2 normalization term
void update_batchnorm_layer(layer l, float rate, float momentum, float decay){}

layer make_batchnorm_layer(int groups)
{
    layer l = {0};
    l.channels = groups;
    l.x = calloc(1, sizeof(matrix));

    l.rolling_mean = make_matrix(1, groups);
    l.rolling_variance = make_matrix(1, groups);

    l.forward = forward_batchnorm_layer;
    l.backward = backward_batchnorm_layer;
    l.update = update_batchnorm_layer;
    return l;
}
