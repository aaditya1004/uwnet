from uwnet import *
def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

def conv_net_with_batch_norm():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 2),
            make_batchnorm_layer(8),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 8, 3, 2),
            make_convolutional_layer(8, 8, 8, 16, 3, 1),
            make_batchnorm_layer(16),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 16, 3, 2),
            make_convolutional_layer(4, 4, 16, 32, 3, 1),
            make_batchnorm_layer(32),
            make_activation_layer(RELU),
            make_connected_layer(512, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 1000
rate = .5
momentum = .9
decay = .005

# m = conv_net()
m = conv_net_with_batch_norm()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# 7.6 Question: What do you notice about training the convnet with/without batch normalization? 
# How does it affect convergence? How does it affect what magnitude of learning rate you can use?
# Write down any observations from your experiments:
# TODO: Your answer
'''
For the set of hyper params (batch = 128, iters = 1000, rate = .07, momentum = .7, decay = .005)
conv_net():
training accuracy: %f 0.5102400183677673
test accuracy:     %f 0.5116999745368958

conv_net_with_batch_norm():
training accuracy: %f 0.5992599725723267
test accuracy:     %f 0.5813000202178955

For the set of hyper-params (batch = 128, iters = 1000, rate = .1, momentum = .9, decay = .005)
conv_net():
training accuracy: %f 0.41642001271247864
test accuracy:     %f 0.415800005197525

conv_net_with_batch_norm():
training accuracy: %f 0.5499200224876404
test accuracy:     %f 0.5335999727249146

For the set of hyper-params (batch = 128, iters = 1000, rate = .01, momentum = .9, decay = .005)
conv_net():
training accuracy: %f 0.4913400113582611
test accuracy:     %f 0.4871000051498413

conv_net_with_batch_norm():
training accuracy: %f 0.5824999809265137
test accuracy:     %f 0.5655999779701233

For the set of hyper-params (batch = 128, iters = 1000, rate = .5, momentum = .9, decay = .005)
conv_net():
training accuracy: %f 0.10000000149011612
test accuracy:     %f 0.10000000149011612

conv_net_with_batch_norm():
training accuracy: %f 0.10000000149011612
test accuracy:     %f 0.10000000149011612

With these results, we can see that for every given set of hyper-parameter results, conv nets with batch normalization
outperforms conv nets. In certain cases, we can see significant imporvement in performance in both testing and training 
accuracies. More importantly, conv nets with batch normalization seems to handle higher learning rate values(see lr = 0.1)
better than the ones without. This tells us that we can use higher learning rate values with batch norm. With higher
training and testing accuracy when compared to the normal models, we can also say batch norm allows faster convergences
to a target accuracy/ loss value too
'''
