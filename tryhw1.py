from uwnet import *

# Colab with Varsha Konda -- @vkonda

def conv_net():
    l = [   make_convolutional_layer(32, 32, 3, 8, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(32, 32, 8, 3, 2),
            make_convolutional_layer(16, 16, 8, 16, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(16, 16, 16, 3, 2),
            make_convolutional_layer(8, 8, 16, 32, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(8, 8, 32, 3, 2),
            make_convolutional_layer(4, 4, 32, 64, 3, 1),
            make_activation_layer(RELU),
            make_maxpool_layer(4, 4, 64, 3, 2),
            make_connected_layer(256, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)

def connected_net():
    l = [   make_connected_layer(3072, 310),
            make_activation_layer(RELU),
            make_connected_layer(310, 128),
            make_activation_layer(RELU),
            make_connected_layer(128, 64),
            make_activation_layer(RELU),
            make_connected_layer(64, 32),
            make_activation_layer(RELU),
            make_connected_layer(32, 10),
            make_activation_layer(SOFTMAX)]
    return make_net(l)
    
print("loading data...")
train = load_image_classification_data("cifar/cifar.train", "cifar/cifar.labels")
test  = load_image_classification_data("cifar/cifar.test",  "cifar/cifar.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .005

m = connected_net()
print("training...")
train_image_classifier(m, train, batch, iters, rate, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

# How accurate is the fully connected network vs the convnet when they use similar number of operations?
# Why are you seeing these results? Speculate based on the information you've gathered and what you know about DL and ML.

# Convnet operations: (32 * 32 * 3) * (3 * 3 * 8) + (16 * 16 * 8) * (3 * 3 * 16) + (8 * 8 * 16) * (3 * 3 * 32) + (4 * 4 * 32) * (3 * 3 * 64) + 256 * 10 
#                       = 1105920 + 2560 = 1108480 operations.
# Connected net operations: 1002560 operations.
#  
# For the convnet, training accuracy is ~69% and testing accuracy is ~64%
# For the fully connected model, training accuracy is ~53% and testing accuracy is ~50%
# Clearly, convnet works better than fully connected model. Using convolutional layers we
# are able to extract more useful features and reduce the number of weights/params for learning 
# further as opposed to the fully connected model where some features aren't priortized over 
# and whole images are passed forward in training. 

