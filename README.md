# Building Neural Network with Pytorch - Assignment 3

## Problem Statement

The goal of the project is to build neural network which 
- Takes 2 Inputs
    - An image from MNIST dataset, and
    - A random number between 0-9.

- And gives two outputs
    - The "number" represented by the MNIST Image and
    - "Sum" of this "number with the random number sent as 2nd input.

# Network Architecture

The following architecture is used and has been built using pytorch. The network comprises of 2 Convolution layers, 3 full connected layers and 2 output layers with softmax output, one for number recognition and other for sum.

![Network Architecture](https://raw.githubusercontent.com/chaitanya-vanapamala/pytorch_mnist_multi_label/main/netowrk%20architecture.png)

The output shape/size after every layer is mention below every layer. Te CNN layers and Max Pool outputs follows the following formula

![O=\frac{N+F-2P}{S}+1](https://render.githubusercontent.com/render/math?math=\color{Green}\large%20O%3D%5Cfrac%7BN%2BF-2P%7D%7BS%7D%2B1)

Where 
- N = Input Image Height/Width (As we are using a Square Image)
- F = Convolution Filter Size
- P = Padding(0 in our case)
- S = Stride length

The implementation of the network Pytorch is as follows:

  ```
  Network(
    (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
    (conv2): Conv2d(6, 12, kernel_size=(5, 5), stride=(1, 1))
    (fc1): Linear(in_features=10, out_features=64, bias=True)
    (fc2): Linear(in_features=256, out_features=128, bias=True)
    (fc3): Linear(in_features=128, out_features=64, bias=True)
    (out1): Linear(in_features=64, out_features=10, bias=True)
    (out2): Linear(in_features=64, out_features=19, bias=True)
  )
  ```

## Data representation

### Input Data
The input data for neural network has 2 componenets. Image and the random number between 0-9. Image is a single channel 28x28 tensor and The random number is one hot encoded and then fed to the network.

### Output data
As we know from neural network architecture there are two softmax outputs with size 10 and 19. The first one is number represented by image and 2nd one is sum of number with random input number.

## Data Generation Strategy

As we need to send an extra input to neural network apart from Image we need to create a Custom Dataset class inheriting the Pytorch inbuild Dataset class.

The Custom dataset takes the mnist dataset as input and generates a random numbers between 0-9 of length equal to mnist dataset length.

This Custom Dataset gives a dictionary of inputs and targets keys each having a list of 2 elements.

## Combining 2 Inputs

Before combining the Random number input and Image following steps are performed.
1. The Image will be feeded to 2 Convolution layers and the features are extracted.
2. After the 2nd Convolution layer output the output will be flattened.
3. On the other hand the One hot encoded random number will be feeded to a fully connected layer.
4. The outputs after 2nd and 3rd layers is concatinated and then fed to 2 fully connected layers followed by softmax layers.

## Loss Function - CrossEntropy
The loss function used in the neural network is CrossEntropyLoss for both outputs and then average will be taken. 

As Cross entropy measures the difference between two probability distributions, and when this measure is less which means the two probability distributions are similar(Softmax output and target values). 

The problem we are trying to solve is a multi-class problem the cross entropy loss is the best choise.


## Model Evaluation

The model evaluation is done based on the loss values and accuracies during training and comparing them with test data set. 

## Results

The Following image shows the training logs, and we can see the loss is decreasing as we are increasinng epoch, also the number of correctly recognised numbers and sum are increasing which is nothing but the accuracy is increasing.

![Training Log](https://raw.githubusercontent.com/chaitanya-vanapamala/pytorch_mnist_multi_label/main/training%20log.png)


Let's check the plot of the loss values for 50 epochs.

![Loss Plot](https://raw.githubusercontent.com/chaitanya-vanapamala/pytorch_mnist_multi_label/main/loss%20plot.jpg)

We can clearly see the plot that the training loss is decreasing in a step by step fashion.

### Evaluation on Test Data

Similar Custom DataSet is created but with mnist test data set. While 10k images are passed to the model for inferrence and got 98% accuracy for Number recognition and 75% accuracy for Sum. 

These metrics are approximately same for training dataset also. Hance we can conclude that mode is able to learn and able to generalise the learning on unseen dataset. 

For increasing the accuracy of SUM, some the actions can be taken like increasing the data using augmentation technique and increasing the training epochs.