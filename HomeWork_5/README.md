**Part I:**

In this work, two networks were trained to classify the MNIST dataset.
The first network was based on the LeNet and the other one was a fully
connected network. The results of comparison between these networks can
be seen in the attached file.

<img src="media\image1.png" style="width:6.42708in;height:4.1875in" />

Figure 1: Training and test loss of LeNet5 for 100 epochs

<img src="media\image2.png" style="width:6.70829in;height:4.0625in" />

Figure 2: Training and test loss of fully connected neural net for 100
epochs

**Part II:**

The LeNet architecture was again used to classify the CIFAR10 dataset.
OpenCV was used to stream the live video using webcam. The livestream
was then processed using trained LeNet to generate the real-time class
predictions.
