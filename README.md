

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST.data/",one_hot =True)

Extracting MNIST.data/train-images-idx3-ubyte.gz
Extracting MNIST.data/train-labels-idx1-ubyte.gz
Extracting MNIST.data/t10k-images-idx3-ubyte.gz
Extracting MNIST.data/t10k-labels-idx1-ubyte.gz

type(mnist)

tensorflow.contrib.learn.python.learn.datasets.base.Datasets

mnist.train.images

array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       ..., 
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.],
       [ 0.,  0.,  0., ...,  0.,  0.,  0.]], dtype=float32)

mnist.train.num_examples

55000

mnist.test.num_examples

10000

mnist.validation.num_examples

5000

import matplotlib.pyplot as plt

mnist.train.images[1].shape

(784,)

plt.imshow(mnist.train.images[1].reshape(28,28),cmap='gist_gray')

<matplotlib.image.AxesImage at 0x7f88c85ad940>

plt.imshow(mnist.train.images[1000].reshape(28,28),cmap='gist_gray')

<matplotlib.image.AxesImage at 0x7f88c85263c8>

mnist.train.images[1].max()

1.0

mnist.train.images[1].min()

0.0

#create a model 

x = tf.placeholder(tf.float32, shape=[None,784])

#10 because 0-9 possible numbers
#w = weight
W = tf.Variable(tf.zeros([784,10]))

#b = bias
b = tf.Variable(tf.zeros([10]))

#create a  graph
y = tf.matmul(x,W) + b

#loss and optimzer
y_true = tf.placeholder(tf.float32, [None,10])

#cross Entropy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y)) 

#optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)

train = optimizer.minimize(cross_entropy)

#create the session or execution

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1500):
        
        batch_x, batch_y = mnist.train.next_batch(300)
             
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y})
        
    matches = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
    
    acc = tf.reduce_mean(tf.cast(matches,tf.float32))
    
    print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))
        
        
               
    
    
    
     
   

0.9209

