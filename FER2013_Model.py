import tensorflow as tf
from FER2013_Input import FER2013_Input
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

n_classes = 7
batch_size = 128


x = tf.placeholder('float', [None, 42, 42])
y = tf.placeholder('float', [None, 7])#Ali

def conv2d(x,W):
     #The strides parameter dictates the movement of the window, 1 pixel at a time
     return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
     #Pooling Window Size = 2x2
     #Strides= 2; 2 pixels at a time
     return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x):
     weights = {
     # 5x5 convoltuion, 1 input image, 32 outputs
     'W_conv1': tf.Variable(tf.random_normal([5,5,1,32])),
     # 4x4 convoltuion, 32 inputs, 32 outputs
     'W_conv2': tf.Variable(tf.random_normal([4,4,32,32])),
     # 5x5 convoltuion, 32 inputs, 32 outputs
     'W_conv3': tf.Variable(tf.random_normal([5,5,32,64])),
     'W_fc1': tf.Variable(tf.random_normal([6*6*64, 3072])),
     'W_fc2': tf.Variable(tf.random_normal([3072, 7]))
     }
     biases = {
         'b_conv1' : tf.Variable(tf.random_normal([32])),
         'b_conv2' : tf.Variable(tf.random_normal([32])),
         'b_conv3' : tf.Variable(tf.random_normal([64])),
         'b_fc1' : tf.Variable(tf.random_normal([3072])),
         'b_fc2' : tf.Variable(tf.random_normal([7]))
     }

     x = tf.reshape(x, shape=[-1, 42, 42, 1])


     conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
     conv1 = maxpool2d(conv1)
    
     conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
     conv2 = maxpool2d(conv2)

     conv3 = tf.nn.relu(conv2d(conv2, weights['W_conv3']) + biases['b_conv3'])
     conv3 = maxpool2d(conv3)

     #The image by now is 6x6
     fc1 = tf.reshape(conv3, [-1, 6*6*64])
     fc1 = tf.reshape(fc1, [-1, 6*6*64])
     fc1 = tf.nn.relu(tf.matmul(fc1, weights['W_fc1']) + biases['b_fc1'])

    #not sure if we should reshape fc2
     fc2 = tf.nn.relu(tf.matmul(fc1, weights['W_fc2']) + biases['b_fc2'])
     return fc2

def train_neural_network(x):
     fer = FER2013_Input('/home/alaa/Desktop/GP/')
     #training_data
     #training_labels, training_images = fer.FER2013_Training_Set();
     #training_images = tf.image.resize_images(training_images, [42, 42])
     prediction = convolutional_neural_network(x)#Ali
     
     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))#Ali
     optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
     
     hm_epochs = 16
     with tf.Session() as sess:
        #training_images = tf.image.resize_images(training_images, (42, 42))
        sess.run(tf.global_variables_initializer())#Ali
        #training_images = tf.image.resize_images(training_images, (42, 42)) 
		#Ali
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for batchNum in range(int((28709+batch_size)/batch_size)):
                epoch_y, epoch_x = fer.Get_batch(batchNum, 'Training')
                #epoch_x = tf.image.resize_images(epoch_x, (42,42))
                #print (epoch_x[0].size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
                #print('Hello')

            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        testing_labels, testing_images = fer.FER2013_Testing_Set()
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:testing_images, y:testing_labels}))

train_neural_network(x)