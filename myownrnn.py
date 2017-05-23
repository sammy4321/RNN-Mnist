
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
global batch_size
lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28   
n_steps = 28    
n_hidden_units = 128
n_classes = 10      


x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])


weights = {
    
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    
    X = tf.reshape(X, [-1, n_inputs])

    
    X_in = tf.matmul(X, weights['in']) + biases['in']
    
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

   

    
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    
    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in,time_major=False,dtype=tf.float32)

    
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))    # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']    # shape = (128, 10)

    return results


pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op = tf.train.AdamOptimizer().minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

with tf.Session() as sess:
    
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    hm_epochs=1
    step = 0
    for epoch in range(hm_epochs):
    	epoch_loss=0
    	global epoch_x
    	global epoch_y
    	for _ in range(int(mnist.train.num_examples/batch_size)):
    		epoch_x,epoch_y=mnist.train.next_batch(batch_size)
    		epoch_x=epoch_x.reshape([batch_size,n_steps,n_inputs])
    		_,c=sess.run([train_op,cost],feed_dict={x:epoch_x,y:epoch_y})

    		epoch_loss = epoch_loss + c

    	print('Epoch',epoch,'completed out of',hm_epochs,'loss:',epoch_loss)
    batch_size = 10000

    print('Accuracy :',accuracy.eval({x:mnist.test.images.reshape([-1, n_steps, n_inputs]), y:mnist.test.labels}))

