import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils
import os

data_proc = utils.ImageUtils()
steps = 2500
batch_size = 100
img_width = 500
img_height = 500
# 512 -- 64

X = tf.placeholder(tf.float32, shape=[None, img_width, img_height, 1])
y = tf.placeholder(tf.float32, shape=[None, 2])
l2_reg = tf.contrib.layers.l2_regularizer(scale=0.0001)

conv_1 = tf.layers.conv2d(X, filters=32, kernel_size=(3, 3), strides=4, padding="SAME", activation=tf.nn.relu)
pooling = tf.layers.max_pooling2d(conv_1, pool_size=[2, 2], strides=4, padding="VALID")
conv_2 = tf.layers.conv2d(pooling, filters=128, kernel_size=(2, 2), strides=4, padding="SAME", activation=tf.nn.relu)
flat_layer = tf.reshape(conv_2, [-1, 8192])
regressor_in = tf.layers.dense(inputs=flat_layer, units=8192, activation=tf.nn.relu)
regressor_hidden = tf.layers.dense(inputs=regressor_in, units=512, activation=tf.nn.relu, kernel_regularizer=l2_reg)
regressor_hidden_1 = tf.layers.dense(inputs=regressor_hidden, units=64, activation=tf.nn.relu, kernel_regularizer=l2_reg)
regressor_output = tf.layers.dense(inputs=regressor_hidden_1, units=2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=regressor_output, labels=y))
cost += tf.losses.get_regularization_loss()
optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
init = tf.global_variables_initializer()

predictions = tf.nn.softmax(regressor_output)
saver = tf.train.Saver()


def process(out):
	rtrn = []
	for val in out:
		if val[0] > val[1]:
			rtrn.append([1, 0])
		else:
			rtrn.append([0, 1])
	return rtrn

def correct_predictions(out, actual):
	num_correct = 0
	num_incorrect = 0
	for i in range(len(out)):
		if(np.array_equal(out[i], actual[i])):
			num_correct += 1
		else:
			num_incorrect += 1
	return num_correct, num_incorrect	

def test(sess):
	accuracy = []
	total_correct = 0
	total = 0
	saver.save(sess, './checkpoint_dat.ckpt')
	for i in range(7):
		test_batch = data_proc.get_test_batch(batch_size)
		out = sess.run(predictions, feed_dict = {X: test_batch['image'], y: test_batch['class']})
		corr, not_corr = correct_predictions(process(out), test_batch['class'])
		print(str(i) + '  ' + str(corr) + '/' + str(not_corr))
		accuracy.append((corr/(corr+not_corr)))
		total_correct += corr
		total += corr + not_corr
	print(total_correct/total)
	return (total_correct/total)

def train(sess):
	losses = []
	for i in range(steps):
		train_batch = data_proc.get_train_batch(batch_size)
		#out = sess.run(conv_2, feed_dict = {X: train_batch['image'], y: train_batch['class']})
		sess.run(optimizer, feed_dict = {X: train_batch['image'], y: train_batch['class']})
		loss = sess.run(cost, feed_dict = {X: train_batch['image'], y: train_batch['class']})
		losses.append(loss)
		print(str(i) + '   ' + str(loss))
		if(i % 100 == 0 and i != 0):
			print("Running eval at step: " + str(i))
			acc = test(sess)
			if(acc > 0.87):
				break
	plt.plot(losses)
	plt.show()



with tf.Session() as sess:
	if os.path.isfile('./checkpoint_dat.ckpt.index'):
		print("Restoring session from checkpoint: ")
		saver.restore(sess, './checkpoint_dat.ckpt')
	else:
		init.run()
	test(sess)