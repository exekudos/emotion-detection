import tensorflow as tf
import numpy as np
import data
from skimage.io import *

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
	input_layer = tf.reshape(features["x"], [-1, 48, 48, 1])
	conv1 = tf.layers.conv2d(inputs=input_layer, filters=16, kernel_size=[5,5], padding="valid", activation=tf.nn.relu, kernel_initializer=None)
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
	norm1 = tf.nn.local_response_normalization(input=pool1, depth_radius=5, bias=1, alpha=0.05, beta=0.75)
	conv2 = tf.layers.conv2d(inputs=norm1, filters=24, kernel_size=[5,1], padding="same", activation=tf.nn.relu)
	conv3 = tf.layers.conv2d(inputs=conv2, filters=24, kernel_size=[1,5], padding="same", activation=tf.nn.relu)
	conv4 = tf.layers.conv2d(inputs=conv3, filters=32, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2,2], strides=2)
	norm2 = tf.nn.local_response_normalization(input=pool2, depth_radius=5, bias=1, alpha=0.05, beta=0.75)
	conv5 = tf.layers.conv2d(inputs=norm2, filters=64, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
	pool2_flat = tf.reshape(conv5, [-1, 11*11*64])
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
	dropout = tf.layers.dropout(inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
	logits = tf.layers.dense(inputs=dropout, units=7)
	
	predictions = {"classes":tf.argmax(input=logits, axis=1), "probabilities":tf.nn.softmax(logits, name="softmax_tensor")}
	
	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
	
	onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=7)
	loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
	
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
		train_op=optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	eval_metric_ops = {"accuracy":tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

	
def train():
	data.load()
	train_data = np.asarray(data.train_val, dtype=np.float32)
	train_labels = np.asarray(data.train_label, dtype=np.int32)
	classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./store")
	tensors_to_log = {"probabilities":"softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
	train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels, batch_size=100, num_epochs=None, shuffle=True)
	classifier.train(input_fn=train_input_fn, steps=20000, hooks=[logging_hook])
	
	
def evaluate():
	data.load()
	eval_data = np.asarray(data.test_val, dtype=np.float32)
	eval_labels = np.asarray(data.test_label, dtype=np.int32)
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
	eval_results = classifier.evaluate(input_fn=eval_input_fn)
	
	
	
'''def main(unused_argv):
	train_data = np.asarray(data.train_val, dtype=np.float32)
	train_labels = np.asarray(data.train_label, dtype=np.int32)
	eval_data = np.asarray(data.test_val, dtype=np.float32)
	eval_labels = np.asarray(data.test_label, dtype=np.int32)
	img=capture.images_of_faces
	img=np.asarray(img, dtype=np.float32)
	#img=imread('img33611.jpg')
	img.flatten()
	img=np.array([img], dtype=np.float32)
	classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./store")
	tensors_to_log = {"probabilities":"softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
	train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels, batch_size=100, num_epochs=None, shuffle=True)
	classifier.train(input_fn=train_input_fn, steps=20000, hooks=[logging_hook])
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
	eval_results = classifier.evaluate(input_fn=eval_input_fn)
	ch='y'
	while ch=='y':
		img=capture.images_of_faces
		if len(img)==0:
			print('Face not captured')
			continue
		img=np.asarray(img, dtype=np.float32)
		#img=imread('img33611.jpg')
		img.flatten()
		img=np.array([img], dtype=np.float32)
		pred_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":img}, shuffle=False)
		pred_results = classifier.predict(input_fn=pred_input_fn)
		for i in pred_results:
			emot=i['classes']
			print(emot)
			if emot==0:
				print('Angry')
			elif emot==1:
				print('Disgust')
			elif emot==2:
				print('Fear')
			elif emot==3:
				print('Happy')
			elif emot==4:
				print('Sad')
			elif emot==5:
				print('Surprise')
			else:
				print('Neutral')
		ch=input('Continue?:')
		#print(eval_results)'''

def predict(img):
	classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="./store")
	tensors_to_log = {"probabilities":"softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
	pred_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":img}, shuffle=False)
	pred_results = classifier.predict(input_fn=pred_input_fn)
	emotions=[]
	for i in pred_results:
		emot=i['classes']
		#print(emot)
		if emot==0:
			emotions.append('Angry')
		elif emot==1:
			emotions.append('Disgust')
		elif emot==2:
			emotions.append('Fear')
		elif emot==3:
			emotions.append('Happy')
		elif emot==4:
			emotions.append('Sad')
		elif emot==5:
			emotions.append('Surprise')
		else:
			emotions.append('Neutral')
	return emotions

'''if __name__=="__main__":
	tf.app.run()'''