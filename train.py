import model
import data

if __name__=="__main__":
	model.train()
	
	
	input_layer = tf.reshape(features["x"], [-1, 48, 48, 1])
	conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[7,7], padding="valid", activation=tf.nn.relu, kernel_initializer=None)
	conv2 = tf.layers.conv2d(inputs=conv1, filters=48, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
	#inception block1
	conv3a = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
	pool3b = tf.layers.max_pooling2d(inputs=conv3a, pool_size=[2,2], strides=2)
	conv4a = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
	pool4b = tf.layers.max_pooling2d(inputs=conv4a, pool_size=[2,2], strides=2)
	conv5a = tf.layers.conv2d(inputs=conv2, filters=64, kernel_size=[3,1], padding="same", activation=tf.nn.relu)
	conv5b = tf.layers.conv2d(inputs=conv5a, filters=64, kernel_size=[1,3], padding="same", activation=tf.nn.relu)
	conv5c = tf.layers.conv2d(inputs=conv5b, filters=64, kernel_size=[1,1], strides=2, padding="same", activation=tf.nn.relu)
	#block1 ends
	conc6 = tf.concat(values=[pool3b, pool4b, conv5c], axis=3)
	conv7 = tf.layers.conv2d(inputs=conc6, filters=256, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
	pool8 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[3,3], strides=2)
	#inception block2
	conv9a = tf.layers.conv2d(inputs=pool8, filters=256, kernel_size=[5,5], padding="same", activation=tf.nn.relu)
	pool9b = tf.layers.max_pooling2d(inputs=conv9a, pool_size=[2,2], strides=2)
	conv10a = tf.layers.conv2d(inputs=pool8, filters=256, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
	pool10b = tf.layers.max_pooling2d(inputs=conv10a, pool_size=[2,2], strides=2)
	conv11a = tf.layers.conv2d(inputs=pool8, filters=256, kernel_size=[3,1], padding="same", activation=tf.nn.relu)
	conv11b = tf.layers.conv2d(inputs=conv11a, filters=256, kernel_size=[1,3], padding="same", activation=tf.nn.relu)
	conv11c = tf.layers.conv2d(inputs=conv11b, filters=256, kernel_size=[1,1], strides=2, padding="same", activation=tf.nn.relu)
	#block2 ends
	conc12 = tf.concat(values=[pool9b, pool10b, conv11c], axis=3)
	conc12_flat = tf.reshape(conc12, [-1, 5*5*768])
	dense1 = tf.layers.dense(inputs=conc12_flat, units=4096, activation=tf.nn.relu)
	dropout1 = tf.layers.dropout(inputs=dense1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
	dense2 = tf.layers.dense(inputs=dropout1, units=1024, activation=tf.nn.relu)
	dropout2 = tf.layers.dropout(inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
	logits = tf.layers.dense(inputs=dropout2, units=7)