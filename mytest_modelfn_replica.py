import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

def model_fn(inputs):
	input_shape = inputs.get_shape().as_list()
	if len(input_shape) != 4:
		raise ValueError('input shape error')

	with tf.variable_scope("depthwise_conv2d_1", reuse=tf.AUTO_REUSE):
		lp_base = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]
		lp_base = lp_base / np.linalg.norm(lp_base)
		laplacian_kernel = np.zeros((3, 3, 3), dtype=np.float32)
		for i in range(3):
			laplacian_kernel[:, :, i] = lp_base
		laplacian_kernel = np.expand_dims(laplacian_kernel, axis=3)

		weights = tf.get_variable(name='laplacian_kernel', shape=laplacian_kernel.shape,
							  dtype=tf.float32, initializer=tf.constant_initializer(laplacian_kernel))
<<<<<<< HEAD

		conv1 = tf.nn.depthwise_conv2d(inputs, filter=weights, strides=(1, 1, 1, 1),
									padding='SAME', rate=(2, 2), data_format='NHWC')

	outputs = tf.identity(conv1, name="NetOutput")
=======

		conv1 = tf.nn.depthwise_conv2d(inputs, filter=weights, strides=(1, 1, 1, 1),
									padding='SAME', rate=(2, 2), data_format='NHWC')

		outputs = tf.identity(conv1, name="NetOutput")
>>>>>>> origin/master

	return outputs