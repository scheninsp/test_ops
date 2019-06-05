import tensorflow as tf
import numpy as np

slim = tf.contrib.slim


def model_fn(inputs):
	input_shape = inputs.get_shape().as_list()
	if len(input_shape) != 4:
		raise ValueError('')

	with tf.variable_scope("depthwise_conv2d_1", reuse=tf.AUTO_REUSE):

		# laplacian_kernel = [1, 1, 1, 1, -8, 1, 1, 1, 1]
		lp_base = [[1,1,1], [1, -8, 1], [1,1,1]]
		lp_base = lp_base / np.linalg.norm(lp_base)
		laplacian_kernel = np.expand_dims(lp_base, axis=2)
		laplacian_kernel = np.expand_dims(laplacian_kernel, axis=3)

		weights = tf.get_variable(name='laplacian_kernel', shape=laplacian_kernel.shape,
									dtype=tf.float32,
									initializer=tf.constant_initializer(laplacian_kernel))

		# conv1 = tf.nn.depthwise_conv2d(inputs. filter=weights, strides=(1, 1, 1, 1),
		# 								padding='SAME', rate=(2, 2), data_format='NHWC')

		inputs = tf.split(inputs, 3, axis=3)
		conv2=[]
		for i in range(3):
			conv2.append(tf.nn.conv2d(inputs[i], filter=weights, strides=[1,1,1,1], padding='SAME',
						  dilations=[1,2,2,1], data_format='NHWC'))

		tensor_concat=tf.concat([conv2[0], conv2[1]], 3)
		tensor_concat = tf.concat([tensor_concat, conv2[2]], 3)

		outputs = tf.identity (tensor_concat, name="NetOutput")

	return outputs

