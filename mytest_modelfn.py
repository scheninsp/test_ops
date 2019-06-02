import tensorflow as tf

slim = tf.contrib.slim


def model_fn(inputs):
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('')

    laplacian_kernel = [1, 1, 1, 1, -8, 1, 1, 1, 1]

    conv1 = slim.separable_conv2d(
        inputs,
        num_outputs=None,
        kernel_size=3,
        depth_multiplier=1,
        rate=2,
        # weights_initializer=tf.truncated_normal_initializer(stddev=0.33),
        weights_initializer=tf.constant_initializer(laplacian_kernel),
        weights_regularizer=None,
        scope='Sepconv2d')

    outputs = tf.identity(conv1, name="NetOutput")

    return outputs

