import os
import tensorflow as tf

from mytest_modelfn_replica_v2 import model_fn

slim = tf.contrib.slim


def main(_):
    # filenames = ['./datasets/pascal_voc_seg/tfrecord/val-00000-of-00004.tfrecord']
    output_path = './outputs_replica'

    _batch_size = 1

    image = tf.placeholder(tf.float32, shape=[1, 513, 513, 3], name="ImageTensor")

    with tf.device('/device:GPU:0'):
        outputs = model_fn(image)

    persist_saver = tf.train.Saver(max_to_keep=None)
    initializers = tf.global_variables_initializer()

    summary_writer = tf.summary.FileWriter(output_path)

    with tf.Session() as sess:
        sess.run(initializers)
        persist_saver.save(sess, os.path.join(output_path, 'model.ckpt'))
        summary_writer_graph = tf.summary.FileWriter('./visualize_replica', sess.graph)  # save graph for tensorboard

    summary_writer.close()
    summary_writer_graph.close()


if __name__ == "__main__":
    tf.app.run()

