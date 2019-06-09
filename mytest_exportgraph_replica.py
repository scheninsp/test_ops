import tensorflow as tf
import os

# for freeze_graph_with_def_protos
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.client import session
from tensorflow.python.training import saver as saver_lib


# TODO: Replace with freeze_graph.freeze_graph_with_def_protos when
# newer version of Tensorflow becomes more common.
def freeze_graph_with_def_protos(
        input_graph_def,
        input_saver_def,
        input_checkpoint,
        output_node_names,
        restore_op_name,
        filename_tensor_name,
        clear_devices,
        initializer_nodes,
        variable_names_blacklist=''):
    """Converts all variables in a graph and checkpoint into constants."""
    del restore_op_name, filename_tensor_name  # Unused by updated loading code.

    # 'input_checkpoint' may be a prefix if we're using Saver V2 format
    if not saver_lib.checkpoint_exists(input_checkpoint):
        raise ValueError(
            "Input checkpoint ' + input_checkpoint + ' does not exist!")

    if not output_node_names:
        raise ValueError(
            'You must supply the name of a node to --output_node_names.')

    # Remove all the explicit device specifications for this node. This helps
    # to make the graph more portable.
    if clear_devices:
        for node in input_graph_def.node:
            node.device = ''

    with tf.Graph().as_default():
        tf.import_graph_def(input_graph_def, name='')
        config = tf.ConfigProto(graph_options=tf.GraphOptions())
        with session.Session(config=config) as sess:
            if input_saver_def:
                saver = saver_lib.Saver(saver_def=input_saver_def)
                saver.restore(sess, input_checkpoint)
            else:
                var_list = {}
                reader = pywrap_tensorflow.NewCheckpointReader(
                    input_checkpoint)
                var_to_shape_map = reader.get_variable_to_shape_map()
                for key in var_to_shape_map:
                    try:
                        tensor = sess.graph.get_tensor_by_name(key + ':0')
                    except KeyError:
                        # This tensor doesn't exist in the graph (for example
                        # it's 'global_step' or a similar housekeeping element)
                        # so skip it.
                        continue
                    var_list[key] = tensor
                saver = saver_lib.Saver(var_list=var_list)
                saver.restore(sess, input_checkpoint)
                if initializer_nodes:
                    sess.run(initializer_nodes)

            variable_names_blacklist = (variable_names_blacklist.split(',') if
                                        variable_names_blacklist else None)
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                output_node_names.split(','),
                variable_names_blacklist=variable_names_blacklist)
    return output_graph_def


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICE'] = "0"
    output_path = './outputs_replica'

    checkpoint_to_use = os.path.join(output_path, 'model.ckpt')
    output_graph_path = './export/exported_graph_replica_v2.pb'
    output_node_names = 'NetOutput'

    tf.reset_default_graph()

    with tf.Session() as sess:
        saver_restore = tf.train.import_meta_graph(os.path.join(output_path, 'model.ckpt.meta'))
        saver_restore.restore(sess, tf.train.latest_checkpoint(output_path))

        saver_export = tf.train.Saver(max_to_keep=10)

        frozen_graph_def = freeze_graph_with_def_protos(
            input_graph_def=tf.get_default_graph().as_graph_def(add_shapes=True),
            input_saver_def=saver_export.as_saver_def(),
            input_checkpoint=checkpoint_to_use,
            output_node_names=output_node_names,
            restore_op_name=None,
            filename_tensor_name=None,
            clear_devices=True,
            initializer_nodes=None)

        with gfile.GFile(output_graph_path, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())


