# mytest_inference.py
import tensorflow as tf
import os

if __name__ == "__main__":
	os.environ['CUDA_VISIBLE_DEVICE'] = "0"
	output_path = "./outputs"
	input_path = "./inputs"

	N_EPOCHS=1

	with open(os.path.join(input_path, "input_list.txt")) as f:
		lines = f.readlines()
		img_tr = [x.split('\t')[0] for x in lines]
		lbs_tr = [x.split('\t')[1].strip() for x in lines]

	tf.reset_default_graph()

	with tf.Session as sess:
		saver_restore = tf.train.import_meta_graph(os.path.join(output_path, 'model.ckpt'))
		saver_restore.restore(sess, tf.train.latest_checkpoint(output_path))

		innode = sess.graph.get_tensor_by_name("ImageTensor:0")
		input_shape = innode.shape.as_list()
		outnode = sess.graph.get_tensor_by_name("depthwise_conv2d_1/NetOutput:0")
		
		with tf.name_scope('load_data'):
			image_filename_q, label_filename_q = tf.train.slice_input_producer([img_tr, lbs_tr], num_epochs=N_EPOCHS, shuffle=False)
			im_q = tf.read_file(image_filename_q)
			im_q = tf.image.decode_image(im_q)
			im_q = tf.cast(im_q, tf.float32)

			lb_q = tf.read_file(label_filename_q)
			lb_q = tf.image.decode_image(lb_q)

			im_q = tf.expand_dims(im_q, 0)
			lb_q = tf.expand_dims(lb_q, 0)

			image = tf.image.resize_bilinear(im_q, input_shape[1:3], align_corner=False)
			label = tf.image.resize_nearest_neighbor(lb_q, input_shape[1:3], align_corner=False)

			# image = image[0]
			# image.set_shape(input_shape[1:])
			# label = label[0,:,:,0]
			# label.set_shape(input_shape[1:3])
			# image_batch = tf.train.batch([image], batch_size = 1)
			# image_batch is useless , use placeholder saved in meta graph 

			# slice_input_producer 'num_epochs' is a local variable that needs initialize
			sess.run(tf.local_variables_initializer()) # local initializer must inside name_scope

		coord = tf.train.Coordinator()
		thread = tf.train.start_queue_runners(sess=sess, coord=coord)

		try:
			while not coord.should_stop():
				results = sess.run(outnode, feed_dict = {innode : image.eval()})
				print(results.shape)
		except tf.errors.OutOfRangeError:
			print("done")
		finally:
			coord.request_stop()
		coord.join(thread)
