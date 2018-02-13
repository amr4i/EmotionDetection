import sys
import json
from extract_features import get_test_features

parameter_file = "test_params.json"
params = json.loads(open(parameter_file).read())

x_pred = get_prediction_features(training_file, data_directory, params['vgg_file'], params['gistFile'], params['semF_file'])

timestamp = sys.argv[1]

out_dir = os.path.abspath(os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modelData/trained_model_" + timestamp))
checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))

with tf.Session() as sess:    
	saver = tf.train.import_meta_graph(checkpoint_dir)
	saver.restore(sess,tf.train.latest_checkpoint('./'))

	x_pred = np.reshape(x_pred, (len(x_pred), 1))
	output = sess.run(prediction , feed_dict={ imageAV.input_x: batch_x})

	print output
