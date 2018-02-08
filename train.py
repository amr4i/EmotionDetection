import os
import sys
import json
import time
import logging
import numpy as np 
import tensorflow as tf 
from imageAVmodel import ImageAVmodel
from data_helper import batch_iter
from extract_features import get_features
from sklearn.model_selection import train_test_split
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat

# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='image-AV-emotion.log',
                    filemode='w')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logs_path = './tmp/emotion/' + datetime.now().isoformat()


'''
==================================================================================================================================
Argument 1: a document where each line contains the name of the image file, and its corresponding A-V value, separated by a comma. 
Argument 2: the folder that contains all the image files.
==================================================================================================================================
'''
def train_imageAVmodel():

	training_file = sys.argv[1]
	data_directory = sys.argv[2]

    parameter_file = "params.json"
    params = json.loads(open(parameter_file).read())

	x_raw, y_raw = get_features(training_file, data_directory, params['VGG_weights'], params['gistFile'])

	x = np.array(x_raw)
	y = np.array(y_raw)

	

	""" randomly shuffle data """
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    """ split the original dataset into train_ and test sets"""
    x_, x_test, y_, y_test = train_test_split(x_shuffled, y_shuffled, test_size=0.1, random_state=42)


    """ shuffle the train_ set and split the train set into train and val sets"""
    shuffle_indices = np.random.permutation(np.arange(len(y_)))
    x_shuffled = x_[shuffle_indices]
    y_shuffled = y_[shuffle_indices]
    x_train, x_val, y_train, y_val = train_test_split(x_shuffled, y_shuffled, test_size=0.1)

    logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

    """ build a graph """
    graph = tf.Graph()
    with graph.as_default():
    	session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
        	imageAV = ImageAVmodel(
        		input_length = x_train.shape[1],
        		num_hidden_layers = params['num_hidden_layers'],
        		num_neurons_in_layers = params['num_neurons_in_layers'] )

        	global_step = tf.Variable(0, name="global_step", trainable=False)
        	optimizer = tf.train.MomentumOptimzer(params['learning_rate'], params['momentum'])
        	grads_and_vars = optimizer.compute_gradients(imageAV.loss)
        	train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        	timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join( os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modelData/trained_model_" + timestamp))

            saved_model_dir = os.path.abspath(os.path.join(out_dir, "saved_model"))
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            # One training step: train the model with one batch
            def train_step(x_batch, y_batch):
                feed_dict = {
                    imageAV.input_x: x_batch,
                    imageAV.input_y: y_batch}
                _, step, loss, loss_S = sess.run([train_op, global_step, imageAV.lossy, imageAV.loss_summary], feed_dict)
                return loss, loss_S

            # One evaluation step: evaluate the model with one batch
            def val_step(x_batch, y_batch):
                feed_dict = {imageAV.input_x: x_batch, imageAV.input_y: y_batch}
                step, loss, loss_S = sess.run([global_step, imageAV.loss, imageAV.loss_summary], feed_dict)
                return loss, loss_S


            sess.run(tf.global_variables_initializer())

            train_batches = batch_iter(list(zip(x_train, y_train)), params['batch_size'],params['num_epochs'])
            min_loss, min_at_step = 0, 0

            """ train the cnn model with x_train and y_train (batch by batch)"""
            for train_batch in train_batches:
                x_train_batch, y_train_batch = zip(*train_batch)
                train_loss, _train_loss_summary = train_step(x_train_batch, y_train_batch)
                current_step = tf.train.global_step(sess, global_step)
                logging.debug('Train Step {}, Loss: {}'.format(current_step,train_loss))
                writer.add_summary(_train_loss_summary, current_step)
                
                """ evaluate the model with x_val and y_val (batch by batch)"""
                if current_step % params['evaluate_every'] == 0:
                    val_batches = batch_iter(list(zip(x_val, y_val)), params['batch_size'], 1)
                    total_val_loss = 0.0
                    for val_batch in val_batches:
                        x_val_batch, y_val_batch = zip(*dev_batch)
                        val_loss, _val_loss_summary = val_step(x_val_batch, y_val_batch)
                        total_val_loss += val_loss

                    writer.add_summary(_val_loss_summary, current_step)
                    logging.info('Total loss on val set: {}'.format(total_val_loss))
                    logging.info('Average loss on val set: {}'.format(total_val_loss/len(y_val)))

                    """ save the model if it is the best based on loss on the val set """
                    if total_val_loss <= min_loss and not min_loss == 0:
                        min_loss, min_at_step = total_val_loss, current_step

                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logging.info('Saved model {} at step {}'.format(path, min_at_step))
                        logging.info('Best accuracy {} at step {}'.format(min_loss, min_at_step))

            """ predict x_test (batch by batch)"""
            test_batches = batch_iter(list(zip(x_test, y_test)), params['batch_size'], 1)
            total_test_loss = 0.0
            for test_batch in test_batches:
                x_test_batch, y_test_batch = zip(*test_batch)
                test_loss, _test_loss_summary = val_step(x_test_batch, y_test_batch)
                total_test_loss += test_loss

            avg_test_loss = total_test_loss/len(y_test)
            logging.info('Total loss on the test set is {} based on the best model {}'.format(total_test_loss, path))
            logging.info('Average loss on test set is {} based on the best model {}'.format(avg_test_loss, path))
            logging.info('The training is complete.')

            """ saving the model """
            save_path = saver.save(sess, saved_model_dir)
            logging.info('The model has been saved at path: {}'.foramt(save_path))

if __name__ == '__main__':
    train_imageAVmodel()