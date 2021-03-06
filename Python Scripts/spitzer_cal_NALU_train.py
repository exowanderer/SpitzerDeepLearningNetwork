from multiprocessing import set_start_method, cpu_count
#set_start_method('forkserver')

import os
os.environ["OMP_NUM_THREADS"] = str(cpu_count())  # or to whatever you want

from argparse import ArgumentParser
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
from tqdm import tqdm

import pandas as pd
import numpy as np
import tensorflow as tf
import nalu

time_now = datetime.utcnow().strftime("%Y%m%d%H%M%S")

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def chisq(y_true, y_pred, y_error): return np.sum(((y_true-y_pred)/y_error)**2.)

ap = ArgumentParser()

ap.add_argument('-d', '--directory', type=str, required=False, default='nalu_tf_save_dir/saves_{}'.format(time_now), help='The tensorflow ckpt save file.')
ap.add_argument('-nnl', '--n_nalu_layers', type=int, required=False, default=1, help='Whether to use 1 (default), 2, or ... N NALU layers.')
ap.add_argument('-nnn', '--n_nalu_neurons', type=int, required=False, default=0, help='How many features on the second NALU layer.')
ap.add_argument('-ne', '--n_epochs', type=int, required=False, default=200, help='Number of N_EPOCHS to train the network with.')
ap.add_argument('-nc', '--n_classes', type=int, required=False, default=1, help='n_classes == 1 for Regression (default); > 1 for Classification.')
ap.add_argument('-bs', '--batch_size', type=int, required=False, default=32, help='Batch size: number of samples per batch.')
ap.add_argument('-lr', '--learning_rate', type=float, required=False, default=1e-3, help='Learning rate: how fast the optimizer moves up/down the gradient.')
ap.add_argument('-ts', '--test_size', type=float, required=False, default=0.75, help='How much to split the train / test ratio.')
ap.add_argument('-rs', '--random_state', type=int, required=False, default=42, help='Integer value to initialize train/test splitting randomization.')
ap.add_argument('-pp', '--pre_process', type=str2bool, nargs='?', required=False, default=True, help='Toggle whether to MinMax-preprocess the features.')
ap.add_argument('-pca', '--pca_transform', type=str2bool, nargs='?', required=False, default=True, help='Toggle whether to PCA-pretransform the features.')
ap.add_argument('-v', '--verbose', type=str2bool, nargs='?', required=False, default=False, help='Whether to set verbosity = True or False (default).')
ap.add_argument('-ds', '--data_set', type=str, required=False, default='', help='The csv file containing the data with which to train.')

try:
    args = vars(ap.parse_args())
except Exception as e:
    print('Error: {}'.format(e))
    args = {}
    args['directory'] = ap.get_default('directory')
    args['n_nalu_layers'] = ap.get_default('n_nalu_layers')
    args['n_nalu_neurons'] = ap.get_default('n_nalu_neurons')
    args['n_epochs'] = ap.get_default('n_epochs')
    args['n_classes'] = ap.get_default('n_classes')
    args['batch_size'] = ap.get_default('batch_size')
    args['learning_rate'] = ap.get_default('learning_rate')
    args['test_size'] = ap.get_default('test_size')
    args['random_state'] = ap.get_default('random_state')
    args['pre_process'] =  ap.get_default('pre_process')
    args['pca_transform'] =  ap.get_default('pca_transform')
    args['verbose'] = ap.get_default('verbose')
    args['data_set'] = ap.get_default('data_set')

DO_PP = args['pre_process']
DO_PCA = args['pca_transform']

verbose = args['verbose']
data_set_fname = args['data_set']

'''
print("loading pipelines on disk vis joblib.")
full_pipe = joblib.load('pmap_full_pipe_transformer_16features.joblib.save')
std_scaler_from_raw = joblib.load('pmap_standard_scaler_transformer_16features.joblib.save')
pca_transformer_from_std_scaled = joblib.load('pmap_pca_transformer_from_stdscaler_16features.joblib.save')
minmax_scaler_transformer_raw = joblib.load('pmap_minmax_scaler_transformer_from_raw_16features.joblib.save')
minmax_scaler_transformer_pca = joblib.load('pmap_minmax_scaler_transformer_from_pca_16features.joblib.save')
'''

label_n_error_filename = 'pmap_raw_labels_and_errors.csv'
print("Loading in raw labels and errors from {}".format(label_n_error_filename))
labels_df = pd.read_csv(label_n_error_filename)

labels = labels_df['Flux'].values[:,None]
labels_err = labels_df['Flux_err'].values

# Feature File Switch
if DO_PP and DO_PCA:
    features_input_filename = 'pmap_full_pipe_transformed_16features.csv'
elif DO_PP:
    features_input_filename = 'pmap_minmax_transformed_from_raw_16features.csv'
elif DO_PCA:
    features_input_filename = 'pmap_pca_transformed_from_stdscaler_16features.csv'
else:
    features_input_filename = 'pmap_raw_16features.csv'

print("Loading in pre-processed features from {}".format(features_input_filename))
features_input = pd.read_csv(feature_input_filename).drop(['idx'], axis=1).values

def nalu(input_layer, num_outputs):
    """ Neural Arithmetic Logic Unit tesnorflow layer
    
    Arguments:
    input_layer - A Tensor representing previous layer
    num_outputs - number of ouput units 
    
    Returns:
    A tensor representing the output of NALU
    """
    
    shape = (int(input_layer.shape[-1]), num_outputs)
    
    # define variables
    W_hat = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
    M_hat = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
    G = tf.Variable(tf.truncated_normal(shape, stddev=0.02))
    
    # operations according to paper
    W = tf.tanh(W_hat) * tf.sigmoid(M_hat)
    m = tf.exp(tf.matmul(tf.log(tf.abs(input_layer) + 1e-7), W))
    g = tf.sigmoid(tf.matmul(input_layer, G))
    a = tf.matmul(input_layer, W)
    out = g * a + (1 - g) * m
    
    return out

if __name__ == "__main__":
    N_FEATURES = features_input.shape[-1]
    
    EXPORT_DIR = args['directory']
    N_NALU_LAYERS = args['n_nalu_layers']
    N_NALU_NEURONS = args['n_nalu_neurons'] if args['n_nalu_neurons'] > 0 else N_FEATURES
    N_CLASSES = args['n_classes'] # = 1 for regression
    TEST_SIZE = args['test_size']
    RANDOM_STATE = args['random_state']
    
    N_EPOCHS = args['n_epochs']
    LEARNING_RATE = args['learning_rate']
    BATCH_SIZE = args['batch_size']
    
    EXPORT_DIR = EXPORT_DIR + '_nnl{}_nnn{}_nc{}_bs{}_lr{}_ne{}_ts{}_rs{}_PP{}_PCA{}/'.format(N_NALU_LAYERS, N_NALU_NEURONS, N_CLASSES, 
                                                                                              BATCH_SIZE, LEARNING_RATE, N_EPOCHS, 
                                                                                              TEST_SIZE, RANDOM_STATE, 
                                                                                              {True:1, False:0}[DO_PP], {True:1, False:0}[DO_PCA])
    
    print("Saving models to path: {}".format(EXPORT_DIR))
    
    idx_train, idx_test = train_test_split(np.arange(labels.size), test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    X_data, Y_data = features_input[idx_train], labels[idx_train]#[:,None]
    
    LAST_BIT = X_data.shape[0]-BATCH_SIZE*(X_data.shape[0]//BATCH_SIZE)
    
    # Force integer number of batches total by dropping last "<BATCH_SIEZ" number of samples
    X_data_use = X_data[:-LAST_BIT].copy()
    Y_data_use = Y_data[:-LAST_BIT].copy()
    
    output_dict = {}
    output_dict['loss'] = np.zeros(N_EPOCHS)
    output_dict['accuracy'] = np.zeros(N_EPOCHS)
    output_dict['R2_train'] = np.zeros(N_EPOCHS)
    output_dict['R2_test'] = np.zeros(N_EPOCHS)
    output_dict['chisq_train'] = np.zeros(N_EPOCHS)
    output_dict['chisq_test'] = np.zeros(N_EPOCHS)
    
    with tf.device("/cpu:0"):
        # tf.reset_default_graph()
        
        # define placeholders and network
        X = tf.placeholder(tf.float32, shape=[None, N_FEATURES])
        Y_true = tf.placeholder(tf.float32, shape=[None, 1])
        
        # Setup NALU Layers
        nalu_layers = {'nalu0':nalu(X,N_NALU_NEURONS)}
        for kn in range(1, N_NALU_LAYERS):
            #with tf.name_scope('nalu{}'.format(kn)):
            nalu_layers['nalu{}'.format(kn)] = nalu(nalu_layers['nalu{}'.format(kn-1)], N_NALU_NEURONS)
        
        # with tf.name_scope("output"):
        Y_pred = nalu(nalu_layers['nalu{}'.format(N_NALU_LAYERS-1)], N_CLASSES) # N_CLASSES = 1 for regression
        
        #with tf.name_scope('loss'):
        # loss and train operations
        loss = tf.nn.l2_loss(Y_pred - Y_true) # NALU uses mse
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
        train_op = optimizer.minimize(loss)
        
        # Add an op to initialize the variables.
        init_op = tf.global_variables_initializer()
        
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()#max_to_keep=N_EPOCHS)
    
    sess_config = tf.ConfigProto(
                    device_count={"CPU": cpu_count()},
                    inter_op_parallelism_threads=cpu_count(),
                    intra_op_parallelism_threads=cpu_count())
    """
    with tf.name_scope("eval"):
        ''' Tensorboard Redouts'''
        ''' Training R-Squared Score'''
        total_error = tf.reduce_sum(tf.square(tf.subtract(Y_true, tf.reduce_mean(Y_true))))
        unexplained_error = tf.reduce_sum(tf.square(tf.subtract(Y_true, Y_pred)))
        R_squared = tf.subtract(1.0, tf.div(unexplained_error, total_error))        
        
        # ''' Testing R-Squared Score'''
        # Y_pred_test = Y_pred.eval(feed_dict={X: features[idx_test]})
        # total_error_test = tf.reduce_sum(tf.square(tf.subtract(Y_data_use, tf.reduce_mean(Y_data_use))))
        # unexplained_error_test = tf.reduce_sum(tf.square(tf.subtract(Y_data_use, Y_pred_test)))
        # R_squared_test = tf.subtract(1, tf.div(unexplained_error, total_error))
        
        ''' Loss and RMSE '''
        squared_error = tf.square(tf.subtract(Y_true, Y_pred))
        loss = tf.reduce_sum(tf.sqrt(tf.cast(squared_error, tf.float32)))
        rmse = tf.sqrt(tf.reduce_mean(tf.cast(squared_error, tf.float32)))
    
    with tf.name_scope("summary"):
        ''' Declare Scalar Tensorboard Terms'''
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('RMSE', rmse)
        tf.summary.scalar('R_sqrd', R_squared)
        
        ''' Declare Histogram Tensorboard Terms'''
        # Squared Error Histogram
        tf.summary.histogram('SqErr Hist', squared_error)
        
        # NALU Layers Histogram
        for kn in range(N_NALU_LAYERS):
            tf.summary.histogram('NALU{}'.format(kn), nalu_layers['nalu{}'.format(kn)])
        
        ''' Merge all the summaries and write them out to `export_dir` + `/logs_train_`time_now`` '''
        merged = tf.summary.merge_all()
    """
    with tf.Session(config=sess_config) as sess:
        ''' Output all summaries to `export_dir` + `/logs_train_`time_now`` '''
        train_writer = tf.summary.FileWriter(EXPORT_DIR + '/logs_train_{}'.format(time_now),sess.graph)
        ''' END Tensorboard Readout Step'''
        
        sess.run(init_op)
        
        best_test_r2 = 0
        for ep in tqdm(range(N_EPOCHS)):
            i = 0
            gts = 0
            
            # Reshuffle features and labels together
            X_data_use, Y_data_use = shuffle(X_data_use, Y_data_use)
            
            for k in tqdm(range(len(X_data_use)//BATCH_SIZE)):
                batch_now = range(k*BATCH_SIZE, (k+1)*BATCH_SIZE)
                # xs, ys = X_data_use[i:i+BATCH_SIZE], Y_data_use[i:i+BATCH_SIZE]
                xs, ys = X_data_use[batch_now], Y_data_use[batch_now]
                
                _, ys_pred, l = sess.run([train_op, Y_pred, loss], 
                        feed_dict={X: xs, Y_true: ys})
                
                # calculate number of correct predictions from batch
                gts += np.sum(np.isclose(ys, ys_pred, atol=1e-4, rtol=1e-4)) 
            
            ytest_pred = Y_pred.eval(feed_dict={X: features_input[idx_test]})
            if np.isnan(ytest_pred).any(): ytest_pred = median_sub_nan(ytest_pred)
            
            test_r2 = r2_score(labels[idx_test], ytest_pred)#[:,None]
            # print("Test R2 Score: {}".format(test_r2_score))
            
            acc = gts/len(Y_data_use)
            train_r2 = r2_score(ys, ys_pred)
            print('\nepoch {}, loss: {:.5}, accuracy: {:.5}, Batch R2: {:.5}, Test R2: {:.5}'.format(ep, l, acc, train_r2, test_r2))
            
            """
            output_dict['loss'][ep] = l
            output_dict['accuracy'][ep] = acc
            output_dict['R2_train'][ep] = train_r2
            output_dict['R2_test'][ep] = test_r2
            output_dict['chisq_train'][ep] = chisq(ys.flatten(), ys_pred.flatten(), labels_err[i:i+BATCH_SIZE])
            output_dict['chisq_test'][ep] = chisq(labels[idx_test], ytest_pred.flatten(), labels_err[idx_test])
            """
            
            if verbose: print('Saving Default to Disk')
            
            now_save_name = EXPORT_DIR + "model_epoch{}_l{:.5}_a{:.5}_BatchR2-{:.5}_TestR2-{:.5}.ckpt".format(  ep, l, acc, train_r2, test_r2)
            save_path = saver.save(sess, now_save_name)
            
            if test_r2 >= best_test_r2:
                if verbose: print('Saving Best to Disk')
                
                best_test_r2 = test_r2
                ''' Store the Best Scored Test-R2 '''
                best_save_name = EXPORT_DIR + "best_test_r2/model_epoch{}_l{:.5}_a{:.5}_BatchR2-{:.5}_TestR2-{:.5}.ckpt".format( ep, l, acc, train_r2, test_r2)
                save_path = saver.save(sess, best_save_name)
        
        ep = '_FINAL'
        final_save_name = EXPORT_DIR+ "model_epoch{}_l{:.5}_a{:.5}_BatchR2-{:.5}_TestR2-{:.5}.ckpt".format(   ep, l, acc, train_r2, test_r2)
        save_path = saver.save(sess, final_save_name)
        
        print("Model saved in path: {}".format(save_path))
        
        """
        try:
            pd.DataFrame(output_dict, index=range(N_EPOCHS)).to_csv(EXPORT_DIR+ "model_loss_acc_BatchR2_TestR2_DataFrame.csv")
        except Exception as e:
            print('DataFrame to CSV broke because', str(e))
        """

'''

with tf.name_scope("loss"):
    def tf_nll(labels, output, uncs, coeff=1):
        error = output - labels
        return tf.reduce_sum(tf.divide(tf.squared_difference(output, labels) , tf.square(uncs)))# + tf.log(tf.square(uncs))
        #return tf.reduce_sum(1 * (coeff * np.log(2*np.pi) + coeff * tf.log(uncs) + (0.5/uncs) * tf.pow(error, 2)))
    
    negloglike  = tf_nll(labels=y, output=output, uncs=unc)
    
    reg_losses  = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    loss        = tf.add_n([negloglike] + reg_losses, name="chisq")

with tf.name_scope("eval"):
    accuracy  = tf.reduce_mean(tf.squared_difference(output, y, name="accuracy"))
    
    SqErrRatio= tf.divide(accuracy,  tf.reduce_mean(tf.squared_difference(y, tf.reduce_mean(y))))
    r2_acc    = 1.0 - SqErrRatio
    
    chsiqMean = tf_nll(labels=y, output=tf.reduce_mean(y), uncs=unc)
    chisqModel= tf_nll(labels=y, output=output, uncs=unc)
    rho2_acc  = 1.0 - chisqModel / chsiqMean"

mse_summary  = tf.summary.scalar('train_acc' , accuracy  )
loss_summary = tf.summary.scalar('loss'      , loss      )
nll_summary  = tf.summary.scalar('negloglike', negloglike)
r2s_summary  = tf.summary.scalar('r2_acc'    , r2_acc    )
p2s_summary  = tf.summary.scalar('rho2_acc'  , rho2_acc  )
val_summary  = tf.summary.scalar('val_acc'   , accuracy  )

# hid1_hist    = tf.summary.histogram('hidden1', hidden1)
# hid2_hist    = tf.summary.histogram('hidden1', hidden1)
# hid3_hist    = tf.summary.histogram('hidden1', hidden1)

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

'''
