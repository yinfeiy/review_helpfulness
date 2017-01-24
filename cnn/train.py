#! /usr/bin/env python

import sys
sys.path.append('../classifier/')
from utils import *

import tensorflow as tf
import numpy as np
import os, time, datetime, copy
import gensim
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from tensorflow.contrib.tensorboard.plugins import projector

# Model Hyperparameters
tf.flags.DEFINE_string("genre", "outdoor", "Genre (default: outdoor)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.1, "L2 regularizaion lambda (default: 0.1)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 50, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 5000, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

genre = FLAGS.genre

data_path = '../dataset_v2/t5/{0}'
cache_path = '../dataset_v2/t5/cache/{0}'

data_path_genre = data_path.format( genre )
cache_path_genre = cache_path.format( genre )

fname_fold = os.path.join(data_path_genre, 'folds.txt')
fname_score = os.path.join(data_path_genre, 'labels.txt')
fname_review = os.path.join(data_path_genre, 'reviews.txt')

folds = [ int(f) for f in get_content(fname_fold) ]
scores = read_scores_from_file_2(fname_score)
reviews = get_content(fname_review)

# Load data
print ('Loading Data...')

fold_ids = range(10)

max_document_length = max([len(x.split(" ")) for x in reviews])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
vocab_processor.fit(reviews)

x_all = np.array(list(vocab_processor.transform(reviews)))
x_all = np.array(x_all)

for fold_id in fold_ids:
    ###################################################################
    # partition
    y_train, x_train = get_train_data(scores, x_all, folds, fold_id)
    y_test, x_test = get_test_data(scores, x_all, folds, fold_id)
    y_dev, x_dev = get_dev_data(scores, x_all, folds, fold_id)

    x_train.extend(x_dev)
    y_train.extend(y_dev)

    x_train = np.array(x_train)
    y_train = np.array([ [y] for y in y_train])
    x_dev = np.array(x_test)
    y_dev = np.array(y_test)

    y_train = np.array(y_train); y_dev = np.array(y_dev)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)

        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn=TextCNN(sequence_length = x_train.shape[1],
                    num_classes = y_train.shape[1],
                    vocab_size = len(vocab_processor.vocabulary_),
                    embedding_size = FLAGS.embedding_dim,
                    filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda = FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs_info", timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            loss_summary = tf.scalar_summary("loss", cnn.loss)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, grad_summaries_merged])
            train_summary_dir = out_dir #os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

            config_pro = projector.ProjectorConfig()
            embedding = config_pro.embeddings.add()
            embedding.tensor_name = cnn.embedding.name
            embedding.metadata_path = os.path.join(out_dir, 'vocab_raw')
            projector.visualize_embeddings(train_summary_writer, config_pro)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = out_dir #os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver()#tf.global_variables())

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))
            vks = vocab_processor.vocabulary_._reverse_mapping
            with open(out_dir + '/vocab_raw', 'w+') as fout:
                for v in vks:
                    fout.write(v+'\n')

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            ## Initialize word_embedding
            print ('Loading w2v model...')
            #w2v_model = gensim.models.Word2Vec.load_word2vec_format('~/workspace/nlp/word2vec/models/GoogleNews-vectors-negative300.bin', binary=True)
            w2v_model = gensim.models.Word2Vec.load_word2vec_format('~/workspace/nlp/word2vec/models/vectors-reviews-restaurants.bin', binary=True)
            print ('Load w2v model done.')

            W_init = []
            for v in vks:
                try:
                    v_vec = w2v_model[v]
                except:
                    v_vec = np.random.uniform(-1, 1, 300)
                W_init.append(v_vec)
            W_init = np.array(W_init)
            sess.run(cnn.embedding.assign(W_init))

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss= sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                if step % 1 == 0:
                    print("{}: step {}, loss {:g}".format(time_str, step, loss))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss = sess.run(
                    [global_step, dev_summary_op, cnn.loss],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}".format(time_str, step, loss))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev)#, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                if current_step % 10000 == 0:
                    break
