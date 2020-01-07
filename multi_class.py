# -*- coding: utf-8 -*-
"""Multi-class classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19u73phW1PF_3PeJf6LrTWeyA--PgK7PH
"""

import datetime
import os
import pdb
import pprint
import sys

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics import f1_score, precision_score, recall_score

import classifier
import classifier_with_tfhub
from utils import evaluate, format_labels, input_features, io, tokenization
from utils.analysis import plot_category_distribution

DATA_ID = 'PLAN'
DATA_ID = 'UPR'

with tf.Session() as session:
    pprint.pprint(session.list_devices())

BUCKET = 'bert_classification_models'
OUTPUT_DIR = 'gs://{}/models/multilabel_{}'.format(BUCKET, DATA_ID)

BERT_MODEL = 'uncased_L-12_H-768_A-12'
BERT_MODEL_HUB = 'https://tfhub.dev/google/bert_' + BERT_MODEL + '/1'

USE_TPU = False

cfg = io.load_yml('./config.yml', DATA_ID)
data = io.load_pickle(cfg['pkl_file'])

plot_category_distribution(data)

all_labels = format_labels.get_unique(data.label.tolist())
tokenizer = classifier_with_tfhub.create_tokenizer_from_hub_module(
    BERT_MODEL_HUB)

train_values = data.sample(frac=0.7, random_state=72)[:100]
test_values = data.drop(train_values.index)[:20]

TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 8
PREDICT_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 1.0
MAX_SEQ_LENGTH = 128
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 500

# Compute number of train and warmup steps from batch size
#processor = multilabel')
train_examples = input_features.create_examples(train_values, 'train',
                                                'multilabel')
num_train_steps = int(len(train_examples) / TRAIN_BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

# Setup TPU related config
#tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(TPU_ADDRESS)
NUM_TPU_CORES = 8
ITERATIONS_PER_LOOP = 1000


def get_run_config(output_dir):
    return tf.contrib.tpu.RunConfig(
        #cluster=tpu_cluster_resolver,
        model_dir=output_dir,
        save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=ITERATIONS_PER_LOOP,
            num_shards=NUM_TPU_CORES,
            per_host_input_for_training=tf.contrib.tpu.InputPipelineConfig
            .PER_HOST_V2))


# Force TF Hub writes to the GS bucket we provide.
os.environ['TFHUB_CACHE_DIR'] = OUTPUT_DIR

model_fn = classifier_with_tfhub.model_fn_builder(
    num_labels=len(all_labels),
    learning_rate=LEARNING_RATE,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=USE_TPU,
    bert_hub_module_handle=BERT_MODEL_HUB)

estimator_from_tfhub = tf.contrib.tpu.TPUEstimator(
    use_tpu=USE_TPU,
    model_fn=model_fn,
    config=get_run_config(OUTPUT_DIR),
    train_batch_size=TRAIN_BATCH_SIZE,
    eval_batch_size=EVAL_BATCH_SIZE,
    predict_batch_size=PREDICT_BATCH_SIZE,
)


# Train the model
def model_train(estimator):
    train_features = classifier.convert_examples_to_features(
        train_examples, all_labels, MAX_SEQ_LENGTH, tokenizer)
    print('***** Started training at {} *****'.format(datetime.datetime.now()))
    print('  Num examples = {}'.format(len(train_examples)))
    print('  Batch size = {}'.format(TRAIN_BATCH_SIZE))
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = classifier.input_fn_builder(features=train_features,
                                                 seq_length=MAX_SEQ_LENGTH,
                                                 is_training=True,
                                                 drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
    print('***** Finished training at {} *****'.format(datetime.datetime.now()))


model_train(estimator_from_tfhub)


def model_predict(estimator):
    # Make predictions on a subset of eval examples
    prediction_examples = input_features.create_examples(test_values, 'test')
    input_features2 = classifier.convert_examples_to_features(
        prediction_examples, all_labels, MAX_SEQ_LENGTH, tokenizer)
    predict_input_fn = classifier.input_fn_builder(features=input_features2,
                                                   seq_length=MAX_SEQ_LENGTH,
                                                   is_training=False,
                                                   drop_remainder=False)
    predictions = estimator.predict(predict_input_fn)

    return [prediction['probabilities'] for prediction in predictions]


results = model_predict(estimator_from_tfhub)

test_values['pred_label'] = [all_labels[elem.argmax()] for elem in results]
test_values['pred_prob'] = [elem.max() for elem in results]

test_values[['text', 'label', 'pred_label', 'pred_prob']][:20]


def eval_category(target, prediction):
    prec = round(precision_score(target, prediction), 4)
    rec = round(recall_score(target, prediction), 4)
    f1 = round(f1_score(target, prediction), 4)
    return {'prec': prec, 'rec': rec, 'f1': f1}


evaluation = {}

for category in all_labels:

    target = test_values['label'] == category
    prediction = test_values['pred_label'] == category

    res = eval_category(target, prediction)

    evaluation[category] = res

evaluation_df = pd.DataFrame.from_dict(evaluation).transpose()
evaluation_df.sort_values('f1', ascending=False, inplace=True)

evaluation_df


def model_eval(estimator):
    # Eval the model.
    eval_examples = input_features.create_examples(test_values, 'text', 'label')
    eval_features = classifier.convert_examples_to_features(
        eval_examples, all_labels, MAX_SEQ_LENGTH, tokenizer)
    print('***** Started evaluation at {} *****'.format(
        datetime.datetime.now()))
    print('  Num examples = {}'.format(len(eval_examples)))
    print('  Batch size = {}'.format(EVAL_BATCH_SIZE))

    # Eval will be slightly WRONG on the TPU because it will truncate
    # the last batch.
    eval_steps = int(len(eval_examples) / EVAL_BATCH_SIZE)
    eval_input_fn = classifier.input_fn_builder(features=eval_features,
                                                seq_length=MAX_SEQ_LENGTH,
                                                is_training=False,
                                                drop_remainder=True)
    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
    print('***** Finished evaluation at {} *****'.format(
        datetime.datetime.now()))
    output_eval_file = os.path.join(OUTPUT_DIR, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
        print("***** Eval results *****")
        for key in sorted(result.keys()):
            print('  {} = {}'.format(key, str(result[key])))
            writer.write("%s = %s\n" % (key, str(result[key])))


model_eval(estimator_from_tfhub)
