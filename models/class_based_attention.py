from __future__ import absolute_import, division, print_function

import tensorflow as tf

from utils import optimization, modeling


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):

    model = modeling.BertModel(config=bert_config,
                               is_training=is_training,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               token_type_ids=segment_ids,
                               use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable("output_bias", [num_labels],
                                  initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        #one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        #per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        per_example_loss = -tf.reduce_sum(labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(use_tpu):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):
        """The `model_fn` for TPUEstimator."""

        predictions = {}

        tags = set()
        if mode == tf.estimator.ModeKeys.TRAIN:
            tags.add("train")

        input_mask = features["input_mask"]
        batch_size = input_mask.shape[0]
        if labels is not None:
            label_ids = tf.cast(labels["label_ids"], tf.float32)

        if "embeddings" not in features:
            input_ids = features["input_ids"]
            segment_ids = features["segment_ids"]

            model = modeling.BertModel(config=params['bert_config'],
                                       is_training=params['trainable_bert'],
                                       input_ids=input_ids,
                                       input_mask=input_mask,
                                       token_type_ids=segment_ids,
                                       use_one_hot_embeddings=True)

            # In the demo, we are doing a simple classification task on the entire
            # TODO: Check is_training === trainable Bert j?
            # model = create_model(bert_config=params['bert_config'],
            #                     is_training=params['trainable_bert'],
            #                     num_labels=params['num_classes'],
            #                     labels=label_ids,
            #                     segment_ids=segment_ids,
            #                     input_ids=input_ids,
            #                     input_mask=input_mask,
            #                     use_one_hot_embeddings=True)

            # TODO: Find correct place
            tvars = tf.trainable_variables()
            initialized_variable_names = {}

            scaffold_fn = None
            if params["init_checkpoint"]:
                (assignment_map, initialized_variable_names
                 ) = modeling.get_assignment_map_from_checkpoint(
                     tvars, params["init_checkpoint"])
                if use_tpu:
                    def tpu_scaffold():
                        tf.train.init_from_checkpoint(params["init_checkpoint"],
                                                      assignment_map)
                        return tf.train.Scaffold()
                    scaffold_fn = tpu_scaffold
                else:
                    tf.train.init_from_checkpoint(params["init_checkpoint"], assignment_map)

            tf.logging.info("**** Variables - INIT FROM CKPT ****")
            for var in tvars:
                if var.name in initialized_variable_names:
                    tf.logging.info("name: {}, shape: {}".format(var.name, var.shape))

            sequence_output = model.get_sequence_output()
            predictions["sequence_output"] = sequence_output

        else:
            sequence_output = features["embeddings"]

        hidden_size = sequence_output.shape[-1].value
        if params["class_based_attention"]:
            shared_query_embedding = tf.get_variable(
                'shared_query', [1, 1, params["shared_size"]],
                initializer=tf.truncated_normal_initializer(stddev=0.02))
            shared_query_embedding = tf.broadcast_to(
                shared_query_embedding,
                [1, params["num_classes"], params["shared_size"]])
            class_query_embedding = tf.get_variable(
                'class_query',
                [1, params["num_classes"], hidden_size - params["shared_size"]],
                initializer=tf.truncated_normal_initializer(stddev=0.02))
            query_embedding = tf.concat(
                [shared_query_embedding, class_query_embedding], axis=2)
            # Reimplement Attention layer to peek into weights.
            scores = tf.matmul(query_embedding,
                               sequence_output,
                               transpose_b=True)
            input_bias = tf.abs(input_mask - 1)
            scores -= 1.e9 * tf.expand_dims(tf.cast(input_bias, tf.float32),
                                            axis=1)
            distribution = tf.nn.softmax(scores)
            pooled_output = tf.matmul(distribution, sequence_output)
        else:
            first_token_tensor = tf.squeeze(sequence_output[:, 0:1, :], axis=1)
            pooled_output = tf.layers.dense(first_token_tensor,
                                            hidden_size,
                                            activation=tf.tanh)

        if mode == tf.estimator.ModeKeys.TRAIN:
            pooled_output = tf.nn.dropout(pooled_output, rate=params["dropout"])

        logits = tf.layers.dense(pooled_output, params["num_classes"])
        logits = tf.matrix_diag_part(logits)

        # probabilities = tf.nn.softmax(logits, axis=-1)  # single-label case
        probabilities = tf.nn.sigmoid(logits)  # multi-label case

        train_op, loss = None, None
        eval_metrics = None
        if mode != tf.estimator.ModeKeys.PREDICT:
            with tf.variable_scope("loss"):
                per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=label_ids, logits=logits)
                loss = tf.reduce_mean(per_example_loss)
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(loss,
                                                     params["learning_rate"],
                                                     params["num_train_steps"],
                                                     params["num_warmup_steps"],
                                                     use_tpu,
                                                     trainable_bert=params['trainable_bert'])
        elif mode == tf.estimator.ModeKeys.EVAL:

            def _f1_score(labels, pred):
                """Computes F1 score, i.e. the harmonic mean of precision and recall."""
                precision = tf.metrics.precision(labels, pred)
                recall = tf.metrics.recall(labels, pred)
                return (2 * precision[0] * recall[0] /
                        (precision[0] + recall[0] + 1e-5),
                        tf.group(precision[1], recall[1]))

            def metric_fn(per_example_loss, labels, probabilities):
                pred = tf.where(probabilities > 0.4,
                                tf.ones_like(probabilities),
                                tf.zeros_like(probabilities))
                return {
                    'absolute/false_positives':
                        tf.metrics.false_positives(labels, pred),
                    'absolute/false_negatives':
                        tf.metrics.false_negatives(labels, pred),
                    'absolute/true_positives':
                        tf.metrics.true_positives(labels, pred),
                    'absolute/true_negatives':
                        tf.metrics.true_negatives(labels, pred),
                    'absolute/total':
                        tf.metrics.true_positives(tf.ones([batch_size]),
                                                  tf.ones([batch_size])),
                    'metric/acc':
                        tf.metrics.accuracy(labels, pred),
                    'metric/prec':
                        tf.metrics.precision(labels, pred),
                    'metric/recall':
                        tf.metrics.recall(labels, pred),
                    'metric/f1':
                        _f1_score(labels, pred),
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, probabilities])

        predictions["probabilities"] = probabilities
        predictions["attention"] = distribution
        predictions["pooled_output"] = pooled_output

        if use_tpu:
            return tf.contrib.tpu.TPUEstimatorSpec(mode=mode,
                                                   loss=loss,
                                                   train_op=train_op,
                                                   scaffold_fn=scaffold_fn,
                                                   eval_metrics=eval_metrics,
                                                   predictions=predictions)
        else:
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op,
                                              predictions=predictions)

    return model_fn
