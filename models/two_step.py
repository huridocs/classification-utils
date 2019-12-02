def model_fn_builder(num_classes, learning_rate, num_train_steps, use_tpu):
  """Returns `model_fn` closure for TPUEstimator."""

  # Compute number of train and warmup steps
  num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

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
      bert_module = hub.Module(BERT_MODEL_HUB, tags=tags, trainable=False)
      bert_inputs = dict(input_ids=input_ids, input_mask=input_mask,
                        segment_ids=segment_ids)
      bert_outputs = bert_module(inputs=bert_inputs, signature="tokens",
                                as_dict=True)
      sequence_output = bert_outputs['sequence_output']
      predictions["sequence_output"] = sequence_output
    else:
      sequence_output = features["embeddings"]

    hidden_size = sequence_output.shape[-1].value
    shared_query_embedding = tf.get_variable(
        'shared_query', [1, 1, SHARED_SIZE],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    shared_query_embedding = tf.broadcast_to(
        shared_query_embedding, [1, num_classes, SHARED_SIZE])
    class_query_embedding = tf.get_variable(
        'class_query', [1, num_classes, hidden_size-SHARED_SIZE],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    query_embedding = tf.concat([shared_query_embedding,
                                 class_query_embedding], axis=2)
    # Reimplement Attention layer to peek into weights.
    scores = tf.matmul(query_embedding, sequence_output, transpose_b=True)
    input_bias = tf.abs(input_mask - 1)
    scores -= 1.e9 * tf.expand_dims(tf.cast(input_bias, tf.float32), axis=1)
    distribution = tf.nn.softmax(scores)
    pooled_output = tf.matmul(distribution, sequence_output)
   
    if mode == tf.estimator.ModeKeys.TRAIN:
      pooled_output = tf.nn.dropout(pooled_output, rate=DROPOUT)

    logits = tf.layers.dense(pooled_output, num_classes)
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
      train_op = optimization.create_optimizer(
        loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(mode=mode, loss=loss,
                                                    train_op=train_op)
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
            'absolute/false_positives': tf.metrics.false_positives(labels, pred),
            'absolute/false_negatives': tf.metrics.false_negatives(labels, pred),
            'absolute/true_positives': tf.metrics.true_positives(labels, pred),
            'absolute/true_negatives': tf.metrics.true_negatives(labels, pred),
            'absolute/total': tf.metrics.true_positives(
              tf.ones([batch_size]), tf.ones([batch_size])),
            'metric/acc': tf.metrics.accuracy(labels, pred),
            'metric/prec': tf.metrics.precision(labels, pred),
            'metric/recall': tf.metrics.recall(labels, pred),
            'metric/f1': _f1_score(labels, pred),
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, probabilities])
      
    predictions["probabilities"] = probabilities
    predictions["attention"] = distribution
    predictions["pooled_output"] = pooled_output
      
    output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metrics=eval_metrics,
        predictions=predictions)
    return output_spec

  return model_fn
