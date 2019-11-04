def model_fn_builder(params):
  """Returns `model_fn` closure for TPUEstimator."""
  
  num_classes = params['num_classes']
  learning_rate = params['learning_rate']
  num_train_steps = params['num_train_steps']
  use_tpu = params['use_tpu']
  warmup_proportion = params['warmup_proportion']
  shared_size = params['shared_size']
  dropout = params['dropout']

  # Compute number of train and warmup steps
  num_warmup_steps = int(num_train_steps * warmup_proportion)

  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""

    input_mask = features["input_mask"]
    batch_size = tf.shape(input_mask)[0]
    sequence_output = features["embeddings"]
    if labels is not None:
      label_ids = tf.cast(labels["label_ids"], tf.float32)

    tags = set()
    if mode == tf.estimator.ModeKeys.TRAIN:
      tags.add("train")

    hidden_size = sequence_output.shape[-1].value
    shared_query_embedding = tf.get_variable(
        'shared_query', [1, 1, shared_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    shared_query_embedding = tf.broadcast_to(
        shared_query_embedding, [1, num_classes, shared_size])
    class_query_embedding = tf.get_variable(
        'class_query', [1, num_classes, hidden_size-shared_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))
    query_embedding = tf.concat([shared_query_embedding,
                                 class_query_embedding], axis=2)
    # pooled_output has shape [bs, num_classes, hidden_size]
    pooled_output = tf.keras.layers.Attention()(
        [query_embedding, sequence_output],
        [tf.ones([batch_size, num_classes], tf.bool),
          tf.cast(input_mask, tf.bool)])
   
    if mode == tf.estimator.ModeKeys.TRAIN:
      pooled_output = tf.nn.dropout(pooled_output, rate=dropout)

    logits = tf.layers.dense(pooled_output, num_classes)
    logits = tf.matrix_diag_part(logits)

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
      def _f1_score(labels, predictions):
        """Computes F1 score, i.e. the harmonic mean of precision and recall."""
        precision = tf.metrics.precision(labels, predictions)
        recall = tf.metrics.recall(labels, predictions)
        return (2 * precision[0] * recall[0] /
                (precision[0] + recall[0] + 1e-5),
                tf.group(precision[1], recall[1]))
  
      def metric_fn(per_example_loss, labels, probabilities):
        predictions = tf.where(probabilities > 0.4,
                               tf.ones_like(probabilities),
                               tf.zeros_like(probabilities))
        return {
            'absolute/false_positives': tf.metrics.false_positives(labels, predictions),
            'absolute/false_negatives': tf.metrics.false_negatives(labels, predictions),
            'absolute/true_positives': tf.metrics.true_positives(labels, predictions),
            'absolute/true_negatives': tf.metrics.true_negatives(labels, predictions),
            'absolute/total': tf.metrics.true_positives(
              tf.ones([batch_size]), tf.ones([batch_size])),
            'metric/acc': tf.metrics.accuracy(labels, predictions),
            'metric/prec': tf.metrics.precision(labels, predictions),
            'metric/recall': tf.metrics.recall(labels, predictions),
            'metric/f1': _f1_score(labels, predictions),
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, probabilities])
      
    output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metrics=eval_metrics,
        predictions={"probabilities": probabilities,
                     "pooled_output": pooled_output})
    return output_spec

  return model_fn
