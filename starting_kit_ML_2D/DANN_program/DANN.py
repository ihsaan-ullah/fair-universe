import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, Dense, Activation, Dropout
from tensorflow.keras.models import Model
import keras
import functools

random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

# ================================
# Label classifier
# ================================

class label_classifier(Layer) :
    def __init__(self,num_classes,name):
        super(label_classifier, self).__init__()
        self.num_classes = num_classes
        self.c_name = name
        
        self.hidden_ftr_1 = Dense(20, activation='relu', name="feature_ext_1")
        self.hidden_lbl_1 = Dense(20, activation='relu', name="label_clf_1")
        self.hidden_lbl_2 = Dense(20, activation='relu', name="label_clf_2")
        self.class_preds = Dense(self.num_classes, activation='softmax', name=self.c_name)

    def call(self,inputs):
        self.out_hidden_ftr = self.hidden_ftr_1(inputs)
        x = self.hidden_lbl_1(self.out_hidden_ftr)
        x = self.hidden_lbl_2(x)
        return self.class_preds(x)
    
# ================================
# Gradient reversal
# ================================

@tf.custom_gradient
def reverse_gradient(x, hp_lambda):
    """
    Flips the sign of the incoming gradient during backpropagation.
    :param x:         Input tensor
    :param hp_lambda: Hyper-parameter lambda (c.f. DANN paper), i.e. an updatable 
                      coefficient applied to the reversed gradient
    :return:          Input tensor with reverse gradient (+ function to compute this reversed gradient)
    """
    
    # Feed-forward operation:
    y = tf.identity(x)
    
    # Back-propagation/gradient-computing operation:
    def _flip_gradient(dy):
        # Since the decorated function `reverse_gradient()` actually has 2 inputs 
        # (counting `hp_lambda), we have to return the gradient for each -- but
        # anyway, the derivative w.r.t `hp_lambda` is null:
        return tf.math.negative(dy) * hp_lambda, tf.constant(0.)
    
    return y, _flip_gradient

class ReversalLayer(tf.keras.layers.Layer):
    '''Flip the sign of gradient during training.'''

    def __init__(self, hp_lambda, name="gradient_rev",**kwargs):
        super().__init__(name=name,**kwargs)
        self.hp_lambda = hp_lambda

    def call(self, inputs, training=None):
        return reverse_gradient(inputs, self.hp_lambda)

    def get_config(self):
        config = super().get_config()
        config['hp_lambda'] = self.hp_lambda
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# ================================
# Domain classifier
# ================================

class domain_classifier(Layer) :
    def __init__(self,num_domains,name):
        super(domain_classifier, self).__init__()
        self.num_domains = num_domains
        self.d_name = name
        
        self.hidden_dmn_1 = Dense(20, activation='relu', name="domain_clf_1")
        #self.activation = Activation("elu", name="domain_clf_3")
        #self.dropout = Dropout(0.5,name="domain_clf_4")
        self.domain_preds = Dense(num_domains, activation='softmax', name=self.d_name)

    def call(self,hp_lambda,out_hidden_ftr):
        x = ReversalLayer(hp_lambda)(out_hidden_ftr)
        x = self.hidden_dmn_1(x)
        #x = self.activation(x)
        #x = self.dropout(x)
        return self.domain_preds(x)

# ================================
# Balanced accuracy
# ================================

class balanced_accuracy(keras.metrics.SparseCategoricalAccuracy):
    def __init__(self, name='balanced_acc', dtype=None):
        super().__init__(name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_flat = y_true
        if y_true.shape.ndims == y_pred.shape.ndims:
            y_flat = tf.squeeze(y_flat, axis=[-1])
        y_true_int = tf.cast(y_flat, tf.int32)

        cls_counts = tf.math.bincount(y_true_int)
        cls_counts = tf.math.reciprocal_no_nan(tf.cast(cls_counts, self.dtype))
        weight = tf.gather(cls_counts, y_true_int)
        return super().update_state(y_true, y_pred, sample_weight=weight)

# ================================
# Reshape data
# ================================

def extract_data (data) :
        x, y = data["data"], data["labels"]
        return  x,y

def set_data_for_DANN (train_sets,test_sets) :
    x_sources, y_sources = [], []
    x_targets, y_targets = [], []
    for train_set, test_set in zip(train_sets,test_sets) :
        x_source, y_source = extract_data (train_set)
        x_target, y_target = extract_data (test_set)
        x_sources.append(x_source)
        y_sources.append(y_source)
        x_targets.append(x_target)
        y_targets.append(y_target)
    return x_sources, y_sources, x_targets, y_targets

# ================================
# Build datasets
# ================================

def build_training_dataset_for_DANN (x_source, y_source, x_target, setting, num_epochs, half_batch_size, portion_of_target_for_train=1) :
    nb_of_events = setting["total_number_of_events"]
    
    nb_target_for_train = int(nb_of_events*portion_of_target_for_train)
    x_target_for_train = x_target[0:nb_target_for_train]

    # Source training dataset composed of points and their labels:
    source_dataset = tf.data.Dataset.from_tensor_slices((x_source, y_source))
    source_dataset = source_dataset.shuffle(buffer_size=nb_of_events).repeat(count=num_epochs)
    source_dataset = source_dataset.batch(half_batch_size)

    # Target training dataset composed of points only:
    target_dataset = tf.data.Dataset.from_tensor_slices(x_target_for_train)
    target_dataset = target_dataset.shuffle(buffer_size=nb_of_events).repeat(count=-1)
    # ^ we repeat the target data indefinitely, counting the epochs w.r.t the source data.
    target_dataset = target_dataset.batch(half_batch_size)

    # Combined training dataset:
    training_dataset = tf.data.Dataset.zip((source_dataset, target_dataset))
    return training_dataset

def _prepare_data_for_dann_training(source_data, target_images,
                                    main_head_name='class_preds', domain_head_name='domain_preds'):
    
    source_images, source_labels = source_data
    
    # We check the number of samples for each domains (to cover edge cases, i.e., 
    # for the possibly smaller batch at the end of each epoch):
    num_source = tf.shape(source_images)[0]
    num_target = tf.shape(target_images)[0]
    
    # Full image batch:
    batch_images = tf.concat((source_images, target_images), axis=0)
    
    # Semantic segmentation label maps:
    # By default, the loss will be computed over the full batch, but we do not have GT
    # label maps for the target images. A simplistic solution is as follows:
    # 1) We create dummy GT label maps for the target images:
    target_labels = tf.zeros([num_target, *source_labels.shape[1:]], dtype=source_labels.dtype)
    batch_labels  = tf.concat((source_labels, target_labels), axis=0)
    # 2) We tell TF/Keras **not** to penalize the model for its predictions on the target
    # images/dummy labels, by assigning a weight = 0 to these elements of the batch:
    source_weight_per_sample = tf.tile([1], [num_source])
    target_weight_per_sample = tf.tile([0], [num_target])
    batch_sample_weights = tf.concat((source_weight_per_sample, target_weight_per_sample), 
                                     axis=0)
    
    # Note: this solution is simple but not optimal. Layer and loss operations will be applied
    # to half the batch even though the results won't be used for backpropagation. A better
    # solution to avoid useless computations would be to implement a custom loss aware it
    # should ignore the 2nd half of the batch. Or even better: to edit the model so that
    # only the source images are feed-forwarded to the classification head (e.g., adding a custom
    # layers dropping ~half the batch).
    
    # Domain classification ground-truth labels:
    # if we assign the label "1" to source data and "0" to target data, then we can simply reuse
    # the `batch_sample_weights` tensor:
    domain_labels = batch_sample_weights
    domain_sample_weights = tf.tile([1], [num_source + num_target])
    
    batch_targets = {main_head_name: batch_labels, 
                     domain_head_name: domain_labels}
    batch_sample_weights = {main_head_name: batch_sample_weights, 
                            domain_head_name: domain_sample_weights}

    return batch_images, batch_targets, batch_sample_weights

def _prepare_data_for_dann_validation(target_images, target_labels,
                                      main_head_name='class_preds', domain_head_name='domain_preds'):
    # The batch contains only validation/test images from the target domain. 
    # This time, we want to evaluate the main loss over these images, so we assign a normal loss
    # weight = 1 to each samples.
    num_samples = tf.shape(target_images)[0]
    
    # We want to evaluate over 
    loss_weights = tf.tile([1], [num_samples])
    
    domain_labels = tf.tile([0], [num_samples])
    
    batch_targets = {main_head_name: target_labels, 
                     domain_head_name: domain_labels}
    batch_sample_weights = {main_head_name: loss_weights, 
                            domain_head_name: loss_weights}

    return target_images, batch_targets, batch_sample_weights

def build_datasets_for_DANN (x_sources,y_sources,x_targets,y_targets,settings,num_epochs,batch_size,portion_of_target_for_train=1) :
    # Build training datasets
    training_datasets = []
    half_batch_size = batch_size//2
    prepare_for_dann_fn = functools.partial(_prepare_data_for_dann_training,
                                        main_head_name="class_preds", 
                                        domain_head_name="domain_preds")
    for x_source,y_source,x_target,setting in zip (x_sources,y_sources,x_targets,settings) :
        training_dataset = build_training_dataset_for_DANN(x_source, y_source, x_target, setting, num_epochs, half_batch_size, portion_of_target_for_train)
        training_dataset = training_dataset.map(prepare_for_dann_fn, num_parallel_calls=4)
        training_datasets.append(training_dataset)

    # Build testing dataset
    testing_datasets = []
    prepare_for_dann_fn = functools.partial(_prepare_data_for_dann_validation,
                                        main_head_name="class_preds", 
                                        domain_head_name="domain_preds")
    for x_target, y_target in zip(x_targets, y_targets) :
        testing_dataset = tf.data.Dataset.from_tensor_slices((x_target, y_target))
        testing_dataset = testing_dataset.batch(batch_size)
        testing_dataset = testing_dataset.map(prepare_for_dann_fn, num_parallel_calls=4)
        testing_datasets.append(testing_dataset)
    
    # Cast datasets features to float32 instead of float64
    def cast_to_float32(features, labels, weights):
        return tf.cast(features, tf.float32), labels, weights
    training_datasets = [training_dataset.map(cast_to_float32) for training_dataset in training_datasets]
    testing_datasets = [testing_dataset.map(cast_to_float32) for testing_dataset in testing_datasets]

    return training_datasets, testing_datasets

# ================================
# Instantiate DANN
# ================================

def build_DANN (name,input_size,hp_lambda) :
    # Set input
    inputs = Input(shape=input_size)
    
    # Set label classifier
    lbl_clssfr = label_classifier(2,"class_preds")
    class_preds = lbl_clssfr.call(inputs)
    
    # Set domain classifier
    dmn_clssfr = domain_classifier(2,"domain_preds")
    domain_preds = dmn_clssfr.call(tf.Variable(hp_lambda),lbl_clssfr.out_hidden_ftr)
    
    # Set DANN
    DANN_model = Model(inputs = inputs,
                       outputs = [class_preds,domain_preds],
                       name = name)
    return DANN_model

# ================================
# DANN Callbacks
# ================================

class CallbacksHistory(tf.keras.callbacks.Callback):
    def __init__(self):
        super(CallbacksHistory, self).__init__()
        self.source_evaluations = {"class_preds_balanced_acc" : [],
                                   "domain_preds_acc" : []}
        self.target_evaluations = {"val_class_preds_balanced_acc" : [],
                                   "val_domain_preds_acc" : []}

    def on_epoch_end(self, epoch, logs=None):
        self.source_evaluations["class_preds_balanced_acc"].append(logs['class_preds_balanced_acc'])
        self.source_evaluations["domain_preds_acc"].append(logs['domain_preds_acc'])
        self.target_evaluations["val_class_preds_balanced_acc"].append(logs['val_class_preds_balanced_acc'])
        self.target_evaluations["val_domain_preds_acc"].append(logs['val_domain_preds_acc'])

# ================================
# DANN functions
# ================================

def compile_DANN (DANN_model) :
    # Compile DANN

    DANN_model.compile(optimizer = "adam",
                       loss = {"class_preds":  'sparse_categorical_crossentropy',
                               "domain_preds": 'sparse_categorical_crossentropy'},
                       loss_weights = {"class_preds": 1,
                                       "domain_preds": 1},
                       weighted_metrics = {"class_preds" : [tf.metrics.SparseCategoricalAccuracy(name='acc'),balanced_accuracy()],
                                           "domain_preds" : [tf.metrics.SparseCategoricalAccuracy(name='acc')]}
        )

def fit_DANN (DANN_model,training_dataset,testing_dataset,num_epochs,train_steps_per_epoch,test_steps_per_epoch) :
    #class_weight={0:1,1:10}
    history = CallbacksHistory()

    DANN_model.fit(training_dataset,
                   #sample_weight=class_weight,
                   epochs=num_epochs,
                   steps_per_epoch=train_steps_per_epoch,
                   validation_data=testing_dataset,
                   validation_steps=test_steps_per_epoch,
                   verbose=0,
                   callbacks=[history])
    return history

def evaluate_DANN (DANN_model,x_source, y_source, x_target, y_target,res_df=None) :
    nb_source_events = x_source.shape[0]
    nb_target_events = x_target.shape[0]
    res_source = DANN_model.evaluate(x_source,{"class_preds" : y_source, "domain_preds" : np.ones((nb_source_events,))})
    res_target = DANN_model.evaluate(x_target,{"class_preds" : y_target, "domain_preds" : np.zeros((nb_target_events,))})

    output_res = {"source_acc" : format(res_source[3], ".3f"),
                  "source_bal_acc" : format(res_source[4], ".3f"),
                  "target_acc" : format(res_target[3], ".3f"),
                 "target_bal_acc" : format(res_target[4], ".3f")}
    if res_df :
        return res_df.append(output_res,ignore_index=True)
    else :
        return output_res