from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List
from hyperopt import STATUS_OK, fmin, Trials, tpe

import tensorflow as tf

INPUT_SHAPE = (220, 220, 3) # color_mode = rgb

@dataclass
class ModelHandler:
    train_dataset: List[str]
    test_dataset: List[str]
    validation_dataset: List[str]
    optimizer: tf.keras.optimizers.Optimizer = None
    loss: tf.keras.losses.Loss = None
    _model: tf.keras.Model = None
    _model_name: str = None
    _batch_size: int = field(init=True, default=None, repr=False)
    _epochs: int = field(init=True, default=None, repr=False)
    _class_weights: Dict[int, float] = None

    @property
    def callbacks(self) -> List[tf.keras.callbacks.Callback]:
        return [tf.keras.callbacks.TensorBoard('model_logs')]

    @property
    def metrics(self) -> List[tf.keras.metrics.Metric]:
        return [
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.BinaryCrossentropy(),
                tf.keras.metrics.BinaryIoU(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.AUC(name='AUC_ROC',curve='ROC'),
                tf.keras.metrics.AUC(name='AUC_PR',curve='PR'),
                tf.keras.metrics.FalseNegatives(),
                tf.keras.metrics.FalsePositives(),
                tf.keras.metrics.TrueNegatives(),
                tf.keras.metrics.TruePositives()
                ]

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, new_batch_size: int) -> None:
        self._batch_size = new_batch_size

    @property
    def class_weights(self) -> Dict[int, float]:
        return self._class_weights

    @class_weights.setter
    def class_weights(self, new_class_weights: Dict[int, float]) -> None:
        self._class_weights = new_class_weights

    @property
    def model_path(self) -> str:
        return f'{self.model_name}.h5'

    @property
    def epochs(self) -> int:
        return self._epochs

    @epochs.setter
    def epochs(self, new_epochs: int) -> None:
        self._epochs = new_epochs

    @property
    def model_name(self) -> str:
        if not self._model_name:
            self._model_name = f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        return self._model_name

    def build(self) -> tf.keras.Model:

        inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)

        x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid')(x)
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        self._model = tf.keras.Model(inputs=[inputs], outputs=output, name=self.model_name)

        self._model.compile(optimizer=self.optimizer,
                            loss=self.loss,
                            metrics=self.metrics)

        self._model.summary()
        return self._model

    def train(self):
        history_fit: tf.keras.callbacks.History = self._model.fit(x=self.train_dataset,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        validation_data=self.test_dataset,
                        callbacks=self.callbacks,
                        class_weight=self.class_weights)

        return history_fit

    def save_model(self):
        self._model.save(self.model_path)

    def evaluate(self):
        evaluate_results = self._model.evaluate(x=self.validation_dataset,
                                                batch_size=self.batch_size)
        evaluation_metrics_names = list(map(lambda name: ''.join(['evaluation_', name]), self._model.metrics_names))
        return dict(zip(evaluation_metrics_names, evaluate_results))
