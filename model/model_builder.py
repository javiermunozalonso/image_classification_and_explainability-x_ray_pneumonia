from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple
from dataset import DatasetFolder, ImagesDataset
from config import ClassificationNames

import tensorflow as tf
import numpy as np

GLOBAL_EPOCHS = 30
BATCH_SIZE = 16
INPUT_SHAPE = (220, 220, 3) # color_mode = rgb

@dataclass
class ModelBuilder:
    train_dataset: List[str]
    test_dataset: List[str]
    validation_dataset: List[str]
    _model: tf.keras.Model
    # model_path: str
    optimizer: tf.keras.optimizers.Optimizer
    loss: tf.keras.losses.Loss
    
    _model_name: str = 'model'
    
    _batch_size: int = BATCH_SIZE
    callbacks: List[tf.keras.callbacks.Callback] = None
    metrics: List[tf.keras.metrics.Metric] = [tf.keras.metrics.Metric.binary_accuracy]
    _epochs: int = GLOBAL_EPOCHS
    # class_weights: Dict[int, float] = None
    class_weights: Dict[int, float] = {
        ClassificationNames.PNEUMONIA.value: 1.0,
        ClassificationNames.NORMAL.value: 0.5
    }

    @property
    def model_path(self) -> str:
        return f'{self.model_name}.h5'

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def set_batch_size(self, new_batch_size: int):
        self._batch_size = new_batch_size

    @property
    def epochs(self) -> int:
        return self._epochs

    @epochs.setter
    def set_epochs(self, epochs: int):
        self.epochs = epochs

    @property
    def model_name(self) -> str:
        return self._model_name

    @model_name.setter
    def set_model_name(self, stamp: str):
        if not stamp:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self._model_name = f'model_{stamp}'

    def add_metrics(self):
        self.metrics = list(set(self.metrics.append(tf.keras.metrics.binary_accuracy)))

    def build(self) -> tf.keras.Model:

        inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)

        # Block One
        x = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='valid')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        # Block Two
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        # Block Three
        x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='valid')(x)
        x = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='valid')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPool2D()(x)
        x = tf.keras.layers.Dropout(0.4)(x)

        # Head
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        #Final Layer (Output)
        output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

        self._model = tf.keras.Model(inputs=[inputs], outputs=output, name=self.model_name)

        self._model.compile(optimizer=self.optimizer,
                            loss=self.loss,
                            metrics=self.metrics)

        self._model.summary()
        return self._model

    def train(self):
        history_fit = self._model.fit(x=self.train_dataset.dataset,
                        epochs=self.epochs,
                        batch_size=self._batch_size,
                        validation_data=self.validation_dataset.dataset,
                        callbacks=self.callbacks,
                        class_weight=self.class_weights)
        self._model.save(self.model_path)
        return history_fit

    def evaluate(self) -> Tuple[float, float]:
        loss, accuracy = self._model.evaluate(x=self.test_dataset.dataset,
                                                batch_size=self._batch_size)
        return loss, accuracy

    def predict_model(self, image_path: str) -> Tuple[int, float]:
        # image = tf.keras.preprocessing.image.load_img(image_path,
        #                                                 target_size=(224, 224))
        # image = tf.keras.preprocessing.image.img_to_array(image)
        # image = np.expand_dims(image, axis=0)
        # image = tf.keras.applications.mobilenet.preprocess_input(image)
        prediction = self._model.predict(image_path)
        return np.argmax(prediction),\
                np.max(prediction),\
                ClassificationNames.to_array[np.argmax(prediction)]

    # def predict_model_from_dataset(self, dataset: ImagesDataset) -> Tuple[int, float]:
    #     image_path = dataset.get_random_image_path()
    #     return self.predict_model(image_path)

    # def predict_model_from_test_dataset(self) -> Tuple[int, float]:
    #     return self.predict_model_from_dataset(self.test_dataset.normal_images_dataset)

    # def predict_model_from_validation_dataset(self) -> Tuple[int, float]:
    #     return self.predict_model_from_dataset(self.validation_dataset.normal_images_dataset)

    # def predict_model_from_train_dataset(self) -> Tuple[int, float]:
    #     return self.predict_model_from_dataset(self.train_dataset.normal_images_dataset)

    # def predict_model_from_pneumonia_test_dataset(self) -> Tuple[int, float]:
    #     return self.predict_model_from_dataset(self.test_dataset.pneumonia_images_dataset)

    # def predict_model_from_pneumonia_validation_dataset(self) -> Tuple[int, float]:
    #     return self.predict_model_from_dataset(self.validation_dataset.pneumonia_images_dataset)