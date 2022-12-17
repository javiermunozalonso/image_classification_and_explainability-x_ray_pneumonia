from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List
from config import ClassificationNames
from tensorflow.keras.optimizers import Adam

import tensorflow as tf

from model.model_handler import ModelHandler

from hyperopt import tpe, hp, fmin, STATUS_OK,Trials, space_eval

INPUT_SHAPE = (220, 220, 3) # color_mode = rgb
TRIALS_MAX_EVALS = 10

@dataclass
class ModelHyperparametrizedHandler(ModelHandler):

    def _create_search_space(self) -> Dict:
        batch_size: int = hp.uniformint('batch_size', 8, 64)
        self.batch_size = batch_size

        epochs: int = hp.choice('epochs', [16, 32, 64])
        self.epochs = epochs

        pneumonia_weight: float = hp.uniform('pneumonia_weight', 0.5, 1.0)
        normal_weight: float = hp.uniform('normal_weight', 0.5, 1.0)
        self.class_weights = {
                                ClassificationNames.PNEUMONIA.value: pneumonia_weight,
                                ClassificationNames.NORMAL.value: normal_weight
                            }

        # learning_rate: float = hp.uniform('learning_rate', 1.e-8, 1.e-5)
        # self.optimizer = Adam(learning_rate = learning_rate)
        # self.optimizer = Adam(learning_rate = 1.e-8)

        return {
            'batch_size': batch_size,
            'epochs': epochs,
            'pneumonia_weight': pneumonia_weight,
            'normal_weight': normal_weight,
            # 'learning_rate': learning_rate
        }

    def set_best_params(self, best_params: Dict):
        self.batch_size = best_params.get('batch_size')
        self.epochs = best_params.get('epochs')
        self.class_weights = {
                                ClassificationNames.PNEUMONIA.value: best_params.get('pneumonia_weight'),
                                ClassificationNames.NORMAL.value: best_params.get('normal_weight')
        }
        # learning_rate = best_params.get('learning_rate')
        # self.optimizer = Adam(learning_rate = learning_rate)



    def get_best_params_for_model(self):
        space = self._create_search_space()

        self.build()

        trials = Trials()
        fmin_result = fmin(self.train_with_eval,
                            space,
                            algo=tpe.suggest,
                            max_evals=TRIALS_MAX_EVALS,
                            trials=trials,
                            verbose=True,
                            show_progressbar=True)

        return space_eval(space, fmin_result)

    def train_with_eval(self, params: Dict):

        epochs = params.get('epochs', self.epochs)
        batch_size = params.get('batch_size', self.batch_size)

        pneumonia_weight: float = params.get('pneumonia_weight',self.class_weights[ClassificationNames.PNEUMONIA.value])
        normal_weight: float = params.get('normal_weight', self.class_weights[ClassificationNames.NORMAL.value])
        class_weights = {
                                ClassificationNames.PNEUMONIA.value: pneumonia_weight,
                                ClassificationNames.NORMAL.value: normal_weight
                            }

        print('-------------------------------')
        print(epochs)
        print(batch_size)
        print(class_weights)
        print('-------------------------------')

        history_fit: tf.keras.callbacks.History = self._model.fit(x=self.train_dataset,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=self.test_dataset,
                        callbacks=self.callbacks,
                        class_weight=class_weights)

        evaluate_results = self._model.evaluate(self.validation_dataset)
        evaluate_results = dict(zip(self._model.metrics_names, evaluate_results))
        return {
            'binary_accuracy': evaluate_results['binary_accuracy'],
            'loss': evaluate_results['loss'],
            'precision': evaluate_results['precision'],
            'recall': evaluate_results['recall'],
            'status': STATUS_OK,
        }