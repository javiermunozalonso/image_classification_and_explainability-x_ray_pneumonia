from dataclasses import dataclass
from typing import Dict
from config import ClassificationNames
# from tensorflow.keras.optimizers import Adam, SGD, RMSprop
# from tensorflow.keras.losses import BinaryCrossentropy, BinaryFocalCrossentropy, Hinge, SquaredHinge

import tensorflow as tf

from model.model_handler import ModelHandler

from hyperopt import tpe, hp, fmin, STATUS_OK,Trials, space_eval

INPUT_SHAPE = (220, 220, 3) # color_mode = rgb
TRIALS_MAX_EVALS = 20

@dataclass
class ModelHyperparametrizedHandler(ModelHandler):

    def _create_search_space(self) -> Dict:
        batch_size = hp.uniformint('batch_size', 8, 64)
        epochs = hp.uniformint('epochs', 8, 64)
        pneumonia_weight = hp.uniform('pneumonia_weight', 0.3, 1.0)
        normal_weight = hp.uniform('normal_weight', 0.5, 1.0)
        optimizer = hp.choice('optimizer', ['Adam', 'RMSprop', 'SGD'])
        loss = hp.choice('loss', ['binary_crossentropy', 'binary_focal_crossentropy', 'hinge', 'squared_hinge'])

        return {
            'batch_size': batch_size,
            'epochs': epochs,
            'pneumonia_weight': pneumonia_weight,
            'normal_weight': normal_weight,
            'optimizer': optimizer,
            'loss': loss
        }

    def set_params(self, params: Dict):
        self.batch_size = params.get('batch_size')
        self.epochs = params.get('epochs')
        self.class_weights = {
                                ClassificationNames.PNEUMONIA.value: params.get('pneumonia_weight'),
                                ClassificationNames.NORMAL.value: params.get('normal_weight')
        }
        self.optimizer = params.get('optimizer')
        self.loss = params.get('loss')

    def get_best_params_for_model(self) -> Dict:
        space = self._create_search_space()

        trials = Trials()
        fmin_result = fmin(fn=self.train_with_eval,
                            space=space,
                            algo=tpe.suggest,
                            max_evals=TRIALS_MAX_EVALS,
                            trials=trials,
                            verbose=True,
                            show_progressbar=True)

        return space_eval(space, fmin_result)

    def train_with_eval(self, params: Dict):
        self.set_params(params=params)

        self.build()

        history_fit: tf.keras.callbacks.History = self._model.fit(x=self.train_dataset,
                                                                    epochs = self.epochs,
                                                                    batch_size = self.batch_size,
                                                                    validation_data = self.test_dataset,
                                                                    callbacks = self.callbacks,
                                                                    class_weight = self.class_weights)

        evaluate_results = self._model.evaluate(self.validation_dataset)
        evaluate_results = dict(zip(self._model.metrics_names, evaluate_results))
        return {
            'binary_accuracy': evaluate_results.get('binary_accuracy', -1),
            'loss': evaluate_results.get('loss',-1),
            'precision': evaluate_results.get('precision', -1),
            'recall': evaluate_results.get('recall',-1),
            'status': STATUS_OK,
        }