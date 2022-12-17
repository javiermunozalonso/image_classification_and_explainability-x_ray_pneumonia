import logging
from config import ClassificationNames

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import pprint

from dataset import create_train_test_validation_dataset
from model.model_hyperparametrized_handler import ModelHyperparametrizedHandler

import time

import mlflow

from utils.create_confusion_matrix_artifact import create_confusion_matrix_artifact

logging.basicConfig(encoding='utf-8', level=logging.INFO)

def run_experiment():
    logging.info('Init run')
    [train_dataset, validation_dataset, test_dataset] = create_train_test_validation_dataset()

    learning_rate: float = 1.e-8
    batch_size: int = 8
    epochs: int = 10
    pneumonia_weight: float = .5
    normal_weight: float = 1

    model_handler = ModelHyperparametrizedHandler(train_dataset = train_dataset,
                                    test_dataset = test_dataset,
                                    validation_dataset = validation_dataset,
                                    loss = BinaryCrossentropy(),
                                    optimizer = Adam(learning_rate = learning_rate))


    model_handler.batch_size = batch_size
    model_handler.epochs = epochs
    model_handler.class_weights = {
                                    ClassificationNames.PNEUMONIA.value: pneumonia_weight,
                                    ClassificationNames.NORMAL.value: normal_weight
                                }

    best_params = model_handler.get_best_params_for_model()

    model_handler.set_best_params(best_params)

    train_results = model_handler.train()

    logging.info(train_results)
    pprint.pprint(train_results)
    pprint.pprint(train_results.history)
    model_handler.save_model()

    evaluate_results = model_handler.evaluate()

    pprint.pprint(evaluate_results)
    print(evaluate_results)

    mlflow.log_metrics(evaluate_results)
    mlflow.log_dict(evaluate_results, artifact_file='evaluation_results.json')

    test_confusion_matrix_figure = create_confusion_matrix_artifact(true_positive=evaluate_results['evaluation_true_positives'],
                                                        true_negative=evaluate_results['evaluation_true_negatives'],
                                                        false_positive=evaluate_results['evaluation_false_positives'],
                                                        false_negative=evaluate_results['evaluation_false_negatives'])

    mlflow.log_figure(test_confusion_matrix_figure, artifact_file='test_confusion_matrix.png')
    logging.info('end run')

    return