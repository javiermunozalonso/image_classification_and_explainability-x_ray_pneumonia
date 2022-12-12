from config import ClassificationNames
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import pprint

from dataset import create_train_test_validation_dataset
from model.model_handler import ModelHandler

import time

import mlflow

import logging

from utils.create_confusion_matrix_artifact import create_confusion_matrix_artifact

logging.basicConfig(encoding='utf-8', level=logging.INFO)

BATCH_SIZE: int = 8
EPOCHS: int = 32
LEARNING_RATE: float = 1.e-6

def run():
    logging.info('Init run')
    [train_dataset, validation_dataset, test_dataset] = create_train_test_validation_dataset()

    model_handler = ModelHandler(train_dataset=train_dataset,
                                    test_dataset=test_dataset,
                                    validation_dataset=validation_dataset,
                                    loss=BinaryCrossentropy(),
                                    optimizer=Adam(learning_rate=LEARNING_RATE))

    model_handler.batch_size = BATCH_SIZE
    model_handler.epochs = EPOCHS
    model_handler.class_weights = {
                                    ClassificationNames.PNEUMONIA.value: 1.0,
                                    ClassificationNames.NORMAL.value: 0.5
                                }

    mlflow.autolog()
    mlflow.tensorflow.autolog()

    model_handler.build()

    train_results = model_handler.train()

    logging.info(train_results)
    pprint.pprint(train_results)
    pprint.pprint(train_results.history)
    # log_metric('evaluate_results', history)

    evaluate_results = model_handler.evaluate()
    # log_metric('evaluate_results', evaluate_results)

    pprint.pprint(evaluate_results)
    print(evaluate_results)
    # logging.info(evaluate_results)

    confusion_matrix_figure = create_confusion_matrix_artifact(true_positive=evaluate_results['true_positives'],
                                                        true_negative=evaluate_results['true_negatives'],
                                                        false_positive=evaluate_results['false_positives'],
                                                        false_negative=evaluate_results['false_negatives'])

    mlflow.log_figure(confusion_matrix_figure, artifact_file='evaluation_confusion_matrix.png')

    test_confusion_matrix_figure = create_confusion_matrix_artifact(true_positive=evaluate_results['true_positives'],
                                                        true_negative=evaluate_results['true_negatives'],
                                                        false_positive=evaluate_results['false_positives'],
                                                        false_negative=evaluate_results['false_negatives'])

    mlflow.log_figure(test_confusion_matrix_figure, artifact_file='test_confusion_matrix.png')
    logging.info('end run')

    return

if __name__ == '__main__':
    logging.info('Init main')
    st = time.time()
    mlflow.set_experiment(experiment_name="pneumonia_classification")
    mlflow.set_tracking_uri(uri='./mlruns')
    with mlflow.start_run():
        run()

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    logging.info('End main')