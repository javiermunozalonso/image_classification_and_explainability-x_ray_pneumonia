import logging

from experiments import run_hyperparametryzed_experiment

import time

import mlflow

logging.basicConfig(encoding='utf-8', level=logging.INFO)

def run():
    logging.info('Init main')
    st = time.time()
    mlflow.set_experiment(experiment_name="pneumonia_classification")
    mlflow.set_tracking_uri(uri='./mlruns')
    with mlflow.start_run():
        # run_fixed_experiment()
        run_hyperparametryzed_experiment()

    et = time.time()
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')

    logging.info('End main')

if __name__ == '__main__':
    run()