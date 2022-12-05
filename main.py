from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

import logging

from dataset import create_train_test_validation_dataset
from model.model_builder import ModelBuilder

LEARNING_RATE=0.0001

def run():
    logging.info('Init run')
    [train_dataset, validation_dataset, test_dataset] = create_train_test_validation_dataset
    
    model_builder = ModelBuilder(train_dataset=train_dataset,
                                    test_dataset=test_dataset,
                                    validation_dataset=validation_dataset,
                                    loss=BinaryCrossentropy(),
                                    optimizer=Adam(learning_rate=LEARNING_RATE),
    )
    
    model_builder.build()
    
    history = model_builder.train()
    
    logging.info(history)
    
    evaluate_results = model_builder.evaluate()
    
    logging.info(evaluate_results)
    
    logging.info('end run')
    return

if __name__ == '__main__':
    logging.info('Init main')
    run()
    logging.info('End main')