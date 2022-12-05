import glob
from tensorflow.keras.preprocessing import image_dataset_from_directory

from config import ColorMode



def create_train_test_validation_dataset():
    files_local_location='./images'
    chest_xray_dir = f"{files_local_location}/chest_xray"
    train_directory = f'{chest_xray_dir}/train'
    test_directory = f'{chest_xray_dir}/test'
    validation_directory = f'{chest_xray_dir}/val'

    image_size = (220, 220)
    batch_size = 16
    input_shape = (220, 220, 3) # color_mode = rgb
    color_mode = ColorMode.RGB.name.lower()
    train_dataset = image_dataset_from_directory(directory=train_directory,
                                                    labels='inferred',
                                                    label_mode='binary',
                                                    color_mode=color_mode,
                                                    batch_size=batch_size,
                                                    image_size=image_size,
                                                    shuffle=True
                                                    )
    validation_dataset = image_dataset_from_directory(directory=validation_directory,
                                                    labels='inferred',
                                                    label_mode='binary',
                                                    color_mode=color_mode,
                                                    batch_size=batch_size,
                                                    image_size=image_size,
                                                    shuffle=True
                                                    )
    test_dataset = image_dataset_from_directory(directory=test_directory,
                                                    labels='inferred',
                                                    label_mode='binary',
                                                    color_mode=color_mode,
                                                    batch_size=batch_size,
                                                    image_size=image_size,
                                                    shuffle=False
                                                    )
    return [train_dataset, validation_dataset, test_dataset]