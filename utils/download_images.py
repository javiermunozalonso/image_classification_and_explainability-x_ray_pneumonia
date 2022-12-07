from zipfile import ZipFile
from pathlib import Path
import shutil
import logging

from kaggle.api.kaggle_api_extended import KaggleApi

IMAGES_DATASET_FOLDER_LOCATION='./images'
KAGGLE_DATASET_NAME = 'paultimothymooney/chest-xray-pneumonia'
# DOWNLOADED_FILE_NAME='chest-xray-pneumonia.zip'
DOWNLOADED_FILE_NAME='archive.zip'
DOWNLOADED_FILE_LOCATION=f'{IMAGES_DATASET_FOLDER_LOCATION}/{DOWNLOADED_FILE_NAME}'
FILE_NAME=''

def __remove_data_folder():
    if Path(IMAGES_DATASET_FOLDER_LOCATION).exists():
        shutil.rmtree(IMAGES_DATASET_FOLDER_LOCATION)
    logging.info('The path is clear to start')
    return

def __create_data_folder():
    Path(IMAGES_DATASET_FOLDER_LOCATION).mkdir(exist_ok=True)
    logging.info('Created the folder for the data')
    return

def __download_images_dataset():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_file(dataset=KAGGLE_DATASET_NAME, file_name=FILE_NAME,
                                path=IMAGES_DATASET_FOLDER_LOCATION)
    logging.info('Downloaded the data from kaggle')
    return

def __decompress_data_file():
    with ZipFile(DOWNLOADED_FILE_LOCATION,'r') as zip:
        zip.extractall(path=IMAGES_DATASET_FOLDER_LOCATION)
    logging.info('The file has been decompressed')
    return

def __remove_downloaded_file():
    if Path(DOWNLOADED_FILE_LOCATION).exists():
        Path(DOWNLOADED_FILE_LOCATION).unlink()
    logging.info('The file downloaded has been deleted')
    return

def __remove_macosx_resources():
    MACOSX_FOLDER = Path(f'{IMAGES_DATASET_FOLDER_LOCATION}/chest_xray/__MACOSX')
    if Path(MACOSX_FOLDER).exists():
        shutil.rmtree(MACOSX_FOLDER)
    logging.info('The macosx files have been deleted')
    return

def __remove_repeated_images_folder_resources():
    REPEATED_IMAGES_FOLDERS = Path(f'{IMAGES_DATASET_FOLDER_LOCATION}/chest_xray/chest_xray')
    if Path(REPEATED_IMAGES_FOLDERS).exists():
        shutil.rmtree(REPEATED_IMAGES_FOLDERS)
    logging.info('The repeated images folder have been deleted')
    return

def run():
    __remove_data_folder()
    __create_data_folder()
    __download_images_dataset()
    __decompress_data_file()
    __remove_downloaded_file()
    __remove_macosx_resources()
    __remove_repeated_images_folder_resources()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run()