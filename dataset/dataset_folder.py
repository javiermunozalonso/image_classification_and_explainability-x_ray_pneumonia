from dataclasses import dataclass
from config.ClassificationNames import ClassificationNames

from dataset.images_dataset import ImagesDataset


@dataclass(init=True)
class DatasetFolder:
    __normal_images_dataset: ImagesDataset
    __pneumonia_images_dataset: ImagesDataset

    def __init__(self, normal_location_folder: str, pneumonia_location_folder: str):
        self.__normal_images_dataset = ImagesDataset(normal_location_folder)
        self.__pneumonia_images_dataset = ImagesDataset(pneumonia_location_folder)

    @property
    def normal_images_dataset(self) -> ImagesDataset:
        return self.__normal_images_dataset

    @property
    def pneumonia_images_dataset(self) -> ImagesDataset:
        return self.__pneumonia_images_dataset

    @property
    def classification_name(self) -> str:
        return self.__classification_name
