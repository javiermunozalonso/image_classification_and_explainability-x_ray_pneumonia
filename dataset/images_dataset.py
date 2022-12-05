from dataclasses import dataclass
from typing import List
import glob


@dataclass(init=True)
class ImagesDataset:
    __location_folder: str
    __glob_images_dataset: List[str]
    __dataset_length: int

    def __init__(self, location_folder: str):
        self.__location_folder = location_folder
        self.__glob_images_dataset = glob.glob(f"{location_folder}/*")
        self.__dataset_length = len(self.__glob_images_dataset)

    @property
    def images_dataset(self) -> List[str]:
        return self.__glob_images_dataset

    @property
    def dataset_length(self) -> int:
        return self.__dataset_length