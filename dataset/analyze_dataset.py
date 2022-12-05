from dataclasses import dataclass
from typing import List
import matplotlib.pyplot as plt

from dataset.dataset_folder import DatasetFolder
from dataset.images_dataset import ImagesDataset

@dataclass
class AnalyzeDataset:
    train_dataset: DatasetFolder
    test_dataset: DatasetFolder
    validation_dataset: DatasetFolder

    def analyze_all_datasets(self):
        fig, (first_ax, middle_ax, last_ax) = plt.subplots(1, 3)
        fig.set_size_inches(20,15)
        fig.add_axes(self._analyze_train_dataset(first_ax))
        fig.add_axes(self._analyze_test_dataset(middle_ax))
        fig.add_axes(self._analyze_validation_dataset(last_ax))
        return fig

    def _analyze_train_dataset(self, figure: plt.Axes) -> plt.Axes:
        values = [self.train_dataset.normal_images_dataset.dataset_length,
                    self.train_dataset.pneumonia_images_dataset.dataset_length]
        legend = self.__get_legend(self.train_dataset)
        title = "Type of images in train folder"
        return self.__analyze_dataset(figure=figure, values=values, legend=legend, title=title)

    def _analyze_test_dataset(self, figure: plt.Axes) -> plt.Axes:
        values = [self.test_dataset.normal_images_dataset.dataset_length,
                    self.test_dataset.pneumonia_images_dataset.dataset_length]
        legend = self.__get_legend(self.test_dataset)
        title = "Type of images in test folder"
        return self.__analyze_dataset(figure=figure, values=values, legend=legend, title=title)

    def _analyze_validation_dataset(self, figure: plt.Axes) -> plt.Axes:
        values = [self.validation_dataset.normal_images_dataset.dataset_length,
                    self.validation_dataset.pneumonia_images_dataset.dataset_length]
        legend = self.__get_legend(self.validation_dataset)
        title = "Type of images in validation folder"
        return self.__analyze_dataset(figure=figure, values=values, legend=legend, title=title)

    def __get_legend(self, dataset: DatasetFolder) -> List:
        return [f"normal ({dataset.normal_images_dataset.dataset_length})",
                f"pneumonia ({dataset.pneumonia_images_dataset.dataset_length})"]

    def __analyze_dataset(self, figure: plt.Axes, values: List, legend: List, title: str) -> plt.Axes:
        figure.pie(x=values,
                    autopct="%.1f%%",
                    explode=[0.2,0],
                    pctdistance=0.5)
        figure.set_title(title, pad=20)
        figure.legend(legend)
        return figure






