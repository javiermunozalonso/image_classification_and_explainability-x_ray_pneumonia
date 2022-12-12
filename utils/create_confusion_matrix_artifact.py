from config.ClassificationNames import ClassificationNames
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def create_confusion_matrix_artifact(true_positive: int, true_negative: int, false_positive: int, false_negative: int):
    """
    Create a confusion matrix with the given values
    :param true_positive: number of true positives
    :param true_negative: number of true negatives
    :param false_positive: number of false positives
    :param false_negative: number of false negatives
    :return: a confusion matrix artifact
    """

    classification_names = list(map(lambda item: item.name, ClassificationNames.to_array()))

    conf_matrix = np.array([[true_positive, true_negative],
                            [false_positive, false_negative]])

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    ax.set_xticklabels([''] + classification_names, fontsize=18)
    ax.set_yticklabels([''] + classification_names, fontsize=18)
    ax.set_xlabel('Predicted', fontsize=18)
    ax.set_ylabel('Actuals', fontsize=18)
    ax.set_title('Confusion Matrix', fontsize=18)
    fig.add_axes(ax)
    return fig