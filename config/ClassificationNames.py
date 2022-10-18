from enum import Enum

class ClassificationNames(Enum):
    NORMAL = 0
    PNEUMONIA = 1

    @staticmethod
    def to_array():
        return list(ClassificationNames)

    @staticmethod
    def to_dictionary():
        return {item.value: item.name for item in list(ClassificationNames)}
