from enum import Enum

class ColorMode(Enum):
    GRAYSCALE = 0
    RGB = 1
    RGBA = 2

    @staticmethod
    def to_array():
        return list(ColorMode)

    @staticmethod
    def to_dictionary():
        return {item.value: item.name for item in list(ColorMode)}