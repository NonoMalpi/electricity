from enum import Enum

from matplotlib.colors import LinearSegmentedColormap

from numpy import linspace


class ImperialColors(Enum):
    blue = "#003E74"
    light_blue = "#D4EFFC"
    seaglass = "#379f9f"
    cool_grey = "#9D9D9D"
    light_grey = "#EBEEEE"
    dark_grey = "#373A36"
    brick = "#A51900"
    orange = "#D24000"
    dark_green = "#02893B"
    red = "#DD2501"


class MatplotlibDefaultColors(Enum):
    blue = "tab:blue"
    orange = "tab:orange"


# source: https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72
def hex_to_rgb(value):
    value = value.strip("#")
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    return [v/256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp
