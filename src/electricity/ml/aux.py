import os
import pickle
import re

from typing import Dict, List, Union

from electricity.ml import GaussianKernel


def prepare_kernel_dict(dim_num: int) -> Dict:
    """ Return a dictionary or a dictionary of dictionaries depending on the dim_num input.

    Parameters
    ----------
    dim_num: int
        The number of dimensions of the independent variables, default = 1. Possible values: 1, 2.

    Returns
    -------
    Dic
        A simple dictionary or a dictionary of dictionaries with keys from 2 to 24.

    Raises
    ------
    ValueError
        If the dim_num is different from 1 or 2.
    """
    if dim_num == 1:
        return {}
    elif dim_num == 2:
        return {h: {} for h in range(2, 25)}
    else:
        raise ValueError


def convert_element_to_hour(element: str, dim_num: int) -> List[int]:
    """ Extract hours from the element pattern.

    Parameters
    ----------
    element: str
        The pattern used to save hours, it could be {hour} or {hour}_{hour} depending on the dimension.

    dim_num: int
        The number of dimensions of the independent variables, default = 1. Possible values: 1, 2.

    Returns
    -------
    List[int]
        List with the extracted hours.
    """
    if "_" in element and dim_num == 2:
        hour_list = element.split("_")
        hour_i, hour_j = int(hour_list[0]), int(hour_list[1])
        return [hour_i, hour_j]
    else:
        hour = int(element)
        return [hour]


def load_kernels_dict(path: str, pattern: str, dim_num: int = 1) -> Dict[int, Union[GaussianKernel, Dict[int, GaussianKernel]]]:
    """ Load fitted kernels from pickles and save them into a dictionary.

    Parameters
    ----------
    path: str
        The local directory where the pickles are saved.

    pattern: str
        The pattern used to save the fitted kernels, it must be "{coefficient}_(.*)_{grid_shape}".
        The elements inside (.*) must contain the hours of the fitted kernel.

    dim_num: int
        The number of dimensions of the independent variables, default = 1. Possible values: 1, 2.

    Returns
    -------
    kernel_dict: Dict[int, Union[GaussianKernel, Dict[int, GaussianKernel]]]
        Dictionary contain the fitted kernels.
        If dim_num = 1, the key is the hour of the independent variable and the value is the fitted kernel.
        If dim_num = 2, the key is the hour of the first independent variable, the value is a dictionary
        containing with the key being the hour of the second independent variable and the value the fitted kernel.

    Raises
    ------
    ValueError
        If the dim_num is different from 1 or 2.
    """
    kernel_dict = prepare_kernel_dict(dim_num=dim_num)
    for filename in os.listdir(path):
        element = re.search(pattern, filename).group(1)
        hours = convert_element_to_hour(element=element, dim_num=dim_num)
        if dim_num == 1:
            kernel_dict[hours[0]] = pickle.load(open(path + filename, "rb"))
        elif dim_num == 2:
            kernel_dict[hours[0]][hours[1]] = pickle.load(open(path + filename, "rb"))
        else:
            raise ValueError

    return kernel_dict
