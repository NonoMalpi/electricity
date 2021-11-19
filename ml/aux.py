import os
import pickle
import re

from typing import Dict, List


def prepare_kernel_dict(dim_num: int) -> Dict:
    if dim_num == 1:
        return {}
    elif dim_num == 2:
        return {h: {} for h in range(2, 25)}
    else:
        raise ValueError


def convert_element_to_hour(element: str, dim_num: int) -> List[int]:
    if "_" in element and dim_num == 2:
        hour_list = element.split("_")
        hour_i, hour_j = int(hour_list[0]), int(hour_list[1])
        return [hour_i, hour_j]
    else:
        hour = int(element)
        return [hour]


def load_kernels_dict(path: str, pattern: str, dim_num: int = 1) -> Dict:
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
