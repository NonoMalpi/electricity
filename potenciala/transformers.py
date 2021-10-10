import numpy as np
import pandas as pd


class Transformer:

    @staticmethod
    def rename_signal(signal_name: str):
        pass

    @staticmethod
    def transform(series: pd.Series):
        pass


class NoneTransformer(Transformer):

    @staticmethod
    def rename_signal(signal_name: str):
        return signal_name

    @staticmethod
    def transform(series: pd.Series):
        return series


class LogTransformer(Transformer):

    @staticmethod
    def rename_signal(signal_name: str):
        return "log_" + signal_name

    @staticmethod
    def transform(series: pd.Series):
        return np.log(1 + series)


class TransformerFactory:

    Constructor = {
        "Log": LogTransformer,
        "None": NoneTransformer,
    }

    @staticmethod
    def build(transformer_type: str = None) -> Transformer:
        if not transformer_type:
            transformer_type = "None"

        if transformer_type not in TransformerFactory.Constructor.keys():
            raise NotImplementedError("{} transformation is not yet implemented".format(transformer_type))

        return TransformerFactory.Constructor[transformer_type]
