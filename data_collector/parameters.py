from abc import ABCMeta, abstractmethod
from enum import Enum

from typing import AnyStr, List


class Period(Enum):
    Year = "year"
    YearMonth = "year_month"


class GCP:
    PROJECT_ID = "electricity-imperial"

    class BigQuery:
        LOCATION = "EU"

        class Omie:
            DATASET_ID = "omie"
            MARGINAL_PRICES_TABLE_ID = "marginal_prices"
            OFFER_CURVES_TABLE_ID = "offer_curves"
            OFFER_CURVES_UNITS_TABLE_ID= "offer_curves_units"


class OmieParameter(metaclass=ABCMeta):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def raw_file_name(self) -> AnyStr:
        pass

    @property
    @abstractmethod
    def skip_rows(self) -> int:
        pass

    @property
    @abstractmethod
    def integer_cols(self) -> List[AnyStr]:
        pass

    @property
    @abstractmethod
    def float_cols(self) -> List[AnyStr]:
        pass

    @property
    @abstractmethod
    def col_names(self) -> List[AnyStr]:
        pass

    @property
    @abstractmethod
    def date_format(self) -> AnyStr:
        pass

    @property
    @abstractmethod
    def zip_period(self) -> Period:
        pass

    @property
    @abstractmethod
    def bq_table(self) -> AnyStr:
        pass


class MarginalPriceParams(OmieParameter):
    raw_file_name = "marginalpdbc"
    skip_rows = 1
    integer_cols = ["year", "month", "day", "hour"]
    float_cols = []
    col_names = integer_cols + ["portugal", "spain"]
    date_format = "%Y-%m-%d"
    zip_period = Period.Year
    bq_table = "marginal_prices"


class OfferCurvesParams(OmieParameter):
    raw_file_name = "curva_pbc"
    skip_rows = 3
    integer_cols = ["hour"]
    float_cols = ["energy", "price"]
    col_names = integer_cols + ["date", "country", "unit", "offer_type"] + float_cols + ["status"]
    date_format = "%d/%m/%Y"
    zip_period = Period.Year
    bq_table = "offer_curves"

    class OfferType(Enum):
        bid = "C"
        ask = "V"

    class OfferStatus(Enum):
        offered = "O"
        cleared = "C"


class OfferCurvesUnitsParams(OfferCurvesParams):
    raw_file_name = "curva_pbc_uof"
    zip_period = Period.YearMonth
    bq_table = "offer_curves_units"
