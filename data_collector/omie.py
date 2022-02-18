import datetime
import logging
import time

from io import BytesIO
from typing import AnyStr, NoReturn
from zipfile import ZipFile

import requests

import pandas as pd

from google.cloud import bigquery
from joblib import Parallel, delayed

from data_collector.parameters import OmieParameter, MarginalPriceParams, OfferCurvesParams, OfferCurvesUnitsParams, Period

logger = logging.getLogger()
logging.basicConfig(format="%(asctime)s|%(name)s|%(levelname)s|%(message)s", level=logging.INFO)


class OmiePeriod:
    def __init__(self, period: Period, year: int, month: int = None):
        self.period = period
        self.year = year
        self.month = month
        self.date_range = self._generate_date_range()
        self.date_str = self._generate_date_str()

    def _generate_date_range(self):

        if self.period == Period.Year:
            return pd.date_range(start=f"{self.year}", end=f"{self.year+1}", freq="D", inclusive="left")
        elif self.period == Period.YearMonth:
            month_limits = pd.date_range(start=f"{self.year}-{self.month}", freq="MS", periods=2)
            return pd.date_range(start=month_limits[0], end=month_limits[1], freq="D", inclusive="left")

    def _generate_date_str(self):

        if self.period == Period.Year:
            return f"{self.year}"
        elif self.period == Period.YearMonth:
            return f"{self.year}{self.month:02d}"


class Omie:
    url_pattern = "https://www.omie.es/es/file-download?parents%5B0%5D={family_file}&filename={filename}"

    date_file_pattern = "{filename}_{date_str}"

    gcloud_client = bigquery.Client(project="electricity-imperial")

    bq_dataset = "omie"

    @staticmethod
    def _clean_df(df: pd.DataFrame, omie_parameter: OmieParameter) -> pd.DataFrame:

        # remove rows without hour
        df = df[~df["hour"].isnull()]

        for col in omie_parameter.integer_cols:
            df[col] = df[col].astype(int)

        for col in omie_parameter.float_cols:
            df[col] = df[col].str.replace(".", "").str.replace(",", ".").astype(float)

        if omie_parameter == MarginalPriceParams:
            df["date"] = pd.to_datetime(
                df["year"].astype(str) + "-" + df["month"].astype(str) + "-" + df["day"].astype(str),
                format="%Y-%m-%d"
            )

        elif omie_parameter in [OfferCurvesParams, OfferCurvesUnitsParams]:
            df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")

        return df

    @staticmethod
    def _create_df_from_bytes(filebytes: bytes, omie_parameter: OmieParameter) -> pd.DataFrame:

        df = pd.read_csv(
            filepath_or_buffer=filebytes,
            delimiter=";",
            skiprows=omie_parameter.skip_rows,
            names=omie_parameter.col_names,
            index_col=False,
            encoding="latin-1"
        )

        df = Omie._clean_df(df=df, omie_parameter=omie_parameter)

        return df

    @staticmethod
    def _read_date_bytes_to_df(unzip_file: ZipFile, omie_parameter: OmieParameter,  date: pd.Timestamp) -> pd.DataFrame:

        date_str = date.strftime("%Y%m%d")

        file_pattern = Omie.date_file_pattern.format(filename=omie_parameter.raw_file_name, date_str=date_str)

        try:
            file_list = [s for s in unzip_file.namelist() if file_pattern in s]

            if len(file_list) > 1:
                logging.warning(f"There are multiple files for {file_pattern}: {len(file_list)}")

            file = file_list[-1]

            filebytes = unzip_file.open(name=file, mode="r")

            df = Omie._create_df_from_bytes(filebytes=filebytes, omie_parameter=omie_parameter)

            if (omie_parameter == MarginalPriceParams) and (df.shape[0] != 24):
                logging.warning(f"Dataframe shape for {file} is not 24, it is {df.shape[0]}")

            return df

        except KeyError as e:
            logging.warning(msg=f"The {file_pattern} was not found in zip")

    @staticmethod
    def _check_zip_dates(unzip_file: ZipFile, period: OmiePeriod) -> NoReturn:

        assert len(period.date_range) <= len(unzip_file.namelist()), \
            f"Zip file for period {period.date_range} does not contain all dates. " \
            f"There are {len(unzip_file.namelist())} dates."

    @staticmethod
    def _parse_unzip_file(unzip_file: ZipFile, omie_parameter: OmieParameter, period: OmiePeriod) -> pd.DataFrame:

        Omie._check_zip_dates(unzip_file=unzip_file, period=period)

        df_list = []
        for i, date in enumerate(period.date_range):
            if i % 30 == 0:
                logging.info(f"Processing date {date} ...")
            df_list.append(
                Omie._read_date_bytes_to_df(unzip_file=unzip_file, omie_parameter=omie_parameter, date=date)
            )

        return pd.concat(df_list)

    @staticmethod
    def _upload_unzip_file(unzip_file: ZipFile,
                           omie_parameter: OmieParameter,
                           period: OmiePeriod,
                           job_config: bigquery.job.LoadJobConfig) -> NoReturn:

        Omie._check_zip_dates(unzip_file=unzip_file, period=period)

        df_list = []
        for i, date in enumerate(period.date_range):
            df_list.append(
                Omie._read_date_bytes_to_df(unzip_file=unzip_file, omie_parameter=omie_parameter, date=date)
            )
            if (i % 100 == 0 and i > 0) or i == len(period.date_range) - 1:
                logging.info(f"Uploading batch until date {date.strftime('%Y-%m-%d')} to BigQuery ...")
                df = pd.concat(df_list)
                job = Omie.gcloud_client.load_table_from_dataframe(
                    dataframe=df,
                    destination=f"{Omie.bq_dataset}.{omie_parameter.raw_file_name}",
                    location="EU",
                    job_config=job_config
                )
                if job.errors:
                    logging.error(job.errors)
                df_list = []

    @staticmethod
    def _decompress_zip(zip_content: requests.Response.content) -> ZipFile:

        filebytes = BytesIO(zip_content)
        unzip_file = ZipFile(filebytes)

        return unzip_file

    @staticmethod
    def _request_content(family_file: AnyStr, filename: AnyStr):

        url = Omie.url_pattern.format(family_file=family_file, filename=filename)
        response = requests.get(url)

        return response.content

    @staticmethod
    def _request_and_decompress_zip(omie_parameter: OmieParameter, date_str: AnyStr) -> ZipFile:

        filename_year_zip = Omie.date_file_pattern.format(filename=omie_parameter.raw_file_name, date_str=date_str) + ".zip"
        zip_content = Omie._request_content(family_file=omie_parameter.raw_file_name, filename=filename_year_zip)
        unzip_content = Omie._decompress_zip(zip_content=zip_content)

        return unzip_content

    # region download
    @staticmethod
    def download_year_file(omie_parameter: OmieParameter, year: int) -> pd.DataFrame:
        """ Download a batch of compressed files for a year from Omie website and process them as a dataframe.

        Param
        ------
        omie_parameter: OmieParamter
            The configuration parameters of the type of file to obtain.

        year: int
            The year to download.

        Return
        ------
        df: pd.DataFrame
            The dataframe with the processed and cleaned file.
        """
        #TODO: Possible refactor to include this logic in OmiePeriod. _parse_unzip_file should be also modified
        if omie_parameter.zip_period == Period.Year:
            period = OmiePeriod(period=omie_parameter.zip_period, year=year)
            unzip_content = Omie._request_and_decompress_zip(omie_parameter=omie_parameter, date_str=period.date_str)
            year_df = Omie._parse_unzip_file(unzip_file=unzip_content, omie_parameter=omie_parameter, period=period)

        elif omie_parameter.zip_period == Period.YearMonth:
            year_df = []
            for i in range(1, 13):
                period = OmiePeriod(period=omie_parameter.zip_period, year=year, month=i)
                unzip_content = Omie._request_and_decompress_zip(omie_parameter=omie_parameter, date_str=period.date_str)
                year_df += Omie._parse_unzip_file(unzip_file=unzip_content, omie_parameter=omie_parameter, period=period)

        return year_df

    @staticmethod
    def download_date_file(omie_parameter: OmieParameter, date: AnyStr) -> pd.DataFrame:
        """ Download a plain file for a given date from Omie website and process it as a dataframe.

        Params
        ------
        omie_parameter: OmieParamter
            The configuration parameters of the type of file to obtain.

        date: str
            The date to download, it has must contain year, month and date in this order.

        Return
        ------
        df: pd.DataFrame
            The dataframe with the processed and cleaned file.
        """

        date = pd.to_datetime(date).strftime("%Y%m%d")
        filename_date = Omie.date_file_pattern.format(filename=omie_parameter.raw_file_name, date_str=date) + ".1"
        file_content = Omie._request_content(family_file=omie_parameter.raw_file_name, filename=filename_date)
        date_df = Omie._create_df_from_bytes(filebytes=BytesIO(file_content), omie_parameter=omie_parameter)

        return date_df

    @staticmethod
    def download_period_file(omie_parameter: OmieParameter, start_year: int, end_year: int) -> pd.DataFrame:
        """ Download a plain file for a given period from Omie website and process it as a dataframe.

        Params
        ------
        omie_parameter: OmieParamter
            The configuration parameters of the type of file to obtain.

        start_year: int
            The starting year of the period.

        end_year: int
            The ending year of the period.

        Return
        ------
        df: pd.DataFrame
            The dataframe with the processed and cleaned file.
        """

        assert start_year >= 2016, "Minimum year stored in Omie is 2016."

        current_year_download = False

        if end_year < datetime.date.today().year:
            end_year += 1
        elif end_year >= datetime.date.today().year:
            current_year_download = True
            end_year = datetime.date.today().year

        years = range(start_year, end_year)

        #TODO: consider timer decorator
        start = time.time()

        df_list = Parallel(n_jobs=-1)(
            delayed(Omie.download_year_file)(omie_parameter=omie_parameter, year=year)
            for year in years
        )

        if current_year_download:
            dates = pd.date_range(start=f"{end_year}-01-01", end=datetime.date.today(), freq="D")
            df_list += Parallel(n_jobs=-1)(
                delayed(Omie.download_date_file)(omie_parameter=omie_parameter, date=date)
                for date in dates
            )

        df = pd.concat(df_list)

        end = time.time()
        logging.info(f"Time processing: {end-start}")

        return df
    # endregion

    # region uploadBQ
    @staticmethod
    def upload_bq_year_file(omie_parameter: OmieParameter, year: int, job_config: bigquery.job.LoadJobConfig) -> NoReturn:
        """ Upload to BigQuery a plain file for a given year from Omie website.

        Params
        ------
        omie_parameter: OmieParamter
            The configuration parameters of the type of file to obtain.

        year: int
            The year to download.

        job_config: bigquery.job.LoadJobConfig
            The BigQuery job configuration
        """
        if omie_parameter.zip_period == Period.Year:
            period = OmiePeriod(period=omie_parameter.zip_period, year=year)
            unzip_content = Omie._request_and_decompress_zip(omie_parameter=omie_parameter, date_str=period.date_str)
            Omie._upload_unzip_file(
                unzip_file=unzip_content, omie_parameter=omie_parameter, period=period, job_config=job_config
            )

        elif omie_parameter.zip_period == Period.YearMonth:
            for i in range(1, 13):
                period = OmiePeriod(period=omie_parameter.zip_period, year=year, month=i)
                unzip_content = Omie._request_and_decompress_zip(omie_parameter=omie_parameter, date_str=period.date_str)
                Omie._upload_unzip_file(
                    unzip_file=unzip_content, omie_parameter=omie_parameter, period=period, job_config=job_config
                )

    @staticmethod
    def upload_bq_date_file(omie_parameter: OmieParameter, date: AnyStr, job_config: bigquery.job.LoadJobConfig) -> NoReturn:
        """ Upload to BigQuery a plain file for a given date from Omie website.

        Params
        ------
        omie_parameter: OmieParamter
            The configuration parameters of the type of file to obtain.

        date: str
            The date to download, it has must contain year, month and date in this order.

        job_config: bigquery.job.LoadJobConfig
            The BigQuery job configuration
        """
        df = Omie.download_date_file(omie_parameter=omie_parameter, date=date)
        job = Omie.gcloud_client.load_table_from_dataframe(
            dataframe=df,
            destination=f"{Omie.bq_dataset}.{omie_parameter.raw_file_name}",
            location="EU",
            job_config=job_config
        )
        if job.errors:
            logging.error(job.errors)

    @staticmethod
    def upload_bq_period_file(omie_parameter: OmieParameter,
                              start_year: int,
                              end_year: int,
                              job_config: bigquery.job.LoadJobConfig) -> NoReturn:
        """ Upload to BigQuery a plain file for a given period from Omie website.

        Params
        ------
        omie_parameter: OmieParamter
            The configuration parameters of the type of file to obtain.

        start_year: int
            The starting year of the period.

        end_year: int
            The ending year of the period.

        job_config: bigquery.job.LoadJobConfig
            The BigQuery job configuration

        Return
        ------
        df: pd.DataFrame
            The dataframe with the processed and cleaned file.
        """

        assert start_year >= 2016, "Minimum year stored in Omie is 2016."

        current_year_download = False

        if end_year < datetime.date.today().year:
            end_year += 1
        elif end_year >= datetime.date.today().year:
            current_year_download = True
            end_year = datetime.date.today().year

        years = range(start_year, end_year)

        # TODO: consider timer decorator
        start = time.time()

        logging.info("Uploading previous years ...")
        _ = Parallel(n_jobs=-1)(
            delayed(Omie.upload_bq_year_file)(omie_parameter=omie_parameter, year=year, job_config=job_config)
            for year in years
        )

        if current_year_download:
            logging.info("Uploading current year ...")
            dates = pd.date_range(start=f"{end_year}-01-01", end=datetime.date.today(), freq="D")
            _ = Parallel(n_jobs=-1)(
                delayed(Omie.upload_bq_date_file)(omie_parameter=omie_parameter, date=date, job_config=job_config)
                for date in dates
            )

        end = time.time()
        logging.info(f"Time processing: {end - start}")

    # endregion
    @staticmethod
    def include_old_file(df: pd.DataFrame, filename: str, omie_parameter: OmieParameter) -> pd.DataFrame:

        old_df = pd.read_csv(filename)
        old_df = Omie._clean_df(old_df, omie_parameter=omie_parameter)

        return pd.concat([old_df, df], ignore_index=True)

