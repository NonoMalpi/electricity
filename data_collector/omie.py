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

from data_collector.parameters import OmieParameter, MarginalPriceParams, OfferCurvesParams

logger = logging.getLogger()
logging.basicConfig(format="%(asctime)s|%(name)s|%(levelname)s|%(message)s", level=logging.INFO)


class Omie:
    url_pattern = "https://www.omie.es/es/file-download?parents%5B0%5D={family_file}&filename={filename}"

    date_file_pattern = "{filename}_{date_str}"

    gcloud_client = bigquery.Client(project="electricity-imperial")

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

        elif omie_parameter == OfferCurvesParams:
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
    def _parse_unzip_file(unzip_file: ZipFile, omie_parameter: OmieParameter, year: int) -> pd.DataFrame:

        dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")

        assert len(dates) <= len(unzip_file.namelist()), \
            f"Zip file for year {year} does not contain all dates. " \
            f"There are {len(unzip_file.namelist())} dates."

        df_list = []
        for i, date in enumerate(dates):
            if i % 10 == 0:
                logging.info(f"Processing date {date} ...")
            df_list.append(
                Omie._read_date_bytes_to_df(unzip_file=unzip_file, omie_parameter=omie_parameter, date=date)
            )

        return pd.concat(df_list)

    @staticmethod
    def _upload_unzip_file(unzip_file: ZipFile,
                           omie_parameter: OmieParameter,
                           year: int,
                           job_config: bigquery.job.LoadJobConfig) -> NoReturn:
        #TODO: refactor with _parse_unzip_file
        dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")

        assert len(dates) <= len(unzip_file.namelist()), \
            f"Zip file for year {year} does not contain all dates. " \
            f"There are {len(unzip_file.namelist())} dates."

        df_list = []
        for i, date in enumerate(dates):
            df_list.append(
                Omie._read_date_bytes_to_df(unzip_file=unzip_file, omie_parameter=omie_parameter, date=date)
            )
            if (i % 30 == 0 and i > 0) or i == len(dates) - 1:
                logging.info(f"Uploading batch {i} to BigQuery ...")
                df = pd.concat(df_list)
                Omie.gcloud_client.load_table_from_dataframe(
                    dataframe=df,
                    destination=f"omie.{omie_parameter.raw_file_name}",
                    location="EU",
                    job_config=job_config
                )
                df_list = []

    @staticmethod
    def _decompress_zip(zip_content: requests.Response.content) -> ZipFile:

        filebytes = BytesIO(zip_content)
        unzip_file = ZipFile(filebytes)

        return unzip_file

    @staticmethod
    def _download_content(family_file: AnyStr, filename: AnyStr):

        url = Omie.url_pattern.format(family_file=family_file, filename=filename)
        response = requests.get(url)

        return response.content

    @staticmethod
    def download_year_file(omie_parameter: OmieParameter, year: int) -> pd.DataFrame:
        """ Download a batch of compressed files for a period from Omie website and process them as a dataframe.

        Param
        ------
        omie_parameter: OmieParamter
            The configuration parameters of the type of file to obtain.

        period: str
            The period to download.

        Return
        ------
        df: pd.DataFrame
            The dataframe with the processed and cleaned file.

        """

        filename_year_zip = Omie.date_file_pattern.format(filename=omie_parameter.raw_file_name, date_str=year) + ".zip"
        zip_content = Omie._download_content(family_file=omie_parameter.raw_file_name, filename=filename_year_zip)
        unzip_content = Omie._decompress_zip(zip_content=zip_content)
        year_df = Omie._parse_unzip_file(unzip_file=unzip_content, omie_parameter=omie_parameter, year=year)

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
        file_content = Omie._download_content(family_file=omie_parameter.raw_file_name, filename=filename_date)
        date_df = Omie._create_df_from_bytes(filebytes=BytesIO(file_content), omie_parameter=omie_parameter)

        return date_df

    @staticmethod
    def download_period_file(filename: AnyStr, start_year: int, end_year: int) -> pd.DataFrame:

        assert start_year >= 2016, "Minimum year stored in Omie is 2016."

        current_year_download = False

        if end_year < datetime.date.today().year:
            end_year += 1
        elif end_year >= datetime.date.today().year:
            current_year_download = True
            end_year = datetime.date.today().year

        years = range(start_year, end_year)

        start = time.time()

        df_list = Parallel(n_jobs=-1)(
            delayed(Omie.download_year_file)(filename=filename, year=year) for year in years
        )

        if current_year_download:
            dates = pd.date_range(start=f"{end_year}-01-01", end=datetime.date.today(), freq="D")
            df_list += Parallel(n_jobs=-1)(
                delayed(Omie.download_date_file)(filename=filename, date=date) for date in dates
            )

        df = pd.concat(df_list)

        end = time.time()
        logging.info(f"Time processing: {end-start}")

        return df

    @staticmethod
    def upload_date_file_gcp(omie_parameter: OmieParameter, date: AnyStr, job_config: bigquery.job.LoadJobConfig) -> NoReturn:
        df = Omie.download_date_file(omie_parameter=omie_parameter, date=date)
        Omie.gcloud_client.load_table_from_dataframe(
            dataframe=df,
            destination=f"omie.{omie_parameter.raw_file_name}",
            location="EU",
            job_config=job_config
        )

    @staticmethod
    def upload_year_file_gcp(omie_parameter: OmieParameter, year: int, job_config: bigquery.job.LoadJobConfig) -> NoReturn:
        #TODO: refactor this with download_year_file method
        filename_year_zip = Omie.date_file_pattern.format(filename=omie_parameter.raw_file_name, date_str=year) + ".zip"
        zip_content = Omie._download_content(family_file=omie_parameter.raw_file_name, filename=filename_year_zip)
        unzip_content = Omie._decompress_zip(zip_content=zip_content)
        Omie._upload_unzip_file(
            unzip_file=unzip_content, omie_parameter=omie_parameter, year=year, job_config=job_config
        )

    @staticmethod
    def include_old_file(df: pd.DataFrame, filename: str) -> pd.DataFrame:

        old_df = pd.read_csv(filename)
        old_df = Omie._clean_df(old_df)

        return pd.concat([old_df, df], ignore_index=True)

