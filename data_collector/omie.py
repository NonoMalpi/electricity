import datetime
import logging
import time

from io import BytesIO
from typing import AnyStr
from zipfile import ZipFile

import requests

import pandas as pd

from joblib import Parallel, delayed

logger = logging.getLogger()
logging.basicConfig(format="%(asctime)s|%(name)s|%(levelname)s|%(message)s", level=logging.INFO)


class Omie:

    url_pattern = "https://www.omie.es/es/file-download?parents%5B0%5D" + \
                  "={family_file}&filename={filename}"

    date_file_pattern = "{filename}_{date_str}"

    integer_cols = ["year", "month", "day", "hour"]

    marginalpdbc_col_names = integer_cols + ["portugal", "spain"]

    @staticmethod
    def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
        for col in Omie.integer_cols:
            df[col] = df[col].astype(int)

        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-" + df["month"].astype(str) + "-" + df["day"].astype(str),
            format="%Y-%m-%d"
        )

        return df

    @staticmethod
    def _create_df_from_bytes(filebytes: bytes) -> pd.DataFrame:

        df = pd.read_csv(
            filepath_or_buffer=filebytes, delimiter=";",
            skiprows=1, names=Omie.marginalpdbc_col_names, index_col=False
        )

        df.dropna(inplace=True)

        df = Omie._clean_df(df)

        return df

    @staticmethod
    def _read_date_bytes_to_df(unzip_file: ZipFile, filename: AnyStr,  date: pd.Timestamp) -> pd.DataFrame:

        date_str = date.strftime("%Y%m%d")

        file_pattern = Omie.date_file_pattern.format(filename=filename, date_str=date_str)

        try:
            file_list = [s for s in unzip_file.namelist() if file_pattern in s]

            if len(file_list) > 1:
                logging.warning(f"There are multiple files for {file_pattern}: {len(file_list)}")

            file = file_list[-1]

            filebytes = unzip_file.open(name=file, mode="r")

            df = Omie._create_df_from_bytes(filebytes=filebytes)

            if df.shape[0] != 24:
                logging.warning(f"Dataframe shape for {file} is not 24, it is {df.shape[0]}")

            return df

        except KeyError as e:
            logging.warning(msg=f"The {file_pattern} was not found in zip")

    @staticmethod
    def _parse_unzip_file(unzip_file: ZipFile, filename: AnyStr, year: int) -> pd.DataFrame:

        dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")

        assert len(dates) <= len(unzip_file.namelist()), \
            f"Zip file for year {year} does not contain all dates. " \
            f"There are {len(unzip_file.namelist())} dates."

        df_list = []
        for date in dates:
            df_list.append(Omie._read_date_bytes_to_df(
                unzip_file=unzip_file, filename=filename, date=date
            ))

        return pd.concat(df_list)

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
    def download_year_file(filename: AnyStr, year: int) -> pd.DataFrame:

        filename_year_zip = f"{filename}_{year}.zip"
        zip_content = Omie._download_content(family_file=filename, filename=filename_year_zip)
        unzip_content = Omie._decompress_zip(zip_content=zip_content)
        year_df = Omie._parse_unzip_file(unzip_file=unzip_content, filename=filename, year=year)

        return year_df

    @staticmethod
    def download_date_file(filename: AnyStr, date: AnyStr) -> pd.DataFrame:

        date = pd.to_datetime(date).strftime("%Y%m%d")
        filename_date = f"{filename}_{date}.1"
        file_content = Omie._download_content(family_file=filename, filename=filename_date)
        date_df = Omie._create_df_from_bytes(filebytes=BytesIO(file_content))
        return date_df

    @staticmethod
    def download_period_file(filename: AnyStr, start_year: int, end_year: int):

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

