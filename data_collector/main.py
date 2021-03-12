from io import BytesIO
from typing import AnyStr
from zipfile import ZipFile

import requests

import pandas as pd


class Downloader:

    url_pattern = "https://www.omie.es/es/file-download?parents%5B0%5D" + \
                  "={filename}&filename={filename}_{date}.zip"

    date_file_pattern = "{filename}_{date_str}.1"

    @staticmethod
    def _read_date_bytes_to_df(unzip_file: ZipFile, filename: AnyStr,  date: pd.Timestamp) -> pd.DataFrame:

        date_str = date.strftime("%Y%m%d")

        file = Downloader.date_file_pattern.format(filename=filename, date_str=date_str)

        filebytes = unzip_file.open(name=file, mode="r")

        col_names = ["year", "month", "day", "hour", "portugal", "spain"]

        df = pd.read_csv(
            filepath_or_buffer=filebytes, delimiter=";",
            skiprows=1, names=col_names, index_col=False
        )

        df.dropna(inplace=True)

        assert df.shape[0] == 24

        return df

    @staticmethod
    def _parse_unzip_file(unzip_file: ZipFile, filename: AnyStr, year: AnyStr):

        dates = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="D")

        assert len(dates) == len(unzip_file.namelist()), \
            f"Zip file for year {year} does not contain all dates"

        return Downloader._read_date_bytes_to_df(
            unzip_file=unzip_file, filename=filename, date=dates[0]
        )

    @staticmethod
    def _decompress_zip(zip_content: requests.Response.content) -> ZipFile:

        filebytes = BytesIO(zip_content)
        unzip_file = ZipFile(filebytes)

        return unzip_file

    @staticmethod
    def _download_year_zip(filename: AnyStr, year: AnyStr) -> requests.Response.content:

        url = Downloader.url_pattern.format(filename=filename, date=year)
        response = requests.get(url)

        return response.content

    @staticmethod
    def download_year_file(filename: AnyStr, year: AnyStr) -> pd.DataFrame:

        zip_content = Downloader._download_year_zip(filename=filename, year=year)
        unzip_content = Downloader._decompress_zip(zip_content=zip_content)

        return Downloader._parse_unzip_file(unzip_file=unzip_content, filename=filename, year="2017")


