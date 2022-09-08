from typing import Tuple

import pandas as pd
import numpy as np

from electricity.data_collector import OfferCurvesParams

df = pd.DataFrame()


class OfferCurves:
    """ Class that generate historical ask/bid curves and cleared price for an hour from an input dataframe.

    Parameters
    ---------
    df: pd.DataFrame
        The input dataframe containing all historical records of offer curves for one hour from Omie.

    Attributes
    ----------
    hour: int
        The market hour, it corresponds to one hour of the day.

    min_date: pd.Timestamp
        The minimum date of the historical series of curves.

    max_date: pd.Timestamp
        The maximum date of the historical series of curves.

    ask_df: pd.DataFrame
        The dataframe containing all historical ask (sell) offers (energy and price) by date and units.

    bid_df: pd.DataFrame
        The dataframe containing all historical bid (buy) offers (energy and price) by date and units.

    cleared_price_df: pd.DataFrame
        The dataframe containing cleared price and energy by date.
        It is calculated as the intersection between ask and bid curves for each day.
    """
    def __init__(self, df: pd.DataFrame):
        self.hour = df["hour"].unique()[0]
        self.min_date, self.max_date = df["date"].min(), df["date"].max()
        self.ask_df, self.bid_df = self._process_ask_bid_curves(df=df)
        self.cleared_price_df = self._generated_cleared_price_df()

    def _process_ask_bid_curves(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ask_df = df[df["offer_type"] == OfferCurvesParams.OfferType.ask.value]
        bid_df = df[df["offer_type"] == OfferCurvesParams.OfferType.bid.value]

        ask_df = ask_df.sort_values(["date", "price"])
        bid_df = bid_df.sort_values("date").sort_values("price", ascending=False)

        ask_df["agg_energy"] = ask_df.groupby("date")["energy"].transform("cumsum")
        bid_df["agg_energy"] = bid_df.groupby("date")["energy"].transform("cumsum")

        ask_df.reset_index(drop=True, inplace=True)
        bid_df.reset_index(drop=True, inplace=True)

        return ask_df, bid_df

    def _obtain_cleared_price_for_date(self, ask_df: pd.DataFrame, bid_df: pd.DataFrame, date: pd.Timestamp) -> pd.Series:
        ask_df = ask_df.loc[date].reset_index()
        bid_df = bid_df.loc[date].reset_index()

        ask_agg_energy = ask_df["agg_energy"].values
        bid_agg_energy = bid_df["agg_energy"].values
        agg_energy = np.unique(np.sort(np.concatenate([ask_agg_energy, bid_agg_energy])))

        ask_df = ask_df.set_index("agg_energy").reindex(agg_energy)
        bid_df = bid_df.set_index("agg_energy").reindex(agg_energy)

        cols = ["offer_type", "price"]
        ask_df = ask_df[cols]
        bid_df = bid_df[cols]
        df_price = ask_df.merge(bid_df, left_index=True, right_index=True, suffixes=("_ask", "_bid"))

        df_price["last_ask"] = df_price["price_ask"].fillna(method="ffill")
        df_price["last_bid"] = df_price["price_bid"].fillna(method="ffill")

        df_price["curve_intersection"] = df_price["last_bid"] <= df_price["last_ask"]
        df_price.reset_index(inplace=True)

        cleared_price = df_price[df_price["curve_intersection"]].iloc[0]
        cleared_price = cleared_price.rename({"last_ask": "price"})[["agg_energy", "price"]]
        cleared_price.name = date
        return cleared_price

    def _generated_cleared_price_df(self) -> pd.DataFrame:
        dates = pd.date_range(start=self.min_date, end=self.max_date, freq="D")

        prices_series = []
        ask_df = self.ask_df.set_index("date")
        bid_df = self.bid_df.set_index("date")
        for i, date in enumerate(dates):
            prices_series.append(self._obtain_cleared_price_for_date(ask_df=ask_df, bid_df=bid_df, date=date))

        return pd.concat(prices_series, axis=1).T
