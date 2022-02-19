import logging

from typing import AnyStr, NoReturn, Union

from google.cloud import bigquery
from google.cloud.exceptions import NotFound

from data_collector.parameters import GCP, OfferCurvesParams, OfferCurvesUnitsParams
from data_collector.queries import HOURLY_TABLE

logger = logging.getLogger()
logging.basicConfig(format="%(asctime)s|%(name)s|%(levelname)s|%(message)s", level=logging.INFO)


class BQManager:

    def __init__(self, client: bigquery.Client):
        self.client = client
        self.omie_dataset = GCP.BigQuery.Omie.DATASET_ID

    def _check_table(self, table_id: AnyStr) -> NoReturn:
        try:
            self.client.get_table(table=table_id)
        except NotFound:
            logging.error(f"[!!] Table {table_id} not found ...")

    def generate_hourly_offer_curve_tables(self,
                                           curve_parameter: Union[OfferCurvesParams, OfferCurvesUnitsParams],
                                           offer_status: Union[None, OfferCurvesParams.OfferStatus] = None) -> NoReturn:
        """ Split offer curves table from BigQuery into hourly tables.

        Params
        ------
        curve_parameter: Union[OfferCurvesParams, OfferCurvesUnitsParams]
            The configuration parameters of the Omie curve type.

        offer_status: Union[None, OfferCurvesParams.OfferStatus]
            Filter offer curves status by this parameter, default = None.
        """

        table_id = f"{GCP.PROJECT_ID}.{self.omie_dataset}.{curve_parameter.bq_table}"
        self._check_table(table_id=table_id)

        #TODO: make it parallel
        for i in range(1, 25):
            offer_status_constraint = "" if not offer_status else f"AND status = '{offer_status.value}'"
            query = HOURLY_TABLE.format(table=table_id, hour_index=i, additional_constraint=offer_status_constraint)

            job_config = bigquery.job.QueryJobConfig()
            job_config.create_disposition = bigquery.job.CreateDisposition.CREATE_IF_NEEDED
            job_config.write_disposition = bigquery.job.WriteDisposition.WRITE_TRUNCATE
            job_config.destination = table_id + f"_{i}"
            job_config.use_legacy_sql = False

            logging.info(f"Creating table {job_config.destination} ... ")
            query_job = self.client.query(query=query, job_config=job_config, location=GCP.BigQuery.LOCATION)
            query_job.result()
            if query_job.errors:
                logging.error(query_job.errors)
                raise Exception(f"[!!] Error creating table {job_config.destination}")
