
class MarginalPriceParams:
    RAW_FILE_NAME = "marginalpdbc"
    SKIP_ROWS = 1
    COL_NAMES = ["year", "month", "day", "hour", "portugal", "spain"]


class OfferCurvesParams:
    RAW_FILE_NAME = "curva_pbc_uof"
    SKIP_ROWS = 3
    COL_NAMES = [
        "hour", "date", "country", "unit", "offer_type", "energy", "price", "status"
    ]

    class OfferType:
        BID = "C"
        ASK = "V"

    class OfferStatus:
        OFFERED = "O"
        CLEARED = "C"
