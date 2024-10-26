import datetime as dt


def convert_date_to_datetime(date):
    if isinstance(date, str) or isinstance(date, int):
        for date_format in ["%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"]:
            try:
                date = dt.datetime.strptime(str(date), date_format)
                break
            except ValueError:
                continue
    return date
