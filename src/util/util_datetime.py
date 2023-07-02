from datetime import date, datetime, timedelta


def to_yyyymmdd(date: date) -> str:
    return date.strftime("%Y%m%d")


def from_yyyymmdd(date_string: str) -> date:
    return datetime.strptime(date_string, "%Y%m%d").date()


class DailyDateIterator(object):
    delta: timedelta = timedelta(days=1)

    def __init__(self, start: date, end: date):
        if not isinstance(start, date):
            raise TypeError("start must be a date object")
        if not isinstance(end, date):
            raise TypeError("end must be a date object")

        if start > end:
            raise ValueError("The start date must be earlier than the end date.")

        self.current = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.current > self.end:
            raise StopIteration()

        ret = self.current
        self.current += self.delta
        return ret
