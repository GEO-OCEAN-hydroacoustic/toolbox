import datetime
import numpy as np

LEAP_TIMES = [datetime.datetime(1981,7,1),
                datetime.datetime(1982,7,1),
                datetime.datetime(1983,7,1),
                datetime.datetime(1985,7,1),
                datetime.datetime(1988,1,1),
                datetime.datetime(1990,1,1),
                datetime.datetime(1991,1,1),
                datetime.datetime(1992,7,1),
                datetime.datetime(1993,7,1),
                datetime.datetime(1994,7,1),
                datetime.datetime(1996,1,1),
                datetime.datetime(1997,7,1),
                datetime.datetime(1999,1,1),
                datetime.datetime(2006,1,1),
                datetime.datetime(2009,1,1),
                datetime.datetime(2012,7,1),
                datetime.datetime(2015,7,1),
                datetime.datetime(2017,1,1)]

def get_leap_second(date):
    return np.searchsorted(LEAP_TIMES, date)
