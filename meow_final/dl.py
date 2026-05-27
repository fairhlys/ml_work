import os
import pandas as pd
from tradingcalendar import Calendar
from log import log


class MeowDataLoader(object):
    """负责从按日期存储的 HDF5 文件中读取原始行情数据。"""

    def __init__(self, h5dir):
        """保存数据目录，并初始化交易日历用于校验日期是否合法。"""
        self.h5dir = h5dir
        self.calendar = Calendar()

    def loadDates(self, dates):
        """批量读取多个交易日的数据，并合并成一个 DataFrame。"""
        if len(dates) == 0:
            raise ValueError("Dates empty")
        log.inf("Loading data of {} dates from {} to {}...".format(len(dates), min(dates), max(dates)))
        return pd.concat(self.loadDate(x) for x in dates)

    def loadDate(self, date):
        """读取单个交易日的 H5 文件，并补充 date 列方便后续分组计算。"""
        if not self.calendar.isTradingDay(date):
            raise ValueError("Not a trading day: {}".format(date))
        h5File = os.path.join(self.h5dir, "{}.h5".format(date))
        df = pd.read_hdf(h5File)
        df.loc[:, "date"] = date
        precols = ["symbol", "interval", "date"]
        df = df[precols + [x for x in df.columns if x not in precols]] # re-arrange columns
        return df
