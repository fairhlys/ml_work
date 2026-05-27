import os
import bisect
from log import log


class Calendar(object):
    """读取本项目自带的交易日历，并提供常用交易日查询函数。"""

    def __init__(self):
        """从 resources/calendar 加载交易日，保存为有序列表和集合。"""
        calendarFile = os.path.join(os.path.dirname(__file__), "resources/calendar")
        with open(calendarFile) as f:
            tokens = f.read().splitlines()
            self.tradingDays = sorted([int(x) for x in tokens])
            self.tradingDaySet = set(self.tradingDays)

    def isTradingDay(self, date):
        """判断某个日期是否是交易日。"""
        if not isinstance(date, int):
            date = int(date)
        return date in self.tradingDaySet

    def toTradingDay(self, date):
        """把给定日期转换为当天或之后最近的交易日。"""
        if not isinstance(date, int):
            date = int(date)
        index = bisect.bisect_left(self.tradingDays, date)
        return self.tradingDays[index]

    def next(self, date):
        """返回给定日期之后的下一个交易日。"""
        if not isinstance(date, int):
            date = int(date)
        index = bisect.bisect_right(self.tradingDays, date)
        if index >= len(self.tradingDays):
            return None
        return self.tradingDays[index]

    def prev(self, date):
        """返回给定日期之前的上一个交易日。"""
        if not isinstance(date, int):
            date = int(date)
        index = bisect.bisect_left(self.tradingDays, date)
        if index == 0:
            return None
        return self.tradingDays[index - 1]

    def shift(self, date, n):
        """按交易日向前或向后移动 n 天，n 为负数表示向前。"""
        if not isinstance(date, int):
            date = int(date)
        if not isinstance(n, int):
            log.red("Invalid shift n: {}".format(n))
            return None

        index = bisect.bisect_left(self.tradingDays, date)
        if index == 0:
            log.red("Failed to shift for date {}, n={}".format(date, n))
            return None
        return self.tradingDays[index + n]

    def prevn(self, date, n):
        """返回给定日期之前的 n 个交易日。"""
        if not isinstance(date, int):
            date = int(date)
        if not isinstance(n, int) or n < 1:
            log.red("Invalid prevn: date={},n={}".format(date, n))
            return None

        index = bisect.bisect_left(self.tradingDays, date)
        if index == 0:
            log.red("Failed to find prev trading day for date {}".format(date))
            return None
        if index < n:
            log.yellow("Not enough days for prevn: date={},n={},index={}".format(date, n, index))

        return self.tradingDays[max(index - n, 0) : index]

    def nextn(self, date, n):
        """返回给定日期之后的 n 个交易日。"""
        if not isinstance(date, int):
            date = int(date)
        if not isinstance(n, int) or n < 1:
            log.red("Invalid nextn: date={},n={}".format(date, n))
            return None

        index = bisect.bisect_right(self.tradingDays, date)
        if index >= len(self.tradingDays):
            log.red("Failed to find next trading day for date {}".format(date))
            return None
        if index + n > len(self.tradingDays):
            log.yellow("Not enough days for next: date={},n={},index={}".format(date, n, index))

        return self.tradingDays[index: min(index + n, len(self.tradingDays))]

    def range(self, startDate, endDate):
        """返回闭区间 [startDate, endDate] 内所有交易日。"""
        if not isinstance(startDate, int):
            startDate = int(startDate)
        if not isinstance(endDate, int):
            endDate = int(endDate)
        if startDate > endDate:
            log.red("Invalid range - startDate is larger than endDate: startDate={},endDate={}".format(startDate, endDate))
            return None

        startIndex = bisect.bisect_left(self.tradingDays, startDate)
        if (startIndex == len(self.tradingDays)):
            log.red("No valid trading days found within the range [{}, {})".format(startDate, endDate))
            return None

        endIndex = bisect.bisect_right(self.tradingDays, endDate)
        return self.tradingDays[startIndex : endIndex]
