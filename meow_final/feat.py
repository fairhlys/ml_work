import numpy as np
import pandas as pd
from log import log


class MeowFeatureGenerator(object):
    """为分钟级股票预测生成微观结构特征。

    这里的“特征”可以理解为模型看到的输入信息，例如买卖盘强弱、成交
    方向、过去几分钟收益率、同一时刻相对市场平均水平的偏离等。
    """

    def __init__(self, cacheDir=None):
        """设置目标列、索引列，并准备保存本次生成的特征名。"""
        self.cacheDir = cacheDir
        self.ycol = "fret12"
        self.mcols = ["symbol", "date", "interval"]
        self.featureNames = []

    def genFeatures(self, df):
        """主入口：对原始行情表做排序、派生特征，并返回模型输入 x 和目标 y。"""
        log.inf("Generating microstructure features...")
        df = df.copy()
        rawCols = set(df.columns)
        self._sortData(df)
        self._addOrderBookFeatures(df)
        self._addDeepBookFeatures(df)
        self._addWeightedImbalance(df)
        self._addTradeFeatures(df)
        self._addOrderFlowFeatures(df)
        self._addReturnFeatures(df)
        self._addVolatilityFeatures(df)
        self._addRollingFeatures(df)
        self._addInteractionFeatures(df)
        self._addCrossSectionalFeatures(df)
        self._addPerStockFeatures(df)

        self.featureNames = self._collectFeatureNames(df, rawCols)
        idx = pd.MultiIndex.from_arrays(
            [df[c].to_numpy() for c in self.mcols], names=self.mcols
        )
        xcols = {c: df[c].to_numpy(dtype="float32") for c in self.featureNames}
        yarr = df[self.ycol].to_numpy(dtype="float32")
        xdf = pd.DataFrame(xcols, index=idx)
        ydf = pd.DataFrame({self.ycol: yarr}, index=idx)
        xdf = xdf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ydf = ydf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        log.inf("Generated {} features".format(len(self.featureNames)))
        return xdf.astype("float32"), ydf.astype("float32")

    @staticmethod
    def _sortData(df):
        """按股票、日期、分钟顺序排序，保证滞后和滚动窗口不会乱序。"""
        sortCols = [x for x in ["symbol", "date", "interval"] if x in df.columns]
        df.sort_values(sortCols, inplace=True)
        df.reset_index(drop=True, inplace=True)

    @staticmethod
    def _safeDiv(a, b):
        """安全除法：把分母为 0 的位置改成 NaN，避免产生无穷大。"""
        return a / b.replace(0, np.nan)

    def _addOrderBookFeatures(self, df):
        """构造盘口特征，描述买卖报价、挂单深度和买卖盘力量对比。"""
        eps = 1e-12
        if {"ask0", "bid0"}.issubset(df.columns):
            df.loc[:, "spread"] = self._safeDiv(df["ask0"] - df["bid0"], df["midpx"].abs() + eps)
            df.loc[:, "micro_price"] = self._safeDiv(
                df["ask0"] * df["bsize0"] + df["bid0"] * df["asize0"],
                df["bsize0"] + df["asize0"] + eps
            )
            df.loc[:, "micro_price_gap"] = self._safeDiv(df["micro_price"] - df["midpx"], df["midpx"].abs() + eps)

        bookPairs = [
            ("bsize0", "asize0", "ob_imb0"),
            ("bsize0_4", "asize0_4", "ob_imb0_4"),
            ("bsize5_9", "asize5_9", "ob_imb5_9"),
        ]
        for bidCol, askCol, outCol in bookPairs:
            if {bidCol, askCol}.issubset(df.columns):
                df.loc[:, outCol] = self._safeDiv(df[bidCol] - df[askCol], df[bidCol] + df[askCol] + eps)

        valuePairs = [
            ("btr0_4", "atr0_4", "book_turnover_imb0_4"),
            ("btr5_9", "atr5_9", "book_turnover_imb5_9"),
        ]
        for bidCol, askCol, outCol in valuePairs:
            if {bidCol, askCol}.issubset(df.columns):
                df.loc[:, outCol] = self._safeDiv(df[bidCol] - df[askCol], df[bidCol] + df[askCol] + eps)

        if {"bid4", "ask4", "midpx"}.issubset(df.columns):
            # 盘口宽度越大，通常表示买一到卖五之间的价格跨度更大、流动性可能更弱。
            df.loc[:, "book_width0_4"] = self._safeDiv(df["ask4"] - df["bid4"], df["midpx"].abs() + eps)

        if {"bsize0_4", "asize0_4", "midpx"}.issubset(df.columns):
            # log1p 可以压缩特别大的挂单量，降低极端值对模型的影响。
            df.loc[:, "depth0_4"] = np.log1p(df["bsize0_4"].clip(lower=0) + df["asize0_4"].clip(lower=0))

    def _addDeepBookFeatures(self, df):
        """构造更深档位（5-9、10-19）的订单簿特征，补充已有浅档信息。"""
        eps = 1e-12
        if {"bsize10_19", "asize10_19"}.issubset(df.columns):
            df.loc[:, "ob_imb10_19"] = self._safeDiv(
                df["bsize10_19"] - df["asize10_19"],
                df["bsize10_19"] + df["asize10_19"] + eps
            )
            df.loc[:, "depth10_19"] = np.log1p(df["bsize10_19"].clip(lower=0) + df["asize10_19"].clip(lower=0))

        if {"btr10_19", "atr10_19"}.issubset(df.columns):
            df.loc[:, "book_turnover_imb10_19"] = self._safeDiv(
                df["btr10_19"] - df["atr10_19"],
                df["btr10_19"] + df["atr10_19"] + eps
            )

        if {"bid19", "ask19", "midpx"}.issubset(df.columns):
            df.loc[:, "book_width0_19"] = self._safeDiv(df["ask19"] - df["bid19"], df["midpx"].abs() + eps)
            # 全档位深度
            cols = [x for x in ["bsize0_4", "asize0_4", "bsize5_9", "asize5_9", "bsize10_19", "asize10_19"] if x in df.columns]
            if len(cols) >= 2:
                totalDepth = sum(df[c].clip(lower=0) for c in cols)
                df.loc[:, "depth_full"] = np.log1p(totalDepth)

    def _addWeightedImbalance(self, df):
        """用价格距离加权各档位不平衡，越近的档位权重越高。"""
        eps = 1e-12
        sizeCols = ["bsize0", "bsize0_4", "bsize5_9", "bsize10_19"]
        askCols = ["asize0", "asize0_4", "asize5_9", "asize10_19"]
        available = [b for b, a in zip(sizeCols, askCols) if {b, a}.issubset(df.columns)]
        if len(available) < 2:
            return
        weights = [4.0, 3.0, 2.0, 1.0]
        wBid, wAsk, wSum = np.zeros(len(df)), np.zeros(len(df)), 0.0
        for i, (b, a) in enumerate(zip(sizeCols, askCols)):
            if b in df.columns and a in df.columns:
                w = weights[i]
                wBid += w * df[b].clip(lower=0)
                wAsk += w * df[a].clip(lower=0)
                wSum += w
        if wSum > 0:
            df.loc[:, "w_ob_imb"] = self._safeDiv(wBid - wAsk, wBid + wAsk + eps)
            df.loc[:, "w_depth"] = np.log1p((wBid + wAsk) / wSum)

    def _addOrderFlowFeatures(self, df):
        """利用 Add/Cxl 事件构造订单流特征，反映真实买卖供需压力。"""
        eps = 1e-12
        if {"addBuyQty", "addSellQty", "cxlBuyQty", "cxlSellQty"}.issubset(df.columns):
            netBuy = df["addBuyQty"] - df["cxlBuyQty"]
            netSell = df["addSellQty"] - df["cxlSellQty"]
            df.loc[:, "order_flow_qty_imb"] = self._safeDiv(netBuy - netSell, netBuy.abs() + netSell.abs() + eps)
            df.loc[:, "add_qty_imb"] = self._safeDiv(
                df["addBuyQty"] - df["addSellQty"],
                df["addBuyQty"] + df["addSellQty"] + eps
            )
            df.loc[:, "cxl_qty_imb"] = self._safeDiv(
                df["cxlBuyQty"] - df["cxlSellQty"],
                df["cxlBuyQty"] + df["cxlSellQty"] + eps
            )
            df.loc[:, "add_qty_sum_log"] = np.log1p((df["addBuyQty"] + df["addSellQty"]).clip(lower=0))
            df.loc[:, "cxl_qty_sum_log"] = np.log1p((df["cxlBuyQty"] + df["cxlSellQty"]).clip(lower=0))
            # 撤单比例高说明不确定性大
            addTotal = df["addBuyQty"] + df["addSellQty"]
            cxlTotal = df["cxlBuyQty"] + df["cxlSellQty"]
            df.loc[:, "cxl_add_ratio"] = self._safeDiv(cxlTotal, addTotal + eps)

        if {"nAddBuy", "nAddSell", "nCxlBuy", "nCxlSell"}.issubset(df.columns):
            df.loc[:, "add_count_imb"] = self._safeDiv(
                df["nAddBuy"] - df["nAddSell"],
                df["nAddBuy"] + df["nAddSell"] + eps
            )
            df.loc[:, "cxl_count_imb"] = self._safeDiv(
                df["nCxlBuy"] - df["nCxlSell"],
                df["nCxlBuy"] + df["nCxlSell"] + eps
            )

        if {"addBuyTurnover", "addSellTurnover", "cxlBuyTurnover", "cxlSellTurnover"}.issubset(df.columns):
            df.loc[:, "add_turnover_imb"] = self._safeDiv(
                df["addBuyTurnover"] - df["addSellTurnover"],
                df["addBuyTurnover"] + df["addSellTurnover"] + eps
            )
            df.loc[:, "cxl_turnover_imb"] = self._safeDiv(
                df["cxlBuyTurnover"] - df["cxlSellTurnover"],
                df["cxlBuyTurnover"] + df["cxlSellTurnover"] + eps
            )

        if {"addBuyQty", "addSellQty", "tradeBuyQty", "tradeSellQty"}.issubset(df.columns):
            flowTotal = df["addBuyQty"] + df["addSellQty"]
            tradeTotal = df["tradeBuyQty"] + df["tradeSellQty"]
            df.loc[:, "flow_trade_ratio"] = self._safeDiv(flowTotal, tradeTotal + eps)

    def _addVolatilityFeatures(self, df):
        """构造已实现波动率和非流动性特征。"""
        eps = 1e-12
        if "lag_ret1" not in df.columns:
            return
        groupKeys = ["symbol", "date"]
        retGroup = df.groupby(groupKeys, sort=False)["lag_ret1"]
        for win in [3, 6, 12, 24]:
            shifted = retGroup.shift(1)
            df.loc[:, "realized_vol_{}".format(win)] = (
                shifted.groupby([df["symbol"], df["date"]], sort=False)
                .rolling(win, min_periods=3)
                .std()
                .reset_index(level=[0, 1], drop=True)
            )

        if "trade_qty_sum" in df.columns:
            absRet = retGroup.shift(1).abs()
            qtyGroup = df.groupby(groupKeys, sort=False)["trade_qty_sum"]
            turnover = qtyGroup.shift(1)
            illiq = self._safeDiv(absRet, turnover + eps)
            for win in [3, 6]:
                df.loc[:, "amihud_{}".format(win)] = (
                    illiq.groupby([df["symbol"], df["date"]], sort=False)
                    .rolling(win, min_periods=3)
                    .mean()
                    .reset_index(level=[0, 1], drop=True)
                )

    def _addInteractionFeatures(self, df):
        """构造关键特征之间的交互项。"""
        eps = 1e-12
        pairs = [
            ("spread", "trade_qty_imb"),
            ("ob_imb0", "lag_ret1"),
            ("spread", "trade_turnover_imb"),
        ]
        for a, b in pairs:
            if a in df.columns and b in df.columns:
                df.loc[:, "{}_x_{}".format(a, b)] = df[a] * df[b]

        if "realized_vol_12" in df.columns:
            if "trade_qty_imb" in df.columns:
                df.loc[:, "vol_x_trade_imb"] = self._safeDiv(
                    df["trade_qty_imb"], df["realized_vol_12"] + eps
                )
            if "spread" in df.columns:
                df.loc[:, "spread_x_vol"] = df["spread"] * df["realized_vol_12"]

    def _addPerStockFeatures(self, df):
        """对关键特征做股票内滚动 z-score 标准化，消除个股量纲差异。"""
        groupKeys = ["symbol", "date"]
        cols = [x for x in ["spread", "trade_qty_imb", "ob_imb0", "order_flow_qty_imb", "lag_ret1"] if x in df.columns]
        for col in cols:
            shifted = df.groupby(groupKeys, sort=False)[col].shift(1)
            rollMean = (
                shifted.groupby([df["symbol"], df["date"]], sort=False)
                .rolling(12, min_periods=6)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
            rollStd = (
                shifted.groupby([df["symbol"], df["date"]], sort=False)
                .rolling(12, min_periods=6)
                .std()
                .reset_index(level=[0, 1], drop=True)
            )
            df.loc[:, "{}_sz".format(col)] = self._safeDiv(shifted - rollMean, rollStd)

    def _addTradeFeatures(self, df):
        """构造成交特征，描述主动买卖成交量、成交额和成交笔数的差异。"""
        eps = 1e-12
        if {"tradeBuyQty", "tradeSellQty"}.issubset(df.columns):
            qtySum = df["tradeBuyQty"] + df["tradeSellQty"] + eps
            df.loc[:, "trade_qty_imb"] = self._safeDiv(df["tradeBuyQty"] - df["tradeSellQty"], qtySum)
            df.loc[:, "trade_qty_sum"] = np.log1p(qtySum.clip(lower=0))

        if {"tradeBuyTurnover", "tradeSellTurnover"}.issubset(df.columns):
            turnoverSum = df["tradeBuyTurnover"] + df["tradeSellTurnover"] + eps
            df.loc[:, "trade_turnover_imb"] = self._safeDiv(
                df["tradeBuyTurnover"] - df["tradeSellTurnover"],
                turnoverSum
            )
            df.loc[:, "trade_turnover_sum"] = np.log1p(turnoverSum.clip(lower=0))

        if {"nTradeBuy", "nTradeSell"}.issubset(df.columns):
            countSum = df["nTradeBuy"] + df["nTradeSell"] + eps
            df.loc[:, "trade_count_imb"] = self._safeDiv(df["nTradeBuy"] - df["nTradeSell"], countSum)
            df.loc[:, "trade_count_sum"] = np.log1p(countSum.clip(lower=0))

        if {"tradeBuyHigh", "tradeBuyLow", "midpx"}.issubset(df.columns):
            df.loc[:, "trade_buy_range"] = self._safeDiv(
                df["tradeBuyHigh"] - df["tradeBuyLow"],
                df["midpx"].abs() + eps
            )

        if {"tradeSellHigh", "tradeSellLow", "midpx"}.issubset(df.columns):
            df.loc[:, "trade_sell_range"] = self._safeDiv(
                df["tradeSellHigh"] - df["tradeSellLow"],
                df["midpx"].abs() + eps
            )

        for col in ["buyVwad", "sellVwad"]:
            if col in df.columns:
                df.loc[:, "{}_log".format(col)] = np.log1p(df[col].clip(lower=0))

    def _addReturnFeatures(self, df):
        """构造价格收益相关特征，包括当前 K 线变化和过去若干分钟的收益率。"""
        eps = 1e-12
        if {"open", "high", "low", "lastpx", "midpx"}.issubset(df.columns):
            df.loc[:, "bar_ret"] = self._safeDiv(df["lastpx"] - df["open"], df["open"].abs() + eps)
            df.loc[:, "bar_range"] = self._safeDiv(df["high"] - df["low"], df["midpx"].abs() + eps)
            df.loc[:, "close_mid_gap"] = self._safeDiv(df["lastpx"] - df["midpx"], df["midpx"].abs() + eps)

        group = df.groupby(["symbol", "date"], sort=False)
        for win in [1, 3, 6, 12, 24, 60]:
            base = group["midpx"].shift(win)
            df.loc[:, "lag_ret{}".format(win)] = self._safeDiv(df["midpx"] - base, base.abs() + eps)

    def _addRollingFeatures(self, df):
        """构造滚动窗口特征，用历史均值和波动率概括最近一段时间的状态。"""
        groupKeys = ["symbol", "date"]
        windows = [3, 6, 12, 24, 60]
        baseCols = [x for x in [
            "lag_ret1", "trade_qty_imb", "ob_imb0", "spread",
            "order_flow_qty_imb", "cxl_qty_imb", "w_ob_imb",
            "realized_vol_3", "add_qty_imb", "add_turnover_imb",
        ] if x in df.columns]

        for col in baseCols:
            group = df.groupby(groupKeys, sort=False)[col]
            for win in windows:
                shifted = group.shift(1)
                df.loc[:, "{}_mean{}".format(col, win)] = shifted.groupby(
                    [df["symbol"], df["date"]], sort=False
                ).rolling(win, min_periods=2).mean().reset_index(level=[0, 1], drop=True)
                df.loc[:, "{}_std{}".format(col, win)] = shifted.groupby(
                    [df["symbol"], df["date"]], sort=False
                ).rolling(win, min_periods=2).std().reset_index(level=[0, 1], drop=True)

        if "trade_qty_imb" in df.columns:
            # EMA 更重视最近的数据，适合刻画成交方向是否正在持续偏向买入或卖出。
            df.loc[:, "trade_qty_imb_ema6"] = df.groupby(groupKeys, sort=False)["trade_qty_imb"].transform(
                lambda s: s.shift(1).ewm(halflife=6, adjust=False).mean()
            )
        if "order_flow_qty_imb" in df.columns:
            df.loc[:, "order_flow_qty_imb_ema12"] = df.groupby(groupKeys, sort=False)["order_flow_qty_imb"].transform(
                lambda s: s.shift(1).ewm(halflife=12, adjust=False).mean()
            )

    def _addCrossSectionalFeatures(self, df):
        """构造横截面特征，比较某只股票在同一分钟相对其他股票的位置。"""
        crossKeys = ["date", "interval"]
        sourceCols = [x for x in [
            "lag_ret12", "lag_ret24", "trade_qty_imb", "ob_imb0_4", "spread",
            "order_flow_qty_imb", "cxl_qty_imb", "ob_imb10_19",
            "realized_vol_12", "w_ob_imb",
        ] if x in df.columns]
        for col in sourceCols:
            mean = df.groupby(crossKeys, sort=False)[col].transform("mean")
            std = df.groupby(crossKeys, sort=False)[col].transform("std")
            df.loc[:, "{}_cx".format(col)] = df[col] - mean
            df.loc[:, "{}_z".format(col)] = self._safeDiv(df[col] - mean, std)

    def _collectFeatureNames(self, df, rawCols):
        """只收集本类新生成的数值列，避免把标签列或原始标识列误当作特征。"""
        excluded = set(self.mcols + [self.ycol, "symbol"])
        featureNames = []
        for c in df.columns:
            if c in excluded or c in rawCols:
                continue
            if df[c].dtype.kind in ("f", "i"):
                featureNames.append(c)
        return sorted(featureNames)
