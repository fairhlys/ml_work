import os
from pathlib import Path
from log import log
from dl import MeowDataLoader
from feat import MeowFeatureGenerator
from mdl import MeowModel
from eval import MeowEvaluator
from tradingcalendar import Calendar


class MeowEngine(object):
    """项目主引擎，串联数据读取、特征生成、模型训练和评估。"""

    def __init__(self, h5dir, cacheDir=None):
        """检查数据目录并初始化各个模块。"""
        self.calendar = Calendar()
        self.h5dir = h5dir
        if not os.path.exists(h5dir):
            raise ValueError("Data directory not exists: {}".format(self.h5dir))
        if not os.path.isdir(h5dir):
            raise ValueError("Invalid data directory: {}".format(self.h5dir))
        self.cacheDir = cacheDir
        self.dloader = MeowDataLoader(h5dir=h5dir)
        self.featGenerator = MeowFeatureGenerator(cacheDir=cacheDir)
        self.model = MeowModel(cacheDir=cacheDir)
        self.evaluator = MeowEvaluator(cacheDir=cacheDir)

    def _loadFeatures(self, dates):
        """读取给定日期列表，并生成模型输入和目标。"""
        rawData = self.dloader.loadDates(dates)
        return self.featGenerator.genFeatures(rawData)

    def _availableDates(self):
        """从数据目录中的 h5 文件自动识别可用交易日。"""
        dates = []
        for path in Path(self.h5dir).glob("*.h5"):
            try:
                date = int(path.stem)
            except ValueError:
                continue
            if self.calendar.isTradingDay(date):
                dates.append(date)
        dates = sorted(set(dates))
        if len(dates) < 10:
            raise ValueError("Not enough h5 trading-day files found in {}".format(self.h5dir))
        return dates

    @staticmethod
    def _splitDates(dates):
        """按时间顺序切分训练、验证、测试集，避免未来数据泄漏。"""
        n = len(dates)
        trainEnd = max(1, int(n * 0.70))
        validEnd = max(trainEnd + 1, int(n * 0.85))
        if validEnd >= n:
            validEnd = n - 1
        return dates[:trainEnd], dates[trainEnd:validEnd], dates[validEnd:]

    def fit(self, startDate, endDate):
        """在指定日期区间内读取训练数据、生成特征并训练模型。"""
        dates = self.calendar.range(startDate, endDate)
        self.fitDates(dates)

    def fitDates(self, dates):
        """在给定交易日列表上训练模型。"""
        log.inf("Running model fitting...")
        xdf, ydf = self._loadFeatures(dates)
        self.model.fit(xdf, ydf)

    def predict(self, xdf):
        """对已经生成好的特征表进行预测。"""
        return self.model.predict(xdf)

    def tuneBlend(self, startDate, endDate):
        """在验证集上搜索最佳 Ridge/Tree 融合权重，并写回模型。"""
        dates = self.calendar.range(startDate, endDate)
        return self.tuneBlendDates(dates)

    def tuneBlendDates(self, dates):
        """在给定验证日列表上搜索最佳 Ridge/Tree 融合权重。"""
        log.inf("Running blend weight tuning...")
        xdf, ydf = self._loadFeatures(dates)
        for col, pred in self.model.predictComponents(xdf).items():
            ydf.loc[:, col] = pred
        self.evaluator.evalModelColumns(ydf)
        best = self.evaluator.searchBlendWeights(ydf)
        self.model.setBlendWeights(best["ridge_weight"], best["tree_weight"])
        return best

    def eval(self, startDate, endDate):
        """在指定日期区间上生成预测，并输出评估指标。"""
        dates = self.calendar.range(startDate, endDate)
        self.evalDates(dates)

    def evalDates(self, dates):
        """在给定测试日列表上生成预测，并输出评估指标。"""
        log.inf("Running model evaluation...")
        xdf, ydf = self._loadFeatures(dates)
        for col, pred in self.model.predictComponents(xdf).items():
            ydf.loc[:, col] = pred
        ydf.loc[:, "forecast"] = ydf["forecast_blend"]
        self.evaluator.evalModelColumns(ydf)

    def runAutoSplit(self):
        """自动识别数据日期，并按 70%/15%/15% 切分训练、验证、测试。"""
        dates = self._availableDates()
        trainDates, validDates, testDates = self._splitDates(dates)
        log.inf(
            "Auto date split: train {}-{} ({} days), valid {}-{} ({} days), test {}-{} ({} days)".format(
                trainDates[0], trainDates[-1], len(trainDates),
                validDates[0], validDates[-1], len(validDates),
                testDates[0], testDates[-1], len(testDates)
            )
        )
        self.fitDates(trainDates)
        self.tuneBlendDates(validDates)
        self.evalDates(testDates)


if __name__ == "__main__":
    # 默认从环境变量读取数据目录；也可以直接把 <data_dir_path> 改成真实路径。
    dataDir = os.environ.get("MEOW_DATA_DIR", "<data_dir_path>")
    engine = MeowEngine(h5dir=dataDir, cacheDir=None)
    if all(os.environ.get(x) for x in [
        "MEOW_TRAIN_START",
        "MEOW_TRAIN_END",
        "MEOW_VALID_START",
        "MEOW_VALID_END",
        "MEOW_TEST_START",
        "MEOW_TEST_END",
    ]):
        engine.fit(int(os.environ["MEOW_TRAIN_START"]), int(os.environ["MEOW_TRAIN_END"]))
        engine.tuneBlend(int(os.environ["MEOW_VALID_START"]), int(os.environ["MEOW_VALID_END"]))
        engine.eval(int(os.environ["MEOW_TEST_START"]), int(os.environ["MEOW_TEST_END"]))
    else:
        engine.runAutoSplit()
