import os
import numpy as np
from log import log
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except ImportError:
    LGBMRegressor = None


def _setupCudaTempDirs():
    """把 CuPy/NVRTC 临时目录放到纯英文路径，规避 Windows 中文用户名乱码问题。"""
    baseDir = os.path.dirname(os.path.abspath(__file__))
    tempDir = os.path.join(baseDir, ".cuda_tmp")
    cacheDir = os.path.join(baseDir, ".cupy_cache")
    os.makedirs(tempDir, exist_ok=True)
    os.makedirs(cacheDir, exist_ok=True)

    os.environ.setdefault("TEMP", tempDir)
    os.environ.setdefault("TMP", tempDir)
    os.environ.setdefault("TMPDIR", tempDir)
    os.environ.setdefault("CUPY_CACHE_DIR", cacheDir)


_setupCudaTempDirs()

try:
    import cupy as cp
except ImportError:
    cp = None


class MeowModel(object):
    """股票收益率预测模型。

    这个类把一个线性模型 Ridge 和一个树模型组合起来：Ridge 更稳定，
    树模型更擅长学习非线性关系，最后按固定权重融合两者的预测结果。
    """

    def __init__(self, cacheDir=None, ridgeAlpha=100.0, treeRows=800000,
                 useGpuTreeModel=None, cpuTreeParams=None, gpuTreeParams=None,
                 treeModelType="xgboost", lgbTreeParams=None,
                 useFeatureSelection=False, featuresToKeep=130):
        """初始化模型、训练参数和预测值裁剪配置。"""
        self.cacheDir = cacheDir
        self.ridgeAlpha = ridgeAlpha
        self.treeRows = treeRows
        self.cpuTreeParams = self._defaultCpuTreeParams()
        if cpuTreeParams is not None:
            self.cpuTreeParams.update(cpuTreeParams)
        self.gpuTreeParams = self._defaultGpuTreeParams()
        if gpuTreeParams is not None:
            self.gpuTreeParams.update(gpuTreeParams)
        self.lgbTreeParams = self._defaultLgbTreeParams()
        if lgbTreeParams is not None:
            self.lgbTreeParams.update(lgbTreeParams)
        self.treeModelType = treeModelType  # "xgboost", "lightgbm", or "sklearn"
        self.useFeatureSelection = useFeatureSelection
        self.featuresToKeep = featuresToKeep
        self.selectedFeatureIdx = None
        self.selectedFeatureNames = None
        # 线性模型流水线：缺失值填补 -> 标准化 -> Ridge 回归。
        self.linearModel = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=self.ridgeAlpha, fit_intercept=True, solver="lsqr", random_state=2025)
        )
        # CPU 版本的梯度提升树，对特征缩放不敏感，能学习非线性关系。
        self.treeModel = self._newCpuTreeModel()
        self.useTreeModel = True
        self.useGpuTreeModel = XGBRegressor is not None if useGpuTreeModel is None else bool(useGpuTreeModel)
        self.useGpuPrediction = self.useGpuTreeModel and XGBRegressor is not None and cp is not None
        if treeModelType == "lightgbm" and LGBMRegressor is None:
            log.yellow("LightGBM not installed, fallback to XGBoost")
            self.treeModelType = "xgboost"
        self.clipValue = None
        self.linearWeight = 0.47
        self.treeWeight = 0.53

    @staticmethod
    def _defaultCpuTreeParams():
        """返回 sklearn CPU 树模型的默认参数。"""
        return {
            "loss": "squared_error",
            "learning_rate": 0.045,
            "max_iter": 180,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 80,
            "l2_regularization": 0.05,
            "random_state": 2025,
        }

    @staticmethod
    def _defaultGpuTreeParams():
        """返回 XGBoost GPU 树模型的默认参数。"""
        return {
            "objective": "reg:squarederror",
            "n_estimators": 500,
            "learning_rate": 0.025,
            "max_depth": 5,
            "min_child_weight": 100,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 3.0,
            "reg_alpha": 0.0,
            "tree_method": "hist",
            "device": "cuda",
            "n_jobs": 0,
            "random_state": 2025,
        }

    @staticmethod
    def _defaultLgbTreeParams():
        """返回 LightGBM GPU 树模型的默认参数。"""
        return {
            "objective": "regression",
            "n_estimators": 500,
            "learning_rate": 0.025,
            "max_depth": 5,
            "num_leaves": 31,
            "min_child_samples": 100,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 3.0,
            "reg_alpha": 0.0,
            "device": "gpu",
            "random_state": 2025,
            "verbose": -1,
        }

    def _newCpuTreeModel(self):
        """创建 sklearn CPU 树模型。"""
        return HistGradientBoostingRegressor(**self.cpuTreeParams)

    def _newLgbTreeModel(self):
        """创建 LightGBM GPU 树模型。"""
        return LGBMRegressor(**self.lgbTreeParams)

    def setBlendWeights(self, linearWeight, treeWeight):
        """设置 Ridge 和树模型的融合权重，两个权重通常应相加为 1。"""
        self.linearWeight = float(linearWeight)
        self.treeWeight = float(treeWeight)
        log.inf(
            "Set blend weights: Ridge={:.2f}, Tree={:.2f}".format(
                self.linearWeight,
                self.treeWeight
            )
        )

    def _newGpuTreeModel(self):
        """创建 XGBoost GPU 树模型；如果环境没有 GPU 或 XGBoost 会自动回退。"""
        return XGBRegressor(**self.gpuTreeParams)

    def selectFeatures(self, xdf, ydf, topN=None):
        """训练一个快速 XGBoost 后用 feature importance 选择 topN 特征。"""
        if topN is None:
            topN = self.featuresToKeep
        x, y = self._prepareTrainingData(xdf, ydf)
        n = min(x.shape[0], 500000)
        rng = np.random.default_rng(2025)
        idx = rng.choice(x.shape[0], size=n, replace=False)
        probe = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                             tree_method="hist", device="cuda" if self.useGpuTreeModel else "cpu",
                             random_state=2025, verbosity=0)
        probe.fit(x[idx], y[idx])
        importance = probe.feature_importances_
        topk = min(topN, x.shape[1])
        topIdx = np.argsort(importance)[-topk:]
        self.selectedFeatureIdx = np.sort(topIdx)
        self.selectedFeatureNames = [xdf.columns[i] for i in self.selectedFeatureIdx]
        log.inf("Feature selection: kept {}/{} features".format(topk, x.shape[1]))
        return self.selectedFeatureNames

    def _applyFeatureSelection(self, xdf):
        """如果启用了特征筛选，只保留选中的列。"""
        if self.selectedFeatureNames is None:
            return xdf
        return xdf[self.selectedFeatureNames]

    def _prepareTrainingData(self, xdf, ydf):
        """把 DataFrame 转成训练数组，并对目标值做缺失值处理和极端值裁剪。"""
        x = xdf.to_numpy(dtype=np.float32)
        y = ydf.to_numpy(dtype=np.float32).reshape(-1)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        self.clipValue = float(np.nanpercentile(np.abs(y), 99.5))
        if self.clipValue <= 0:
            self.clipValue = 1.0
        y = np.clip(y, -self.clipValue, self.clipValue)
        return x, y

    def fitLinearOnly(self, xdf, ydf):
        """只训练 Ridge 线性模型，用于快速比较不同 alpha。"""
        x, y = self._prepareTrainingData(xdf, ydf)
        log.inf(
            "Fitting ridge-only model alpha={} on {} rows and {} features...".format(
                self.ridgeAlpha,
                x.shape[0],
                x.shape[1]
            )
        )
        self.linearModel.fit(x, y)

    def fit(self, xdf, ydf):
        """训练模型：先清洗目标值，再训练 Ridge，最后抽样训练树模型。"""
        # 特征筛选（如果启用）
        if self.useFeatureSelection:
            self.selectFeatures(xdf, ydf)
            xdf = self._applyFeatureSelection(xdf)

        x, y = self._prepareTrainingData(xdf, ydf)

        log.inf("Fitting ridge model alpha={} on {} rows and {} features...".format(self.ridgeAlpha, x.shape[0], x.shape[1]))
        self.linearModel.fit(x, y)

        # 树模型训练更慢，因此最多抽样 80 万行，兼顾速度和样本覆盖。
        treeRows = min(x.shape[0], self.treeRows)
        if treeRows < 10000:
            self.useTreeModel = False
            log.inf("Skip tree model because training rows are limited")
            return

        rng = np.random.default_rng(2025)
        sampleIndex = rng.choice(x.shape[0], size=treeRows, replace=False)

        # LightGBM GPU
        if self.treeModelType == "lightgbm" and LGBMRegressor is not None:
            self.treeModel = self._newLgbTreeModel()
            try:
                log.inf("Fitting LightGBM GPU model on {} sampled rows...".format(treeRows))
                self.treeModel.fit(x[sampleIndex], y[sampleIndex])
                log.inf("Done fitting with LightGBM GPU model")
                return
            except Exception as exc:
                self.treeModelType = "xgboost"
                log.yellow("LightGBM GPU failed, fallback to XGBoost: {}".format(exc))

        # XGBoost GPU
        if self.treeModelType == "xgboost" and XGBRegressor is not None:
            self.treeModel = self._newGpuTreeModel()
            try:
                log.inf("Fitting XGBoost GPU model on {} sampled rows...".format(treeRows))
                self.treeModel.fit(x[sampleIndex], y[sampleIndex], verbose=False)
                log.inf("Done fitting with GPU tree model")
                return
            except Exception as exc:
                self.useGpuTreeModel = False
                log.yellow("GPU tree model failed, fallback to sklearn CPU model: {}".format(exc))
                self.treeModel = self._newCpuTreeModel()

        log.inf("Fitting histogram gradient boosting CPU model on {} sampled rows...".format(treeRows))
        self.treeModel.fit(x[sampleIndex], y[sampleIndex])
        log.inf("Done fitting")

    def _predictTree(self, x):
        """生成树模型预测；优先尝试 GPU 输入，失败时自动回退到普通 NumPy 输入。"""
        if self.treeModelType == "lightgbm":
            return self.treeModel.predict(x)

        if self.useGpuTreeModel and self.useGpuPrediction:
            try:
                return cp.asnumpy(self.treeModel.predict(cp.asarray(x)))
            except Exception as exc:
                self.useGpuPrediction = False
                log.yellow("CuPy GPU prediction failed, fallback to CPU input prediction: {}".format(exc))

        if self.useGpuTreeModel and cp is None:
            log.yellow("CuPy is not installed; XGBoost GPU prediction will use CPU input data")
        return self.treeModel.predict(x)

    def predictComponents(self, xdf):
        """同时返回 Ridge、Tree 和默认融合预测，便于对比不同模型贡献。"""
        xdf = self._applyFeatureSelection(xdf)
        x = xdf.to_numpy(dtype=np.float32)
        linearPred = self.linearModel.predict(x)
        if not self.useTreeModel:
            treePred = linearPred.copy()
        else:
            treePred = self._predictTree(x)
        blendPred = self.linearWeight * linearPred + self.treeWeight * treePred
        return {
            "forecast_ridge": np.clip(linearPred, -self.clipValue, self.clipValue),
            "forecast_tree": np.clip(treePred, -self.clipValue, self.clipValue),
            "forecast_blend": np.clip(blendPred, -self.clipValue, self.clipValue),
        }

    def predict(self, xdf):
        """生成当前权重下的融合预测。"""
        pred = self.predictComponents(xdf)["forecast_blend"]
        return np.clip(pred, -self.clipValue, self.clipValue)
