import numpy as np
from log import log


class MeowEvaluator(object):
    """用常见回归指标评估模型预测效果。"""

    def __init__(self, cacheDir=None):
        """记录配置项，并约定真实收益列和预测收益列的列名。"""
        self.cacheDir = cacheDir
        self.predictionCol = "forecast"
        self.ycol = "fret12"

    def eval(self, ydf):
        """计算 MSE、Pearson 相关系数和 R2，帮助判断预测值是否贴近真实值。"""
        ydf = ydf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ytrue = ydf[self.ycol].to_numpy(dtype=float)
        ypred = ydf[self.predictionCol].to_numpy(dtype=float)
        metrics = self._calcMetrics(ytrue, ypred)
        log.inf(
            "Meow evaluation summary: Pearson correlation={:.4f}, R2={:.5f}, MSE={:.8f}".format(
                metrics["pcor"],
                metrics["r2"],
                metrics["mse"]
            )
        )
        return metrics

    @staticmethod
    def _calcMetrics(ytrue, ypred):
        """根据真实值和预测值计算通用回归指标。"""
        mse = np.mean((ypred - ytrue) ** 2)
        pcor = np.corrcoef(ypred, ytrue)[0, 1] if np.std(ypred) > 0 and np.std(ytrue) > 0 else 0.0
        denom = np.sum((ytrue - np.mean(ytrue)) ** 2)
        r2 = 1.0 - np.sum((ypred - ytrue) ** 2) / denom if denom > 0 else 0.0
        return {"pcor": pcor, "r2": r2, "mse": mse}

    def evalColumn(self, ydf, predictionCol, label=None):
        """评估指定预测列，label 用于在日志里区分 Ridge、Tree 或 Blend。"""
        ydf = ydf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ytrue = ydf[self.ycol].to_numpy(dtype=float)
        ypred = ydf[predictionCol].to_numpy(dtype=float)
        metrics = self._calcMetrics(ytrue, ypred)
        name = label or predictionCol
        log.inf(
            "{} evaluation summary: Pearson correlation={:.4f}, R2={:.5f}, MSE={:.8f}".format(
                name,
                metrics["pcor"],
                metrics["r2"],
                metrics["mse"]
            )
        )
        return metrics

    def evalModelColumns(self, ydf):
        """分别评估 Ridge only、Tree only 和默认 Blend 的效果。"""
        results = {}
        for col, label in [
            ("forecast_ridge", "Ridge only"),
            ("forecast_tree", "Tree only"),
            ("forecast_blend", "Current blend"),
        ]:
            if col in ydf.columns:
                results[col] = self.evalColumn(ydf, col, label=label)
        return results

    def searchBlendWeights(self, ydf, ridgeCol="forecast_ridge", treeCol="forecast_tree", step=0.01):
        """遍历融合权重，寻找 Pearson correlation 最高的 Ridge/Tree 组合。"""
        ydf = ydf.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        ytrue = ydf[self.ycol].to_numpy(dtype=float)
        ridgePred = ydf[ridgeCol].to_numpy(dtype=float)
        treePred = ydf[treeCol].to_numpy(dtype=float)

        best = None
        count = int(round(1.0 / step)) + 1
        for i in range(count):
            ridgeWeight = round(i * step, 10)
            treeWeight = 1.0 - ridgeWeight
            pred = ridgeWeight * ridgePred + treeWeight * treePred
            metrics = self._calcMetrics(ytrue, pred)
            candidate = {
                "ridge_weight": ridgeWeight,
                "tree_weight": treeWeight,
                "pcor": metrics["pcor"],
                "r2": metrics["r2"],
                "mse": metrics["mse"],
            }
            if best is None or (candidate["pcor"], candidate["r2"], -candidate["mse"]) > (
                best["pcor"],
                best["r2"],
                -best["mse"],
            ):
                best = candidate

        log.inf(
            "Best blend search result: Ridge weight={:.2f}, Tree weight={:.2f}, Pearson correlation={:.4f}, R2={:.5f}, MSE={:.8f}".format(
                best["ridge_weight"],
                best["tree_weight"],
                best["pcor"],
                best["r2"],
                best["mse"]
            )
        )
        return best
