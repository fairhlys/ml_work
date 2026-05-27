import csv
import os
from datetime import datetime

from dl import MeowDataLoader
from eval import MeowEvaluator
from feat import MeowFeatureGenerator
from log import log
from mdl import MeowModel
from tradingcalendar import Calendar


def _parseFloatList(value, default):
    """把环境变量里的逗号分隔数字转成 float 列表。"""
    if not value:
        return default
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _baseRow(experiment, ridgeAlpha, treeRows, useGpuTree, bestWeight, validMetrics, testMetrics):
    """整理通用实验结果字段。"""
    return {
        "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": experiment,
        "ridge_alpha": ridgeAlpha,
        "tree_rows": treeRows,
        "use_gpu_tree": useGpuTree,
        "valid_best_ridge_weight": bestWeight["ridge_weight"],
        "valid_best_tree_weight": bestWeight["tree_weight"],
        "valid_ridge_pcor": _metricValue(validMetrics, "forecast_ridge", "pcor"),
        "valid_tree_pcor": _metricValue(validMetrics, "forecast_tree", "pcor"),
        "valid_blend_pcor": _metricValue(validMetrics, "forecast_blend", "pcor"),
        "valid_blend_r2": _metricValue(validMetrics, "forecast_blend", "r2"),
        "valid_blend_mse": _metricValue(validMetrics, "forecast_blend", "mse"),
        "test_ridge_pcor": _metricValue(testMetrics, "forecast_ridge", "pcor"),
        "test_tree_pcor": _metricValue(testMetrics, "forecast_tree", "pcor"),
        "test_blend_pcor": _metricValue(testMetrics, "forecast_blend", "pcor"),
        "test_blend_r2": _metricValue(testMetrics, "forecast_blend", "r2"),
        "test_blend_mse": _metricValue(testMetrics, "forecast_blend", "mse"),
    }


def _loadFeatureData(h5dir, startDate, endDate):
    """读取一个日期区间的数据，并生成 x/y 特征表。"""
    calendar = Calendar()
    dloader = MeowDataLoader(h5dir=h5dir)
    featGenerator = MeowFeatureGenerator()
    dates = calendar.range(startDate, endDate)
    rawData = dloader.loadDates(dates)
    return featGenerator.genFeatures(rawData)


def _evaluateModel(model, evaluator, xdf, ydf, tuneWeights=False):
    """评估模型；可选地在该数据集上搜索并应用融合权重。"""
    ydf = ydf.copy()
    for col, pred in model.predictComponents(xdf).items():
        ydf.loc[:, col] = pred
    modelMetrics = evaluator.evalModelColumns(ydf)
    bestWeight = None
    if tuneWeights:
        bestWeight = evaluator.searchBlendWeights(ydf)
        model.setBlendWeights(bestWeight["ridge_weight"], bestWeight["tree_weight"])
        for col, pred in model.predictComponents(xdf).items():
            ydf.loc[:, col] = pred
        modelMetrics = evaluator.evalModelColumns(ydf)
    return modelMetrics, bestWeight


def _metricValue(metrics, col, name):
    """从评估结果中安全取出某个指标。"""
    return metrics.get(col, {}).get(name, "")


def _writeRows(csvPath, rows):
    """把实验结果追加写入 CSV，文件不存在时自动写表头。"""
    os.makedirs(os.path.dirname(csvPath), exist_ok=True)
    fieldnames = list(rows[0].keys())
    needHeader = not os.path.exists(csvPath)
    with open(csvPath, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if needHeader:
            writer.writeheader()
        writer.writerows(rows)


def runRidgeAlphaExperiment():
    """固定树模型，只比较不同 Ridge alpha 对验证集和测试集表现的影响。"""
    h5dir = os.environ.get("MEOW_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
    outCsv = os.environ.get("MEOW_EXPERIMENT_CSV", os.path.join(os.path.dirname(__file__), "实验结果.csv"))
    alphas = _parseFloatList(os.environ.get("MEOW_RIDGE_ALPHAS"), [50.0, 100.0, 200.0, 500.0])
    treeRows = int(os.environ.get("MEOW_TREE_ROWS", "800000"))
    useGpuTree = os.environ.get("MEOW_USE_GPU_TREE", "0") == "1"

    trainRange = (20230601, 20231031)
    validRange = (20231101, 20231130)
    testRange = (20231201, 20231229)

    log.inf("Loading train/valid/test feature data for ridge alpha experiment...")
    trainX, trainY = _loadFeatureData(h5dir, trainRange[0], trainRange[1])
    validX, validY = _loadFeatureData(h5dir, validRange[0], validRange[1])
    testX, testY = _loadFeatureData(h5dir, testRange[0], testRange[1])

    evaluator = MeowEvaluator()
    baseModel = None
    rows = []

    for idx, alpha in enumerate(alphas):
        log.inf("Running ridge alpha experiment: alpha={}".format(alpha))
        model = MeowModel(ridgeAlpha=alpha, treeRows=treeRows, useGpuTreeModel=useGpuTree)
        if idx == 0:
            model.fit(trainX, trainY)
            baseModel = model
        else:
            model.treeModel = baseModel.treeModel
            model.useTreeModel = baseModel.useTreeModel
            model.useGpuTreeModel = baseModel.useGpuTreeModel
            model.useGpuPrediction = baseModel.useGpuPrediction
            model.fitLinearOnly(trainX, trainY)

        validMetrics, bestWeight = _evaluateModel(model, evaluator, validX, validY, tuneWeights=True)
        testMetrics, _ = _evaluateModel(model, evaluator, testX, testY, tuneWeights=False)

        row = _baseRow("ridge_alpha", alpha, treeRows, useGpuTree, bestWeight, validMetrics, testMetrics)
        row.update({
            "cpu_learning_rate": "",
            "cpu_max_iter": "",
            "cpu_max_leaf_nodes": "",
            "cpu_min_samples_leaf": "",
            "cpu_l2_regularization": "",
        })
        rows.append(row)

    _writeRows(outCsv, rows)
    log.inf("Experiment results written to {}".format(outCsv))


def _cpuTreeConfigs():
    """返回一组小规模 CPU 树模型粗搜配置。"""
    return [
        {
            "name": "baseline",
            "params": {
                "learning_rate": 0.045,
                "max_iter": 180,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 80,
                "l2_regularization": 0.05,
            },
        },
        {
            "name": "stronger_regularization",
            "params": {
                "learning_rate": 0.045,
                "max_iter": 220,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 150,
                "l2_regularization": 0.50,
            },
        },
        {
            "name": "smaller_tree",
            "params": {
                "learning_rate": 0.050,
                "max_iter": 220,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 120,
                "l2_regularization": 0.20,
            },
        },
        {
            "name": "larger_tree",
            "params": {
                "learning_rate": 0.035,
                "max_iter": 260,
                "max_leaf_nodes": 63,
                "min_samples_leaf": 80,
                "l2_regularization": 0.20,
            },
        },
    ]


def runCpuTreeExperiment():
    """比较几组 CPU 梯度提升树参数，观察树模型复杂度和正则化方向。"""
    h5dir = os.environ.get("MEOW_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
    outCsv = os.environ.get("MEOW_EXPERIMENT_CSV", os.path.join(os.path.dirname(__file__), "实验结果.csv"))
    ridgeAlpha = float(os.environ.get("MEOW_RIDGE_ALPHA", "100.0"))
    treeRows = int(os.environ.get("MEOW_TREE_ROWS", "800000"))

    trainRange = (20230601, 20231031)
    validRange = (20231101, 20231130)
    testRange = (20231201, 20231229)

    log.inf("Loading train/valid/test feature data for CPU tree experiment...")
    trainX, trainY = _loadFeatureData(h5dir, trainRange[0], trainRange[1])
    validX, validY = _loadFeatureData(h5dir, validRange[0], validRange[1])
    testX, testY = _loadFeatureData(h5dir, testRange[0], testRange[1])

    evaluator = MeowEvaluator()
    rows = []

    for cfg in _cpuTreeConfigs():
        params = cfg["params"]
        log.inf("Running CPU tree experiment: {}".format(cfg["name"]))
        model = MeowModel(
            ridgeAlpha=ridgeAlpha,
            treeRows=treeRows,
            useGpuTreeModel=False,
            cpuTreeParams=params,
        )
        model.fit(trainX, trainY)
        validMetrics, bestWeight = _evaluateModel(model, evaluator, validX, validY, tuneWeights=True)
        testMetrics, _ = _evaluateModel(model, evaluator, testX, testY, tuneWeights=False)

        row = _baseRow("cpu_tree_" + cfg["name"], ridgeAlpha, treeRows, False, bestWeight, validMetrics, testMetrics)
        row.update({
            "cpu_learning_rate": params["learning_rate"],
            "cpu_max_iter": params["max_iter"],
            "cpu_max_leaf_nodes": params["max_leaf_nodes"],
            "cpu_min_samples_leaf": params["min_samples_leaf"],
            "cpu_l2_regularization": params["l2_regularization"],
        })
        rows.append(row)

    _writeRows(outCsv, rows)
    log.inf("Experiment results written to {}".format(outCsv))


def _gpuTreeConfigs():
    """返回一组小规模 XGBoost GPU 树模型粗搜配置。"""
    return [
        {
            "name": "baseline",
            "params": {
                "n_estimators": 320,
                "learning_rate": 0.035,
                "max_depth": 6,
                "min_child_weight": 80,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 1.0,
                "reg_alpha": 0.0,
            },
        },
        {
            "name": "shallower_regularized",
            "params": {
                "n_estimators": 420,
                "learning_rate": 0.030,
                "max_depth": 4,
                "min_child_weight": 120,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 3.0,
                "reg_alpha": 0.0,
            },
        },
        {
            "name": "medium_regularized",
            "params": {
                "n_estimators": 500,
                "learning_rate": 0.025,
                "max_depth": 5,
                "min_child_weight": 100,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 3.0,
                "reg_alpha": 0.0,
            },
        },
        {
            "name": "deeper_regularized",
            "params": {
                "n_estimators": 500,
                "learning_rate": 0.025,
                "max_depth": 6,
                "min_child_weight": 120,
                "subsample": 0.80,
                "colsample_bytree": 0.80,
                "reg_lambda": 5.0,
                "reg_alpha": 0.0,
            },
        },
    ]


def runGpuTreeExperiment():
    """比较几组 XGBoost GPU 参数，观察树模型复杂度和正则化方向。"""
    h5dir = os.environ.get("MEOW_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
    outCsv = os.environ.get("MEOW_EXPERIMENT_CSV", os.path.join(os.path.dirname(__file__), "GPU实验结果.csv"))
    ridgeAlpha = float(os.environ.get("MEOW_RIDGE_ALPHA", "100.0"))
    treeRows = int(os.environ.get("MEOW_TREE_ROWS", "800000"))

    trainRange = (20230601, 20231031)
    validRange = (20231101, 20231130)
    testRange = (20231201, 20231229)

    log.inf("Loading train/valid/test feature data for XGBoost GPU experiment...")
    trainX, trainY = _loadFeatureData(h5dir, trainRange[0], trainRange[1])
    validX, validY = _loadFeatureData(h5dir, validRange[0], validRange[1])
    testX, testY = _loadFeatureData(h5dir, testRange[0], testRange[1])

    evaluator = MeowEvaluator()
    rows = []

    for cfg in _gpuTreeConfigs():
        params = cfg["params"]
        log.inf("Running XGBoost GPU experiment: {}".format(cfg["name"]))
        model = MeowModel(
            ridgeAlpha=ridgeAlpha,
            treeRows=treeRows,
            useGpuTreeModel=True,
            gpuTreeParams=params,
        )
        model.fit(trainX, trainY)
        validMetrics, bestWeight = _evaluateModel(model, evaluator, validX, validY, tuneWeights=True)
        testMetrics, _ = _evaluateModel(model, evaluator, testX, testY, tuneWeights=False)

        row = _baseRow("gpu_tree_" + cfg["name"], ridgeAlpha, treeRows, True, bestWeight, validMetrics, testMetrics)
        row.update({
            "gpu_n_estimators": params["n_estimators"],
            "gpu_learning_rate": params["learning_rate"],
            "gpu_max_depth": params["max_depth"],
            "gpu_min_child_weight": params["min_child_weight"],
            "gpu_subsample": params["subsample"],
            "gpu_colsample_bytree": params["colsample_bytree"],
            "gpu_reg_lambda": params["reg_lambda"],
            "gpu_reg_alpha": params["reg_alpha"],
        })
        rows.append(row)

    _writeRows(outCsv, rows)
    log.inf("Experiment results written to {}".format(outCsv))


if __name__ == "__main__":
    experiment = os.environ.get("MEOW_EXPERIMENT", "ridge_alpha")
    if experiment == "gpu_tree":
        runGpuTreeExperiment()
    elif experiment == "cpu_tree":
        runCpuTreeExperiment()
    else:
        runRidgeAlphaExperiment()
