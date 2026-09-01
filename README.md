# 合成城市需求时序预测

一个可复现的小时级需求预测示例，包含数据生成、严格时间回测、递归多步预测和 Streamlit 诊断看板。数据为合成数据，不代表任何真实城市或业务。

## 快速开始

```bash
python -m pip install -r requirements.txt
python generate_data.py
python train.py --model linear
python -m pytest
streamlit run app.py
```

浏览器打开 Streamlit 输出的本地地址。数据缺失时训练脚本和看板会自动生成 `data/demand.csv`。

## 文件说明

- `generate_data.py`：固定种子生成小时级温度、降水、周末标记和需求。
- `train.py`：特征工程、模型训练、递归预测和扩展窗口回测；支持 `linear` 与 `random_forest`。
- `app.py`：选择模型、训练窗口、回测窗口和预测时长，查看预测曲线、误差趋势和下载 CSV。
- `tests/test_core.py`：覆盖可复现性、特征无泄漏、递归预测和指标边界。

## 看板用法

启动后在左侧选择模型、初始训练天数、回测窗口和预测时长。页面会显示数据范围、未来预测曲线、预测明细 CSV 下载，以及按窗口展开的 MAE/RMSE/MAPE/SMAPE 诊断。随机森林回测较慢，首次运行请等待模型计算完成。

## 评估口径

每个回测窗口只用窗口起点之前的观测需求训练。测试窗口的第一步使用训练末尾历史滞后，后续步骤使用模型自己的预测递归生成 `lag_1`、`lag_24` 和滚动均值，因此不会读取测试期真实 `demand`。天气在演示预测中沿用最后观测值；生产场景应替换为天气预报。

MAE 和 RMSE 保持需求原单位；MAPE、SMAPE 为百分比。MAPE 对零值使用极小分母保护，SMAPE 更适合低需求区间。`train.py` 会将窗口明细写入 `data/rolling_metrics.csv`。

## 边界与下一步

这是教学和作品集基线，不含节假日、空间维度、实时数据、模型注册或生产服务。进一步升级可加入真实天气、节假日特征、预测区间和模型版本追踪。
