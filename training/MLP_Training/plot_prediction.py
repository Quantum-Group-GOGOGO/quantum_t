#!/usr/bin/env python3
# visualize_predictions_with_slider.py

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# === 配置区: 在这里直接定义参数 ===
# 数据根目录和文件路径
data_base = 'D:/quantum/quantum_t_data/quantum_t_data'
data_path = data_base + '/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1_with_predictions.pkl'
# 滑动窗口默认起始行号和窗口长度
init_start = 1000000  # 初始起始行号
window = 200          # 窗口长度

# 读取 DataFrame
df = pd.read_pickle(data_path)
split_index = int(len(df) * 0.9)
df = df.iloc[split_index:]
# 最大可滑动起始位置，确保窗口不越界
max_start = len(df) - window
if init_start < 0 or init_start > max_start:
    init_start = 0

# 初始化窗口数据
def get_segment(start):
    seg = df.iloc[start:start+window]
    if "datetime" in seg.columns:
        x = pd.to_datetime(seg["datetime"])
    else:
        x = seg.index
    y_eval = seg["evaluation_30"].values
    y_pred = seg["prediction1"].values
    return x, y_eval, y_pred

x, y_eval, y_pred = get_segment(init_start)

# 创建图形和主坐标轴
graph_fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, bottom=0.25)

# 绘制初始曲线
line_eval, = ax.plot(x, y_eval, label="evaluation")
line_pred, = ax.plot(x, y_pred, label="prediction1")
ax.set_xlabel("时间" if "datetime" in df.columns else "Index")
ax.set_ylabel("值")
ax.set_title(f"Evaluation vs Prediction1（行 {init_start} 到 {init_start + window - 1}）")
ax.legend()

# 设置 x, y 轴范围
def update_limits(x_vals, y1, y2):
    ax.set_xlim(x_vals.min(), x_vals.max())
    y_min = min(y1.min(), y2.min())
    y_max = max(y1.max(), y2.max())
    ax.set_ylim(y_min * 0.9, y_max * 1.1)

update_limits(x, y_eval, y_pred)

# 增加 Slider 控件，放在底部
ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
slider = Slider(
    ax=ax_slider,
    label='Start Index',
    valmin=0,
    valmax=max_start,
    valinit=init_start,
    valstep=1
)

# Slider 事件响应函数
def on_slider_change(val):
    start = int(slider.val)
    x_new, y_eval_new, y_pred_new = get_segment(start)
    # 更新数据
    line_eval.set_data(x_new, y_eval_new)
    line_pred.set_data(x_new, y_pred_new)
    # 更新标题和坐标范围
    ax.set_title(f"Evaluation vs Prediction1（行 {start} 到 {start + window - 1}）")
    update_limits(x_new, y_eval_new, y_pred_new)
    graph_fig.canvas.draw_idle()

slider.on_changed(on_slider_change)

plt.show()
