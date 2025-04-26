import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# === 配置区 ===
data_base = 'D:/quantum/quantum_t_data/quantum_t_data'
data_path = data_base + "/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1_with_predictions_120to80_2LSTM_future1.pkl"

init_start = 1000000  # 初始起始行号
window = 200          # 窗口长度

# 读取并截取后 10% 数据
df = pd.read_pickle(data_path)
split_index = int(len(df) * 0.9)
df = df.iloc[split_index:]
max_start = len(df) - window
if init_start < 0 or init_start > max_start:
    init_start = 0

def get_segment(start):
    seg = df.iloc[start:start+window]
    # x 轴
    if "datetime" in seg.columns:
        x = pd.to_datetime(seg["datetime"])
    else:
        x = seg.index
    return x, seg["close"].values, seg["evaluation_120"].values, seg["prediction3"].values

# 初始数据
x, y_close, y_eval, y_pred = get_segment(init_start)

# 创建图形和两个子图（上下排列），共用 x 轴
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.25, hspace=0.1)

# 子图 1：close
line_close, = ax1.plot(x, y_close, label="close")
ax1.set_ylabel("Close")
ax1.legend(loc="upper left")

# 子图 2：evaluation & prediction
line_eval, = ax2.plot(x, y_eval, label="evaluation")
line_pred, = ax2.plot(x, y_pred, label="prediction")
ax2.set_xlabel("时间" if "datetime" in df.columns else "Index")
ax2.set_ylabel("值")
ax2.legend(loc="upper left")

# 自动调整 y 轴上下限
def update_limits(ax, y1, y2=None):
    y_min = y1.min() if y2 is None else min(y1.min(), y2.min())
    y_max = y1.max() if y2 is None else max(y1.max(), y2.max())
    ax.set_ylim(y_min * 0.9, y_max * 1.1)

ax1.set_ylim(y_close.min(), y_close.max())
update_limits(ax2, y_eval, y_pred)

# 滑块
ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
slider = Slider(ax=ax_slider, label='Start Index',
                valmin=0, valmax=max_start, valinit=init_start, valstep=1)

# 滑块事件
def on_slider_change(val):
    start = int(slider.val)
    x_new, close_new, eval_new, pred_new = get_segment(start)

    # 更新 close 曲线
    line_close.set_data(x_new, close_new)
    # 将 ax1 的 y 轴下限设为 close_new.min(), 上限设为 close_new.max()
    ax1.set_ylim(close_new.min(), close_new.max())

    # 更新 evaluation & prediction
    line_eval.set_data(x_new, eval_new)
    line_pred.set_data(x_new, pred_new)
    update_limits(ax2, eval_new, pred_new)

    # 其余保持不变……
    ax1.set_xlim(x_new.min(), x_new.max())
    fig.suptitle(f"窗口行 {start} 到 {start + window - 1}")
    fig.canvas.draw_idle()

slider.on_changed(on_slider_change)

# 初始标题
fig.suptitle(f"窗口行 {init_start} 到 {init_start + window - 1}")
plt.show()