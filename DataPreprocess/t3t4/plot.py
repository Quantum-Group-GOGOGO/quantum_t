import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# 数据读取
data_base='/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
T4_data_path=data_base+'/type4/Nasdaq_qqq_align_labeled_base_evaluated_300.pkl'
df = pd.read_pickle(T4_data_path)

# df切片 *optional
#df = df.head(1000)

print(df.tail())

# 初始化 Dash 应用
app = Dash(__name__)

# 定义布局
app.layout = html.Div([
    dcc.Graph(id='line-chart-evaluation'),
    dcc.Graph(id='line-chart-close'),
    dcc.Slider(
        id='slider',
        min=0,
        max=len(df) - 200,
        value=0,
        marks={i: str(df['datetime'].iloc[i].strftime('%Y-%m-%d %H:%M')) for i in range(0, len(df), len(df)//10)},
        step=100,
    )
])

# 定义回调函数
@app.callback(
    [Output('line-chart-evaluation', 'figure'),
     Output('line-chart-close', 'figure')],
    [Input('slider', 'value')]
)
def update_graph(selected_index):
    # 根据滑动条位置选择数据窗口
    selected_data = df.iloc[selected_index:selected_index + 200]  # 提取200个点

    # 创建evaluation列的折线图
    fig_evaluation = go.Figure(data=[
        go.Scattergl(x=selected_data['datetime'], y=selected_data['evaluation_300'], mode='lines', name='Evaluation')
    ])

    fig_evaluation.update_layout(
        title='Evaluation Over Time',
        xaxis_title='Datetime',
        yaxis_title='Evaluation Value'
    )

    # 创建close列的折线图
    fig_close = go.Figure(data=[
        go.Scattergl(x=selected_data['datetime'], y=selected_data['close'], mode='lines', name='Close')
    ])

    fig_close.update_layout(
        title='Close Value Over Time',
        xaxis_title='Datetime',
        yaxis_title='Close Value'
    )

    return fig_evaluation, fig_close

# 运行应用
if __name__ == '__main__':
    app.run_server(debug=True)