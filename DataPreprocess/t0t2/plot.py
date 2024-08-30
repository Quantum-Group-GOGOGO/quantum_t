import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# 数据读取
data_base='/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
T2_data_path=data_base+'/type2/Nasdaq_qqq_align_base.pkl'
df = pd.read_pickle(T2_data_path)

# df切片 *optional
#df = df.head(1000)

print(df.tail())

# 初始化 Dash 应用
app = Dash(__name__)

# 定义布局
app.layout = html.Div([
    dcc.Graph(id='line-chart'),
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
    Output('line-chart', 'figure'),
    [Input('slider', 'value')]
)
def update_graph(selected_index):
    # 根据滑动条位置选择数据窗口
    selected_data = df.iloc[selected_index:selected_index + 200]  # 提取10000个点

    # 使用 WebGL 渲染器创建折线图
    fig = go.Figure(data=[
        go.Scattergl(x=selected_data['datetime'], y=selected_data['open'], mode='lines', name='Open')
    ])

    fig.update_layout(
        title='Open Value Over Time',
        xaxis_title='Datetime',
        yaxis_title='Open Value'
    )

    return fig

# 运行应用
if __name__ == '__main__':
    app.run_server(debug=True)