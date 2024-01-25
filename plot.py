import plotly.graph_objects as go
import plotly.io as pio
pio.kaleido.scope.mathjax = None

# 数据
categories = [10, 20, 30, 50]

data = {
    'ICEWS14': [0.339, 0.343, 0.339, 0.338],
    'ICEWS05-15': [0.383, 0.390, 0.391, 0.393],
    'ICEWS18': [0.209, 0.218, 0.220, 0.225],
    'YAGO': [0.526, 0.527, 0.527, 0.527]
}

# 创建折线图
fig = go.Figure()
fig.update_yaxes(range=[0.18, 0.57])
colors = [(230, 111, 81), (42, 157, 142), (38, 70, 83), (243, 162, 97)]
# 添加数据折线，分别设置颜色和文本显示在最上方
for i, (category, values) in enumerate(data.items()):
    line_color = f'rgb{colors[i]}'
    fig.add_trace(go.Scatter(x=categories, y=values, mode='lines+markers', name=category, text=values, textposition='top center', line=dict(color=line_color)))

    # 添加文本，位置在最上方
    for x, y, text in zip(categories, values, values):
        fig.add_annotation(
            x=x, y=y+0.008,
            text=str(text),
            showarrow=False,
            font=dict(family='Helvetica, sans-serif', size=14, color='black'),
            xshift=5,
            yshift=5
        )
# 添加数据折线，分别设置颜色
for k, v in data.items():
    fig.add_shape(
        type="line",
        x0=5,
        y0=v[0],
        x1=55,
        y1=v[0],
        line=dict(color="lightgray", dash="dot"),
        layer='below'  # 设置图层为最下层
    )
for x_value in categories:
    fig.add_shape(
        type="line",
        x0=x_value,
        y0=0.18,
        x1=x_value,
        y1=0.57,
        line=dict(color="lightgray", dash="dot"),
        layer='below'  # 设置图层为最下层
    )


# 设置布局
fig.update_layout(
    title=None,
    xaxis=dict(title='History Length', showline=True, linewidth=1, linecolor='black', mirror=True),
    yaxis=dict(title='Hits@1', showline=True, linewidth=1, linecolor='black', mirror=True, title_standoff=5),
    legend=dict(title='Dataset'),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Helvetica, sans-serif', size=14, color='black'),
)

# 显示图表
fig.write_image("line_chart.pdf")
