from flask import Flask, render_template, request, jsonify
import numpy as np
from optimizer import FirepowerOptimizer
import plotly
import plotly.graph_objects as go
import time

app = Flask(__name__)

EXAMPLES = {
    "example1": {
        "name": "Пример 1 (3x3)",
        "matrix": [[5, 4, 2], [4, 5, 4], [2, 4, 5]],
        "k": 2
    },
    "example2": {
        "name": "Пример 2 (4x4)",
        "matrix": [[3, 5, 2, 4], [4, 2, 5, 3], [5, 3, 4, 2], [2, 4, 3, 5]],
        "k": 3
    },
    "example3": {
        "name": "Пример 3 (2x2)",
        "matrix": [[3, 6], [5, 2]],
        "k": 2
    }
}

@app.route('/')
def index():
    return render_template('index.html', examples=EXAMPLES)

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.get_json()
        matrix = data['matrix']
        k = int(data['k'])
        
        # Конвертируем в numpy массив
        matrix = np.array(matrix, dtype=int)
        
        optimizer = FirepowerOptimizer(matrix.tolist(), k)
        results = optimizer.optimize()
        
        # Создаем визуализации
        matrix_fig = create_matrix_plot(matrix, results['schedule'])
        schedule_fig = create_schedule_plot(results['schedule'])
        
        return jsonify({
            'success': True,
            'results': {
                'schedule': results['schedule'],
                'total_power': results['total_power'],
                'initial_power': results['initial_power'],
                'computation_time': optimizer.computation_time
            },
            'matrix_plot': plotly.io.to_json(matrix_fig),
            'schedule_plot': plotly.io.to_json(schedule_fig)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def create_matrix_plot(matrix, schedule):
    n = len(matrix)
    # Создаем аннотации для выделения выбранных целей
    annotations = []
    shapes = []
    
    for j in range(n):
        for i in schedule[j]:
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text="🔴",  # Красный кружок для атакованных целей
                    showarrow=False,
                    font=dict(size=14, color='red')
                ))
            shapes.append(
                dict(
                    type="rect",
                    x0=j-0.5, x1=j+0.5,
                    y0=i-0.5, y1=i+0.5,
                    line=dict(color="red", width=2),
                    fillcolor="rgba(255,0,0,0.1)"
                ))
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        colorscale='YlOrRd',
        x=[f"Пер.{j+1}" for j in range(n)],
        y=[f"Отр.{i+1}" for i in range(n)],
        hoverinfo="text",
        hovertext=[[f"Отр.{i+1} Пер.{j+1}\nМощность: {matrix[i][j]}" 
                  for j in range(n)] for i in range(n)],
        text=matrix,
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    
    fig.update_layout(
        title="Матрица огневой мощи (🔴 - атакованные цели)",
        xaxis_title="Периоды времени",
        yaxis_title="Подразделения", 
        annotations=annotations,
        shapes=shapes,
        width=600,
        height=600,
        margin=dict(l=60, r=30, t=80, b=60)
    )
    
    return fig

def create_schedule_plot(schedule):
    n = len(schedule)
    # Преобразуем расписание в матрицу для визуализации
    data = []
    for i in range(n):
        row = []
        for j in range(n):
            if i in schedule[j]:
                # 1 - если подразделение i атаковано в период j
                row.append(1)
            else:
                row.append(0)
        data.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale=[[0, 'white'], [1, 'red']],
        x=[f"Пер.{j+1}" for j in range(n)],
        y=[f"Отр.{i+1}" for i in range(n)],
        hoverinfo="text",
        hovertext=[[f"Отр.{i+1} {'атаковано' if data[i][j] else 'не атаковано'} в Пер.{j+1}" 
                  for j in range(n)] for i in range(n)],
        text=[["🔴" if val else "" for val in row] for row in data],
        texttemplate="%{text}",
        textfont={"size": 16}
    ))
    
    fig.update_layout(
        title="Расписание атак (2 удара за период)",
        xaxis_title="Периоды времени",
        yaxis_title="Подразделения",
        width=600,
        height=600,
        margin=dict(l=60, r=30, t=80, b=60)
    )
    
    return fig
    
if __name__ == '__main__':
    app.run(debug=True)