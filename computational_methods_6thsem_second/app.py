from flask import Flask, render_template, request, jsonify
import json
import numpy as np
from optimizer import FirepowerOptimizer
import plotly
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)

# Примеры матриц
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
        
        # Конвертируем numpy типы в native Python типы
        matrix = [[int(val) for val in row] for row in matrix]
        
        optimizer = FirepowerOptimizer(matrix, k)
        results = optimizer.optimize()
        
        # Конвертируем результаты в сериализуемый формат
        results_serializable = {
            'schedule': results['schedule'],
            'total_power': float(results['total_power']),
            'initial_power': int(results['initial_power']),
            'computation_time': results['computation_time']
        }
        
        matrix_fig = create_matrix_plot(matrix, results['schedule'])
        schedule_fig = create_schedule_plot(results['schedule'])
        
        return jsonify({
            'success': True,
            'results': results_serializable,
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
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        colorscale='YlOrRd',
        x=[f"Пер. {j+1}" for j in range(n)],
        y=[f"Отр. {i+1}" for i in range(n)],
        text=matrix,
        texttemplate="%{text}",
        textfont={"size": 12}
    ))
    
    # Добавляем прямоугольники для выделения выбранных элементов
    for j in range(n):
        for i in schedule[j]:
            fig.add_shape(
                type="rect",
                x0=j-0.5, x1=j+0.5,
                y0=i-0.5, y1=i+0.5,
                line=dict(color="RoyalBlue", width=3),
                fillcolor="rgba(0,0,0,0)"
            )
    
    fig.update_layout(
        title="Матрица огневой мощи",
        xaxis_title="Период",
        yaxis_title="Отряд",
        width=600,
        height=600
    )
    
    return fig

def create_schedule_plot(schedule):
    n = len(schedule)
    periods = list(range(1, n+1))
    units = list(range(1, n+1))
    
    # Создаем матрицу для визуализации
    data = np.zeros((n, n))
    for j in range(n):
        for i in schedule[j]:
            data[i][j] = 1
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale=[[0, 'white'], [1, 'red']],
        x=periods,
        y=units,
        showscale=False
    ))
    
    fig.update_layout(
        title="Расписание атак (2 выстрела за период)",
        xaxis_title="Период",
        yaxis_title="Отряд",
        width=600,
        height=600
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True)