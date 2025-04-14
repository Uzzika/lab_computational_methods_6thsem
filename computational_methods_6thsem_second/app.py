from flask import Flask, render_template, request, jsonify
import numpy as np
from optimizer import FirepowerOptimizer
import plotly
import plotly.graph_objects as go
import json
from datetime import datetime

app = Flask(__name__)

EXAMPLES = {
    "example1": {
        "name": "Пример 1 (3x3) из пособия",
        "matrix": [[5, 4, 2], [4, 5, 4], [2, 4, 5]],
        "k": 2,
        "expected": 25  # Ожидаемое значение S7 из пособия
    },
    "example2": {
        "name": "Пример 2 (4x4)", 
        "matrix": [[3, 5, 2, 4], [4, 2, 5, 3], [5, 3, 4, 2], [2, 4, 3, 5]],
        "k": 3,
        "expected": None
    },
    "example3": {
        "name": "Пример 3 (2x2)",
        "matrix": [[3, 6], [5, 2]],
        "k": 2,
        "expected": None
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
        
        # Проверка для примера из пособия
        example_key = data.get('example_key')
        if example_key and 'expected' in EXAMPLES.get(example_key, {}):
            expected = EXAMPLES[example_key]['expected']
            if expected and abs(results.get('S7', 0) - expected) > 1e-6:
                app.logger.warning(f"Результат {results.get('S7')} не совпадает с ожидаемым {expected}")
        
        # Создаем визуализации
        matrix_fig = create_matrix_plot(matrix, results['schedule'])
        schedule_fig = create_schedule_plot(results['schedule'])
        sigma_plot = create_sigma_plot(results.get('sigma1'), results.get('sigma2'))
        
        return jsonify({
            'success': True,
            'results': {
                'schedule': results['schedule'],
                'total_power': results['total_power'],
                'initial_power': results['initial_power'],
                'computation_time': results['computation_time'],
                'efficiency': 100 * (1 - results['total_power'] / results['initial_power']),
                'S7': results.get('S7', 0)
            },
            'visualizations': {
                'matrix_plot': plotly.io.to_json(matrix_fig),
                'schedule_plot': plotly.io.to_json(schedule_fig),
                'sigma_plot': plotly.io.to_json(sigma_plot)
            }
        })
    except Exception as e:
        app.logger.error(f"Ошибка расчета: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def create_matrix_plot(matrix, schedule):
    """Визуализация матрицы с выделением атакованных целей"""
    n = len(matrix)
    annotations = []
    
    # Подсчитываем количество атак по каждому подразделению
    attack_counts = np.zeros((n, n))
    for j, targets in enumerate(schedule):
        for i in targets:
            attack_counts[i, j] += 1
    
    # Создаем матрицу для отображения с учетом ослабления атакованных целей
    display_matrix = matrix.copy()
    for j in range(n):
        for i in range(n):
            if attack_counts[i, j] > 0:
                # Если подразделение атаковано, его мощность уменьшается в k раз
                # Если атаковано дважды, мощность уменьшается в k раз для каждой атаки
                display_matrix[i, j] = matrix[i, j] / (k ** attack_counts[i, j])
    
    # Создаем текст для подсказок
    hovertext = [[
        f"Отр.{i+1} Пер.{j+1}<br>" +
        f"Исходная мощность: {matrix[i,j]}<br>" +
        f"Текущая мощность: {display_matrix[i,j]:.1f}<br>" +
        f"Коэффициент ослабления: {k}<br>" +
        f"Количество атак: {int(attack_counts[i,j])}<br>" +
        f"Финальный коэффициент ослабления: {k ** attack_counts[i,j]}"
        for j in range(n)] for i in range(n)]
    
    # Создаем цветовую шкалу для отображения мощности
    max_value = matrix.max()
    min_value = display_matrix.min()
    colorscale = [
        [0, 'rgb(255,255,255)'],  # Белый для нулевых значений
        [0.2, 'rgb(200,230,255)'],  # Светло-синий для слабых значений
        [0.4, 'rgb(150,200,255)'],
        [0.6, 'rgb(100,150,255)'],
        [0.8, 'rgb(50,100,255)'],
        [1.0, 'rgb(0,0,255)']  # Синий для максимальных значений
    ]
    
    # Создаем фигуру
    fig = go.Figure()
    
    # Добавляем основную тепловую карту с мощностью
    fig.add_trace(go.Heatmap(
        z=display_matrix,
        colorscale=colorscale,
        x=[f"Пер.{j+1}" for j in range(n)],
        y=[f"Отр.{i+1}" for i in range(n)],
        hoverinfo="text",
        hovertext=hovertext,
        texttemplate="%{z:.1f}",
        textfont={"size": 12, "color": "black"},
        showscale=True,
        colorbar=dict(
            title="Мощность",
            titleside="right"
        )
    ))
    
    # Добавляем маркеры для атакованных целей
    for j in range(n):
        for i in schedule[j]:
            annotations.append(dict(
                x=j, y=i,
                text="⚔️" if attack_counts[i,j] == 1 else "⚔️⚔️",
                showarrow=False,
                font=dict(size=14, color='red')
            ))
    
    # Добавляем линии для разделения периодов
    shapes = []
    for i in range(n+1):
        shapes.append(dict(
            type="line",
            x0=-0.5,
            x1=n-0.5,
            y0=i-0.5,
            y1=i-0.5,
            line=dict(color="black", width=1)
        ))
        shapes.append(dict(
            type="line",
            x0=i-0.5,
            x1=i-0.5,
            y0=-0.5,
            y1=n-0.5,
            line=dict(color="black", width=1)
        ))
    
    # Добавляем рамку для выделения атакованных целей
    for j in range(n):
        for i in schedule[j]:
            shapes.append(dict(
                type="rect",
                x0=j-0.5,
                x1=j+0.5,
                y0=i-0.5,
                y1=i+0.5,
                line=dict(color="red", width=2),
                fillcolor="rgba(255,0,0,0.1)"
            ))
    
    fig.update_layout(
        title="Матрица огневой мощи с учетом ослабления атакованных целей",
        xaxis_title="Периоды времени",
        yaxis_title="Подразделения", 
        annotations=annotations,
        shapes=shapes,
        width=600,
        height=600,
        margin=dict(l=60, r=30, t=80, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_schedule_plot(schedule):
    """Визуализация расписания атак"""
    n = len(schedule)
    attack_matrix = np.zeros((n, n))
    
    # Создаем матрицу атак с дополнительной информацией
    for j in range(n):
        for i in schedule[j]:
            attack_matrix[i, j] += 1
    
    # Создаем текст для подсказок с информацией о последовательности атак
    hovertext = [[
        f"Отр.{i+1} Пер.{j+1}<br>" +
        f"{'Не атаковано' if attack_matrix[i,j] == 0 else f'Атаковано {int(attack_matrix[i,j])} раз'}<br>" +
        f"{'Первая атака' if attack_matrix[i,j] == 1 else 'Вторая атака' if attack_matrix[i,j] == 2 else ''}"
        for j in range(n)] for i in range(n)]
    
    fig = go.Figure(data=go.Heatmap(
        z=attack_matrix,
        colorscale=[[0, 'white'], [0.5, 'lightcoral'], [1, 'red']],
        x=[f"Пер.{j+1}" for j in range(n)],
        y=[f"Отр.{i+1}" for i in range(n)],
        hoverinfo="text",
        hovertext=hovertext,
        text=[["" if val == 0 else "1" if val == 1 else "2" for val in row] for row in attack_matrix],
        texttemplate="%{text}",
        textfont={"size": 14, "color": "black"}
    ))
    
    fig.update_layout(
        title="Оптимальное расписание атак (1 - первая атака, 2 - вторая атака)",
        xaxis_title="Периоды времени",
        yaxis_title="Подразделения",
        width=600,
        height=600,
        margin=dict(l=60, r=30, t=80, b=60)
    )
    
    return fig

def create_sigma_plot(sigma1, sigma2):
    """Визуализация перестановок σ1 и σ2"""
    if sigma1 is None or sigma2 is None:
        return go.Figure()
    
    n = len(sigma1)
    fig = go.Figure()
    
    # Добавляем перестановку σ1
    fig.add_trace(go.Scatter(
        x=list(range(1, n+1)),
        y=[x+1 for x in sigma1],
        mode='lines+markers',
        name='σ1 (первое расписание)',
        line=dict(color='blue', width=2),
        marker=dict(size=10)
    ))
    
    # Добавляем перестановку σ2
    fig.add_trace(go.Scatter(
        x=list(range(1, n+1)),
        y=[x+1 for x in sigma2],
        mode='lines+markers',
        name='σ2 (второе расписание)',
        line=dict(color='red', width=2),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title="График перестановок расписаний",
        xaxis_title="Период",
        yaxis_title="Номер подразделения",
        width=600,
        height=400,
        margin=dict(l=60, r=30, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True)