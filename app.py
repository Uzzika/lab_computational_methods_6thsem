from flask import Flask, render_template, request, jsonify
import numpy as np
from optimizer import FirepowerOptimizer
import plotly
import plotly.graph_objects as go
import json
from datetime import datetime

app = Flask(__name__)

EXAMPLES = {
    "example2x2_1": {
        "name": "Пример 1 (2x2) - Базовый",
        "matrix": [[3, 6], [5, 2]],
        "k": 2,
        "expected": None,
        "task_type": 1
    },
    "example2x2_2": {
        "name": "Пример 2 (2x2) - Повышенная мощность",
        "matrix": [[7, 4], [3, 8]],
        "k": 3,
        "expected": None,
        "task_type": 1
    },
    "example3x3_1": {
        "name": "Пример 3 (3x3) из пособия (Задача 1)",
        "matrix": [[5, 4, 2], [4, 5, 4], [2, 4, 5]],
        "k": 2,
        "expected": 15,
        "task_type": 1
    },
    "example3x3_2": {
        "name": "Пример 4 (3x3) - Асимметричный (Задача 1)",
        "matrix": [[6, 3, 5], [2, 7, 4], [4, 5, 3]],
        "k": 2,
        "expected": None,
        "task_type": 1
    },
    "example3x3_3": {
        "name": "Пример 5 (3x3) - Задача 2",
        "matrix": [[5, 4, 2], [4, 5, 4], [2, 4, 5]],
        "k": 2,
        "expected": None,
        "task_type": 2
    },
    "example4x4_1": {
        "name": "Пример 6 (4x4) - Симметричный (Задача 1)", 
        "matrix": [[3, 5, 2, 4], [4, 2, 5, 3], [5, 3, 4, 2], [2, 4, 3, 5]],
        "k": 3,
        "expected": None,
        "task_type": 1
    },
    "example4x4_2": {
        "name": "Пример 7 (4x4) - Циклический (Задача 2)",
        "matrix": [[4, 2, 6, 3], [3, 4, 2, 6], [6, 3, 4, 2], [2, 6, 3, 4]],
        "k": 2,
        "expected": None,
        "task_type": 2
    },
    "example5x5_1": {
        "name": "Пример 8 (5x5) - Циклический (Задача 1)",
        "matrix": [
            [4, 2, 5, 3, 1],
            [2, 5, 3, 1, 4],
            [5, 3, 1, 4, 2],
            [3, 1, 4, 2, 5],
            [1, 4, 2, 5, 3]
        ],
        "k": 2,
        "expected": None,
        "task_type": 1
    },
    "example5x5_2": {
        "name": "Пример 9 (5x5) - Разные значения (Задача 3)",
        "matrix": [
            [6, 3, 4, 2, 5],
            [4, 5, 2, 6, 3],
            [2, 6, 5, 3, 4],
            [5, 2, 3, 4, 6],
            [3, 4, 6, 5, 2]
        ],
        "k": 3,
        "expected": None,
        "task_type": 3
    },
    "example6x6_1": {
        "name": "Пример 10 (6x6) - Сбалансированный (Задача 1)",
        "matrix": [
            [5, 3, 4, 6, 2, 1],
            [2, 6, 3, 1, 5, 4],
            [4, 1, 5, 2, 3, 6],
            [3, 5, 1, 4, 6, 2],
            [6, 2, 4, 3, 1, 5],
            [1, 4, 6, 5, 2, 3]
        ],
        "k": 2,
        "expected": None,
        "task_type": 1
    },
    "example6x6_2": {
        "name": "Пример 11 (6x6) - Высокая мощность (Задача 2)",
        "matrix": [
            [8, 4, 6, 3, 7, 5],
            [5, 7, 3, 8, 4, 6],
            [6, 3, 7, 5, 8, 4],
            [4, 8, 5, 6, 3, 7],
            [7, 5, 4, 8, 6, 3],
            [3, 6, 8, 4, 5, 7]
        ],
        "k": 3,
        "expected": None,
        "task_type": 2
    },
    "example10x10": {
        "name": "Пример 12 (10x10) - Большая матрица",
        "matrix": [
            [5, 3, 7, 2, 8, 4, 6, 1, 9, 2],
            [2, 8, 4, 6, 1, 9, 3, 7, 5, 3],
            [7, 5, 9, 3, 6, 2, 8, 4, 1, 6],
            [4, 6, 2, 8, 5, 7, 1, 9, 3, 4],
            [9, 1, 6, 4, 7, 3, 5, 2, 8, 5],
            [3, 7, 1, 9, 4, 8, 2, 6, 4, 7],
            [8, 4, 5, 1, 9, 6, 7, 3, 2, 8],
            [6, 2, 8, 5, 3, 1, 9, 5, 7, 1],
            [1, 9, 3, 7, 2, 5, 4, 8, 6, 9],
            [5, 3, 7, 2, 8, 4, 6, 1, 9, 2]
        ],
        "k": 2,
        "expected": None,
        "task_type": 1
    }
}

@app.route('/')
def index():
    return render_template('index.html', examples=EXAMPLES)

def mode_to_task_type(mode):
    if mode == 'task1':
        return 1
    elif mode == 'task2':
        return 2
    else:
        return 3

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.get_json()
        matrix = data['matrix']
        k = float(data['k'])
        mode = data.get('mode', 'task3')
        strategy = data.get('strategy', 'optimal')

        if not matrix or any(len(row) != len(matrix) for row in matrix):
            raise ValueError("Матрица должна быть квадратной")

        for row in matrix:
            for val in row:
                if not isinstance(val, (int, float)) or val < 0:
                    raise ValueError("Все значения матрицы должны быть неотрицательными числами")

        matrix = np.array(matrix, dtype=float)

        start_time = datetime.now()
        optimizer = FirepowerOptimizer(matrix.tolist(), k, mode=mode, strategy=strategy)
        results = optimizer.optimize()
        end_time = datetime.now()
        computation_time = (end_time - start_time).total_seconds() * 1000

        task_type = mode_to_task_type(mode)

        # визуализация
        matrix_fig = create_matrix_plot(matrix, results['schedule'], k, task_type)
        schedule_fig = create_schedule_plot(results['schedule'], task_type)
        sigma_plot = create_sigma_plot(results.get('sigma1'), results.get('sigma2'))

        response_data = {
            'success': True,
            'results': {
                'schedule': results.get('schedule', []),
                'total_power': float(results.get('total_power', 0)),
                'initial_power': float(results.get('initial_power', 1)),
                'computation_time': float(results.get('computation_time', 0)),
                'efficiency': float(100 * (1 - results['total_power'] / results['initial_power'])),
                'S7': float(results.get('S7', 0)),
                'matrix_size': len(matrix),
                'initial_power_matrix': matrix.tolist(),
                'strategy': results.get('strategy', 'optimal'),
                'strategy': strategy  # Добавляем эту строку !!!
            },
            'visualizations': {
                'matrix_plot': plotly.io.to_json(matrix_fig),
                'schedule_plot': plotly.io.to_json(schedule_fig),
                'sigma_plot': plotly.io.to_json(sigma_plot)
            }
        }
        response_data['results']['strategy'] = strategy # !!!
        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Ошибка расчета: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

def create_matrix_plot(matrix, schedule, k, task_type):
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
                display_matrix[i, j] = matrix[i, j] / (k ** attack_counts[i, j])
    
    # Создаем текст для подсказок
    hovertext = [[
        f"Отр.{i+1} Пер.{j+1}<br>" +
        f"Исходная мощность: {matrix[i,j]}<br>" +
        f"Текущая мощность: {display_matrix[i,j]:.1f}<br>" +
        f"Коэффициент ослабления: {k}<br>" +
        f"Количество атак: {int(attack_counts[i,j])}<br>" +
        f"Финальный коэффициент ослабления: {k ** attack_counts[i,j]:.1f}<br>" +
        f"Процент ослабления: {100 * (1 - display_matrix[i,j]/matrix[i,j]):.1f}%"
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
            x=1.02
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
    
    # Добавляем заголовок в зависимости от типа задачи
    if task_type == 1:
        title = "Матрица огневой мощи (Задача 1) - один удар за период"
    elif task_type == 2:
        title = "Матрица огневой мощи (Задача 2) - эффект на 2 периода"
    else:
        title = "Матрица огневой мощи (Задача 3) - два удара за период"
    
    fig.update_layout(
        title=title,
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

def create_schedule_plot(schedule, task_type):
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
    
    # Создаем фигуру
    fig = go.Figure()
    
    # Добавляем тепловую карту атак
    fig.add_trace(go.Heatmap(
        z=attack_matrix,
        colorscale=[[0, 'white'], [0.5, 'lightcoral'], [1, 'red']],
        x=[f"Пер.{j+1}" for j in range(n)],
        y=[f"Отр.{i+1}" for i in range(n)],
        hoverinfo="text",
        hovertext=hovertext,
        text=[["" if val == 0 else "1" if val == 1 else "2" for val in row] for row in attack_matrix],
        texttemplate="%{text}",
        textfont={"size": 14, "color": "black"},
        showscale=False
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
    
    # Добавляем рамки для выделения атакованных целей
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
    
    # Добавляем заголовок в зависимости от типа задачи
    if task_type == 1:
        title = "Расписание атак (Задача 1) - один удар за период"
    elif task_type == 2:
        title = "Расписание атак (Задача 2) - эффект на 2 периода"
    else:
        title = "Расписание атак (Задача 3) - два удара за период (1 - первая атака, 2 - вторая атака)"
    
    fig.update_layout(
        title=title,
        xaxis_title="Периоды времени",
        yaxis_title="Подразделения",
        shapes=shapes,
        width=600,
        height=600,
        margin=dict(l=60, r=30, t=80, b=60),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

def create_sigma_plot(sigma1, sigma2):
    """Визуализация перестановок σ1 и σ2"""
    if sigma1 is None or sigma2 is None:
        return go.Figure()
    
    n = len(sigma1)
    fig = go.Figure()
    
    # Добавляем перестановку σ1 (первое расписание)
    fig.add_trace(go.Scatter(
        x=list(range(1, n+1)),
        y=[x+1 for x in sigma1],
        mode='lines+markers+text',
        name='σ1 (первое расписание)',
        line=dict(color='blue', width=2),
        marker=dict(size=10, color='blue'),
        text=[f"σ1({i+1})" for i in range(n)],
        textposition="top center",
        textfont=dict(size=12, color='blue')
    ))
    
    # Добавляем перестановку σ2 (второе расписание)
    fig.add_trace(go.Scatter(
        x=list(range(1, n+1)),
        y=[x+1 for x in sigma2],
        mode='lines+markers+text',
        name='σ2 (второе расписание)',
        line=dict(color='red', width=2),
        marker=dict(size=10, color='red'),
        text=[f"σ2({i+1})" for i in range(n)],
        textposition="bottom center",
        textfont=dict(size=12, color='red')
    ))
    
    # Добавляем точки пересечения
    intersections = []
    for i in range(n):
        if sigma1[i] == sigma2[i]:
            intersections.append({
                'x': i+1,
                'y': sigma1[i]+1,
                'text': f"({i+1}, {sigma1[i]+1})"
            })
    
    if intersections:
        fig.add_trace(go.Scatter(
            x=[p['x'] for p in intersections],
            y=[p['y'] for p in intersections],
            mode='markers+text',
            name='Точки пересечения',
            marker=dict(size=15, color='green', symbol='star'),
            text=[p['text'] for p in intersections],
            textposition="top center",
            textfont=dict(size=12, color='green')
        ))
    
    fig.update_layout(
        title="График перестановок расписаний (Задача 3)",
        xaxis_title="Период",
        yaxis_title="Номер подразделения",
        width=600,
        height=400,
        margin=dict(l=60, r=30, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            range=[0.5, n+0.5]
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            range=[0.5, n+0.5]
        )
    )
    
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=5002) 