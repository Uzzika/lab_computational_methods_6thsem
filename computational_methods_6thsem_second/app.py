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
        "expected": None
    },
    "example2x2_2": {
        "name": "Пример 2 (2x2) - Повышенная мощность",
        "matrix": [[7, 4], [3, 8]],
        "k": 3,
        "expected": None
    },
    "example3x3_1": {
        "name": "Пример 3 (3x3) из пособия",
        "matrix": [[5, 4, 2], [4, 5, 4], [2, 4, 5]],
        "k": 2,
        "expected": 25
    },
    "example3x3_2": {
        "name": "Пример 4 (3x3) - Асимметричный",
        "matrix": [[6, 3, 5], [2, 7, 4], [4, 5, 3]],
        "k": 2,
        "expected": None
    },
    "example4x4_1": {
        "name": "Пример 5 (4x4) - Симметричный", 
        "matrix": [[3, 5, 2, 4], [4, 2, 5, 3], [5, 3, 4, 2], [2, 4, 3, 5]],
        "k": 3,
        "expected": None
    },
    "example4x4_2": {
        "name": "Пример 6 (4x4) - Циклический",
        "matrix": [[4, 2, 6, 3], [3, 4, 2, 6], [6, 3, 4, 2], [2, 6, 3, 4]],
        "k": 2,
        "expected": None
    },
    "example5x5_1": {
        "name": "Пример 7 (5x5) - Циклический",
        "matrix": [
            [4, 2, 5, 3, 1],
            [2, 5, 3, 1, 4],
            [5, 3, 1, 4, 2],
            [3, 1, 4, 2, 5],
            [1, 4, 2, 5, 3]
        ],
        "k": 2,
        "expected": None
    },
    "example5x5_2": {
        "name": "Пример 8 (5x5) - Разные значения",
        "matrix": [
            [6, 3, 4, 2, 5],
            [4, 5, 2, 6, 3],
            [2, 6, 5, 3, 4],
            [5, 2, 3, 4, 6],
            [3, 4, 6, 5, 2]
        ],
        "k": 3,
        "expected": None
    },
    "example6x6_1": {
        "name": "Пример 9 (6x6) - Сбалансированный",
        "matrix": [
            [5, 3, 4, 6, 2, 1],
            [2, 6, 3, 1, 5, 4],
            [4, 1, 5, 2, 3, 6],
            [3, 5, 1, 4, 6, 2],
            [6, 2, 4, 3, 1, 5],
            [1, 4, 6, 5, 2, 3]
        ],
        "k": 2,
        "expected": None
    },
    "example6x6_2": {
        "name": "Пример 10 (6x6) - Высокая мощность",
        "matrix": [
            [8, 4, 6, 3, 7, 5],
            [5, 7, 3, 8, 4, 6],
            [6, 3, 7, 5, 8, 4],
            [4, 8, 5, 6, 3, 7],
            [7, 5, 4, 8, 6, 3],
            [3, 6, 8, 4, 5, 7]
        ],
        "k": 3,
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
        k = float(data['k'])  # Конвертируем в float
        
        # Проверяем размер матрицы
        if not matrix or any(len(row) != len(matrix) for row in matrix):
            raise ValueError("Матрица должна быть квадратной")
            
        # Проверяем значения матрицы
        for row in matrix:
            for val in row:
                if not isinstance(val, (int, float)) or val < 0:
                    raise ValueError("Все значения матрицы должны быть неотрицательными числами")
        
        # Конвертируем в numpy массив
        matrix = np.array(matrix, dtype=float)
        
        # Замеряем время выполнения
        start_time = datetime.now()
        optimizer = FirepowerOptimizer(matrix.tolist(), k)
        results = optimizer.optimize()
        end_time = datetime.now()
        
        computation_time = (end_time - start_time).total_seconds() * 1000  # в миллисекундах
        
        # Проверка для примера из пособия
        example_key = data.get('example_key')
        if example_key and 'expected' in EXAMPLES.get(example_key, {}):
            expected = EXAMPLES[example_key]['expected']
            if expected and abs(results.get('S7', 0) - expected) > 1e-6:
                app.logger.warning(f"Результат {results.get('S7')} не совпадает с ожидаемым {expected}")
        
        # Создаем визуализации
        matrix_fig = create_matrix_plot(matrix, results['schedule'], k)
        schedule_fig = create_schedule_plot(results['schedule'])
        sigma_plot = create_sigma_plot(results.get('sigma1'), results.get('sigma2'))
        
        # Конвертируем результаты в Python native типы
        response_data = {
            'success': True,
            'results': {
                'schedule': results['schedule'],
                'total_power': float(results['total_power']),
                'initial_power': float(results['initial_power']),
                'computation_time': float(computation_time),
                'efficiency': float(100 * (1 - results['total_power'] / results['initial_power'])),
                'S7': float(results.get('S7', 0)),
                'matrix_size': len(matrix),
                'initial_power_matrix': matrix.tolist()
            },
            'visualizations': {
                'matrix_plot': plotly.io.to_json(matrix_fig),
                'schedule_plot': plotly.io.to_json(schedule_fig),
                'sigma_plot': plotly.io.to_json(sigma_plot)
            }
        }
        
        return jsonify(response_data)
    except Exception as e:
        app.logger.error(f"Ошибка расчета: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def create_matrix_plot(matrix, schedule, k):
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
    
    fig.update_layout(
        title="Оптимальное расписание атак (1 - первая атака, 2 - вторая атака)",
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
        title="График перестановок расписаний",
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