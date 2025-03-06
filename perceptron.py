import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron

# Данные (изначально пустые)
X = []
y = []

# Создаём модель перцептрона
perceptron = Perceptron(max_iter=1000, tol=1e-3, random_state=42)

# Функция для обновления графика
def update_plot():
    plt.clf()  # Очистка графика
    plt.title("Кликните, чтобы добавить точки (ЛКМ - класс 1, ПКМ - класс 0)")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")

    # Если есть точки, рисуем их
    if len(X) > 0:
        X_np = np.array(X)
        y_np = np.array(y)

        plt.scatter(X_np[y_np == 1][:, 0], X_np[y_np == 1][:, 1], color="red", label="Класс 1")
        plt.scatter(X_np[y_np == 0][:, 0], X_np[y_np == 0][:, 1], color="blue", label="Класс 0")

        # Обучаем перцептрон, если есть хотя бы два разных класса
        if len(set(y)) > 1:
            perceptron.fit(X_np, y_np)

            # Рисуем разделяющую линию
            x_values = np.linspace(-10, 10, 100)
            w = perceptron.coef_[0]
            b = perceptron.intercept_[0]
            y_values = -(w[0] * x_values + b) / w[1]
            plt.plot(x_values, y_values, 'g-', linewidth=2, label="Разделяющая линия")

    plt.legend()
    plt.draw()

# Функция для обработки кликов
def on_click(event):
    if event.xdata is None or event.ydata is None:
        return  # Выход, если клик вне области графика

    class_label = 1 if event.button == 1 else 0  # ЛКМ - класс 1, ПКМ - класс 0
    X.append([event.xdata, event.ydata])
    y.append(class_label)

    update_plot()  # Перерисовываем график

# Создаём интерактивный график
fig = plt.figure()
fig.canvas.mpl_connect("button_press_event", on_click)

update_plot()  # Первоначальный пустой график
plt.show()
