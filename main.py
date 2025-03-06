import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Данные
X = []
y = []

'''hidden_layer_sizes=(10, 5) — количество нейронов в скрытых слоях. В данном случае, 10 нейронов в первом скрытом слое и 5 нейронов во втором.
activation='relu' — функция активации ReLU (Rectified Linear Unit), которая используется для того, чтобы нейронная сеть могла моделировать сложные нелинейности.
max_iter=5000 — максимальное количество итераций для обучения модели.
learning_rate_init=0.01 — начальная скорость обучения для нейросети.
random_state=42 — фиксирует случайность для воспроизводимости результатов.'''
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='relu', max_iter=5000, learning_rate_init=0.01, random_state=42)



# Функция для обновления графика
def update_plot():
    plt.clf()
    plt.title("Кликните, чтобы добавить точки (ЛКМ - класс 1, ПКМ - класс 0)")
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")

    # Рисуем точки, если они есть
    if len(X) > 0:
        X_np = np.array(X)
        y_np = np.array(y)

        plt.scatter(X_np[y_np == 1][:, 0], X_np[y_np == 1][:, 1], color="red", label="Класс 1")
        plt.scatter(X_np[y_np == 0][:, 0], X_np[y_np == 0][:, 1], color="blue", label="Класс 0")

        # Обучаем нейросеть, если есть два разных класса
        if len(set(y)) > 1:
            mlp.fit(X_np, y_np)

            # Создаём сетку точек для построения границы решений
            xx, yy = np.meshgrid(np.linspace(-10, 10, 200), np.linspace(-10, 10, 200))
            Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # Рисуем разделяющие линии
            plt.contour(xx, yy, Z, levels=[0.5], colors="green", linewidths=2)

    plt.legend()
    plt.draw()

# Функция для обработки кликов
def on_click(event):
    if event.xdata is None or event.ydata is None:
        return

    class_label = 1 if event.button == 1 else 0
    X.append([event.xdata, event.ydata])
    y.append(class_label)

    update_plot()

# Создаём интерактивный график
fig = plt.figure()
fig.canvas.mpl_connect("button_press_event", on_click)

update_plot()
plt.show()

print("Смещения для скрытых слоёв:")
for i, bias in enumerate(mlp.intercepts_):
    print(f"Слой {i+1}: {bias}")

print("Веса между слоями:")
for i, weights in enumerate(mlp.coefs_):
    print(f"Слой {i+1} -> {i+2}: {weights.shape}")

print("Веса первого скрытого слоя:", mlp.coefs_[0])
print("Веса второго скрытого слоя:", mlp.coefs_[1])
print("Веса третьего скрытого слоя:", mlp.coefs_[2])