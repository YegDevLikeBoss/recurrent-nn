from keras.datasets import mnist
from keras.models import load_model
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Вывод тестового изображения
print("Тестовое изображение")
for i in x_test[0]:
    show = "".join(f"{j:3}" for j in i)
    print(show)

# Загрузка сохраненной модели
model = load_model("./digit_model", compile = True)

# Массив с одним изображением
test_number = []
for i in range(len(x_test[0])):
    row = []
    for j in range(len(x_test[0][i])):
        row.append(np.array([x_test[0][i][j]]))
    test_number.append(row)
samples_to_predict = np.array([test_number])
print("Длина матрицы")
print(samples_to_predict.shape)

# Предсказание
predictions = model.predict(samples_to_predict)
print("Значения на выходном слою")
print(predictions)

# Итоговое число
classes = np.argmax(predictions, axis = 1)
print("Предсказанное число")
print(classes)