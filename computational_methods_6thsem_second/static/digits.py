digits = [3, 6, 8, 9]
numbers = []

# Генерируем все двузначные числа без повторения цифр
for i in digits:
    for j in digits:
        if i != j:
            num = i * 10 + j
            numbers.append(num)

# Проверяем остатки от деления на 8
even_remainders = [0, 2, 4, 6]
valid_numbers = []

for num in numbers:
    remainder = num % 8
    if remainder in even_remainders:
        valid_numbers.append(num)

print("Все возможные числа из цифр 3,6,8,9:", sorted(numbers))
print("Числа с четными остатками при делении на 8:", sorted(valid_numbers))