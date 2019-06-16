import numpy as np
import pandas as pd


def winner(player_one, player_two):
    return {
        'scissors': lambda x: 1 if x == 'paper' else 0 if x == 'scissors' else -1,
        'paper': lambda x: 1 if x == 'rock' else 0 if x == 'paper' else -1,
        'rock': lambda x: 1 if x == 'scissors' else 0 if x == 'rock' else -1
    }[player_one](player_two)


def alias(word):
    return {
        'R': 'rock',
        'S': 'scissors',
        'P': 'paper'
    }[word]


def find_probability_of_all_changes(step, other_step, input_data):
    second_player = input_data.loc[:, 1]  # Берем второй столбец и матрицы (ответы системы)
    step_indexes_second_player = second_player[second_player == step].index.values
    # Достаем значения индексов где степы совпадают
    other_step_indexes_second_player = second_player[second_player == other_step].index.values
    # Достаем так же значения индексов для степа куда хотим перейти
    changes = 0
    for i in step_indexes_second_player:
        changes += (other_step_indexes_second_player == (i + 1)).sum()
        # Находим все переходы из индекса (первого шага) в индекс + 1  (второго шага)
    return changes / other_step_indexes_second_player.shape[0]
    # Делим количество переходов из первого шага во второй на количество второго шага (в который хоти перейти)


def find_probability_of_winning(choice, win_result, input_data):
    end_data = input_data[input_data.iloc[:, 1] == choice]
    # Вытягиваем все строки из матрицы где второе значение является choice
    probability = end_data[end_data.iloc[:, 2] == win_result].shape[0] / end_data.shape[0]
    # Сравниваем все значения в столбце 3 с вин ресалтом и делим на количество строк в матрице
    return probability


# def main():
start = ['paper', 'scissors', 'rock']
probability = [1 / 3, 1 / 3, 1 / 3]

data = []

# Рандомно генерим значения для игрока и системы
while len(data) != 30:
    first_player = np.random.choice(start, replace=True, p=probability)
    second_player = np.random.choice(start, replace=True, p=probability)
    result = winner(first_player, second_player)
    data.append([first_player, second_player, result])

inp_data = pd.DataFrame(data)

step_1 = ['paper', 'scissors', 'rock']

step_prob = [
    [
        find_probability_of_all_changes('paper', 'paper', inp_data),
        find_probability_of_all_changes('scissors', 'paper', inp_data),
        1 - (find_probability_of_all_changes('paper', 'paper', inp_data) + find_probability_of_all_changes(
            'scissors',
            'paper',
            inp_data))
    ],
    [
        find_probability_of_all_changes('paper', 'scissors', inp_data),
        find_probability_of_all_changes('scissors', 'scissors', inp_data),
        1 - (find_probability_of_all_changes('paper', 'scissors', inp_data) + find_probability_of_all_changes(
            'scissors',
            'scissors',
            inp_data))
    ],
    [
        find_probability_of_all_changes('paper', 'rock', inp_data),
        find_probability_of_all_changes('scissors', 'rock', inp_data),
        1 - (find_probability_of_all_changes('paper', 'rock', inp_data) + find_probability_of_all_changes(
            'scissors',
            'rock',
            inp_data))
    ]
]

macierz_przejsc = np.array(step_prob)

winning = [-1, 0, 1]

winning_chance = [
    [
        find_probability_of_winning('paper', -1, inp_data),
        find_probability_of_winning('paper', 0, inp_data),
        find_probability_of_winning('paper', 1, inp_data)
    ],
    [
        find_probability_of_winning('scissors', -1, inp_data),
        find_probability_of_winning('scissors', 0, inp_data),
        find_probability_of_winning('scissors', 1, inp_data)
    ],
    [
        find_probability_of_winning('rock', -1, inp_data),
        find_probability_of_winning('rock', 0, inp_data),
        find_probability_of_winning('rock', 1, inp_data)
    ]
]

macierz_wyjsc = np.array(winning_chance)

initial = np.random.choice(start, replace=True)
n = 20
st = 1

for i in range(n):
    if st:
        state = initial
        st = 0
        print(state)
    if state == alias("P"):
        activity = np.random.choice(winning, p=winning_chance[0])
        print(state)
        print(activity)
        state = np.random.choice(step_1, p=step_prob[0])
    if state == alias("S"):
        activity = np.random.choice(winning, p=winning_chance[1])
        print(state)
        print(activity)
        state = np.random.choice(step_1, p=step_prob[1])
    if state == alias("R"):
        activity = np.random.choice(winning, p=winning_chance[2])
        print(state)
        print(activity)
        state = np.random.choice(step_1, p=step_prob[2])
print("\n")

# if __name__ == '__main__':
#     main()
