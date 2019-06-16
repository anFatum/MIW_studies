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


def agains(who):
    return {
        1: 'rock',
        2: 'paper',
        0: 'scissors'
    }[who]


def find_probability_of_all_changes(step, other_step, input_data):
    first_player = input_data.loc[:, 0]
    second_player = input_data.loc[:, 1]
    step_indexes_first_player = first_player[first_player == step].index.values
    step_indexes_second_player = second_player[second_player == step].index.values
    other_step_indexes_first_player = first_player[first_player == other_step].index.values
    other_step_indexes_second_player = second_player[second_player == other_step].index.values
    changes = 0
    for i in step_indexes_first_player:
        changes += (other_step_indexes_first_player == (i + 1)).sum()
    for i in step_indexes_second_player:
        changes += (other_step_indexes_second_player == (i + 1)).sum()
    return changes / np.concatenate((step_indexes_first_player, step_indexes_second_player)).shape[0]


start = ['paper', 'scissors', 'rock']
probability = [1 / 3, 1 / 3, 1 / 3]

data = []

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
        find_probability_of_all_changes('rock', 'paper', inp_data)
    ],
    [
        find_probability_of_all_changes('paper', 'scissors', inp_data),
        find_probability_of_all_changes('scissors', 'scissors', inp_data),
        find_probability_of_all_changes('rock', 'scissors', inp_data)
    ],
    [
        find_probability_of_all_changes('paper', 'rock', inp_data),
        find_probability_of_all_changes('scissors', 'rock', inp_data),
        find_probability_of_all_changes('rock', 'rock', inp_data)
    ]
]

# =========
import sys

print("Print S for scissors, P for paper and R for rock. If you want to stop game print 0")
turn = input()
game_data = pd.DataFrame()
while turn != "0":
    try:
        if len(game_data) == 0:
            computer_turn = np.random.choice(start, replace=True, p=probability)
            human_turn = alias(turn)
            print(f'{human_turn} VS {computer_turn} => {winner(human_turn, computer_turn)}')
            game_data = game_data.append([[human_turn, computer_turn, winner(human_turn, computer_turn)]])
        else:
            last_turn = game_data.iloc[0][0]
            probabilities = [
                find_probability_of_all_changes(last_turn, 'paper', game_data),
                find_probability_of_all_changes(last_turn, 'scissors', game_data),
                find_probability_of_all_changes(last_turn, 'rock', game_data)
            ]
            computer_turn = agains(np.array(probabilities).argmax())
            human_turn = alias(turn)
            print(f'{human_turn} VS {computer_turn} => {winner(human_turn, computer_turn)}')
            game_data.loc[-1] = [human_turn, computer_turn, winner(human_turn, computer_turn)]
            game_data.index = game_data.index + 1
            game_data = game_data.sort_index()
    except KeyError:
        print('Wrong key')
    turn = input('Go again: ')
