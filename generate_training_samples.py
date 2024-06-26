import numpy as np
from easyAI import TwoPlayerGame
from easyAI import AI_Player, SSS
from tqdm import tqdm
from multiprocessing import cpu_count, Process, Manager
import random
import time

from lib import find_four, RandomPlayer


class ConnectFour(TwoPlayerGame):
    """
    The game of Connect Four, as described here:
    http://en.wikipedia.org/wiki/Connect_Four
    Class based on: https://zulko.github.io/easyAI/
    """

    def __init__(self, players, board=None):
        self.players = players
        self.board = (
            board
            if (board is not None)
            else (np.array([[0 for i in range(7)] for j in range(6)]))
        )
        self.current_player = 1

    def possible_moves(self):
        return [i for i in range(7) if (self.board[:, i].min() == 0)]

    def make_move(self, column):
        if not self.board.sum():  # For first move
            line = random.randint(0, 5)
        else:
            line = np.argmin(self.board[:, column] != 0)
        self.board[line, column] = self.current_player

    def show(self):
        print(
            "\n"
            + "\n".join(
                ["0 1 2 3 4 5 6", 13 * "-"]
                + [
                    " ".join([[".", "O", "X"][self.board[5 - j][i]] for i in range(7)])
                    for j in range(6)
                ]
            )
        )

    def lose(self):
        return find_four(self.board, self.opponent_index)

    def is_over(self):
        return (self.board.min() > 0) or self.lose()

    def scoring(self):
        return -100 if self.lose() else 0


def generate_game(games, num_games):
    for _ in range(num_games):
        ai_algo_sss = SSS(5)
        history = ConnectFour([RandomPlayer(), AI_Player(ai_algo_sss)]).play(verbose=False)
        games.put("".join([str(x[1]) for x in history[:-1]]))


if __name__ == "__main__":
    manager = Manager()
    recorded_games = manager.Queue()

    processes = []
    num_workers = cpu_count()  # Number of CPU cores to use
    number_of_games = 100000
    games_per_worker = number_of_games // num_workers
    lock = manager.Lock()

    with tqdm(total=number_of_games) as pbar:
        for _ in range(num_workers):
            p = Process(target=generate_game, args=(recorded_games, number_of_games))
            processes.append(p)
            p.start()

        while any(p.is_alive() for p in processes):
            pbar.n = recorded_games.qsize()
            pbar.refresh()
            time.sleep(0.1)  # Adjust the sleep time for smoother updates

        for p in processes:
            p.join()

    unique_games = set()
    while not recorded_games.empty():
        unique_games.add(recorded_games.get())

    random_unique_games = list(unique_games)
    random.shuffle(random_unique_games)

    print(F"{len(unique_games)}/{recorded_games.qsize()} unique games")

    with open("training_data.txt", "w") as file:
        for game in random_unique_games[:-500]:
            file.write(" ".join(str(x) for x in game) + "\n")

    with open("validation_data.txt", "w") as file:
        for game in random_unique_games[-500:]:
            file.write(" ".join(str(x) for x in game) + "\n")
