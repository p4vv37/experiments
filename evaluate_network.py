import numpy as np

from easyAI import TwoPlayerGame

import torch
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
from copy import deepcopy

from lib import find_four, RandomPlayer


class GPTPlayer:
    def __init__(self, name="Human"):
        checkpoint_path = './fine-tuned-gpt2'
        self.model = GPT2LMHeadModel.from_pretrained(checkpoint_path).to("cuda:0")
        self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)

        self.special_tokens = {f'[{letter}{number}]': number for letter in 'AB' for number in range(7)}
        self.name = name

    def ask_move(self, game):
        input_decoded = "".join([["[A", "[B"][num % 2] + str(x) + "]" for num, x in enumerate(game.history)])
        input = self.tokenizer(input_decoded, return_tensors='pt', padding=True, truncation=True).to("cuda:0")

        # Tokenize and get the model prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(input['input_ids'], attention_mask=input['attention_mask'],
                                          num_return_sequences=1, max_new_tokens=1,
                                          pad_token_id=self.tokenizer.pad_token_id)
        move_token = self.tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True).split(" ")[-1]
        return self.special_tokens[move_token]


class ConnectFour(TwoPlayerGame):
    """
    The game of Connect Four, as described here:
    http://en.wikipedia.org/wiki/Connect_Four
    """

    def __init__(self, players, board=None):
        self.players = players
        self.history = []
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

    def play(self, nmoves=1000, verbose=True):
        """
        Method for starting the play of a game to completion. If one of the
        players is a Human_Player, then the interaction with the human is via
        the text terminal.

        Parameters
        -----------

        nmoves:
          The limit of how many moves (plies) to play unless the game ends on
          it's own first.

        verbose:
          Setting verbose=True displays additional text messages.
        """

        self.history = []

        if verbose:
            self.show()

        for self.nmove in range(1, nmoves + 1):

            if self.is_over():
                break

            move = self.player.ask_move(self)
            self.history.append(move)
            self.make_move(move)

            if verbose:
                print(
                    "\nMove #%d: player %d plays %s :"
                    % (self.nmove, self.current_player, str(move))
                )
                self.show()
            self.switch_player()
        self.history.append(deepcopy(self))
        return self.history


results = list()
number_of_games = 100
gpt_player = GPTPlayer()
random_player = RandomPlayer()
with tqdm(total=number_of_games) as pbar:
    for n in range(number_of_games):
        result = ConnectFour([random_player, gpt_player]).play(verbose=False)
        results.append(len(result) % 2)
        pbar.n = n
        pbar.refresh()

print(F"GPT2 win-rate: {100.0*sum(results) / number_of_games}%")
