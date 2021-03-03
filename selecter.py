import multiprocessing
import chess.pgn
import numpy as np
import chess.engine
from numpy import save
from pandas.core.common import flatten
import json
import datetime
from time import sleep

# creating the database, accounting for wins/losses and material
#collecting all positions with losing side having less material


def uneven_material(fen):
    # Function counts the material from fen notation
    pieces_value_white = {'q': 9,
                        'r': 5,
                        'b': 3,
                        'n': 3,
                        'p': 1
                        }
    pieces_value_black = {'Q': 9,
                        'R': 5,
                        'B': 3,
                        'N': 3,
                        'P': 1
                        }
    white_material = 0
    black_material = 0
    first_part = True

    for letter in fen:
        if first_part:
            if letter in pieces_value_white:
                white_material += pieces_value_white[letter]
            if letter in pieces_value_black:
                black_material += pieces_value_black[letter]
            if letter == ' ':
                first_part = False

    if white_material > black_material:
        return 'white', white_material - black_material
    if white_material < black_material:
        return 'black', black_material - white_material
    if white_material == black_material:
        return 'none', 0

def fentoonesandzeros(fen):
    # Pieces' binary representation
    pieces = {'K': [1, 0, 0, 0, 0, 0],
              'Q': [0, 1, 0, 0, 0, 0],
              'R': [0, 0, 1, 0, 0, 0],
              'B': [0, 0, 0, 1, 0, 0],
              'N': [0, 0, 0, 0, 1, 0],
              'P': [0, 0, 0, 0, 0, 1],
              'none': [0, 0, 0, 0, 0, 0]
              }
    # Verticals  binary representation
    squares = {'a': [0, 0, 0],
               'b': [1, 0, 0],
               'c': [0, 1, 0],
               'd': [0, 0, 1],
               'e': [1, 1, 0],
               'f': [1, 0, 1],
               'g': [0, 1, 1],
               'h': [1, 1, 1]}

    # Horizontals binary representation
    # Only 3 and 6 different because en passant move can happen only on these rows
    numbers = {'1': [1, 1],
               '2': [1, 1],
               '3': [0, 1],
               '4': [1, 1],
               '5': [1, 1],
               '6': [1, 0],
               '7': [1, 1],
               '8': [1, 1]}

    # No en passant condition
    enpassant_letter = [0, 0, 0]
    enpassant_number = [0, 0]

    white_list = []
    black_list = []
    black_pieces = ['r', 'n', 'b', 'q', 'k', 'p']
    white_pieces = ['R', 'N', 'B', 'Q', 'K', 'P']

    # Space dividers in fen representation
    dividers = 0
    # Castling and next turn
    conditions = [0, 0, 0, 0, 0]

    # From fen to ones and zeros transformation loop
    for letter in fen:
        # First dealing with pieces
        if dividers == 0:
            if letter in numbers:
                letter = int(letter)
                # Using multiplier to represent multiple same pieces on a row
                white_list = [0, 0, 0, 0, 0, 0] * letter + white_list
                black_list = [0, 0, 0, 0, 0, 0] * letter + black_list
            if letter in black_pieces:
                white_list = [0, 0, 0, 0, 0, 0] + white_list
                black_list = pieces[letter.upper()] + black_list
            if letter in white_pieces:
                black_list = [0, 0, 0, 0, 0, 0] + black_list
                white_list = pieces[letter] + white_list
            if letter == ' ':
                dividers += 1
        # Working on castling and next move transformation
        if 4 > dividers > 0:
            if letter == 'K':
                conditions[0] = 1
            if letter == 'Q':
                conditions[1] = 1
            if letter == 'k':
                conditions[2] = 1
            if letter == 'q':
                conditions[3] = 1
            if letter == 'b':
                conditions[4] = 1
            if letter == ' ':
                dividers += 1
        # Working on en passant representation
        if 6 > dividers > 3:
            if letter in squares:
                enpassant_letter = squares[letter]
            if letter in numbers:
                enpassant_number = numbers[letter]
            if letter == ' ':
                dividers += 1

    # Putting it all together
    position = white_list + black_list + conditions + enpassant_letter + enpassant_number
    position = np.asarray(list(flatten(position)))

    return position


def savingdata(process, material_dictMax=None):
    # Number of positions to be saved at once
    # Less is better, saves time if process dies and have to restart
    positions_dump = 250
    # Creating zero array for positions and scores
    positions_array = np.zeros(shape=(positions_dump, 778))
    scores_array = np.zeros(shape=(positions_dump, 1))
    # Opening a database per process
    process_database = 'database_' + str(process) + '.pgn'
    pgn = open('Database3/' + process_database, 'r', encoding="ISO-8859-1")
    # Number of games already loaded and number of all games in database
    games_loaded = 0
    games_load = 3067273
    # Dictionary that collects number of games with various difference in material
    material_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

    # Last database will less games than others
    if process == 5:
        games_load = 3067257

    position_number = 0

    # Loading statistics (white wins, losses, number of games loaded etc)
    with open("Labels/saves" + str(process) + ".txt", "r") as file:
        info = file.read()
        info = json.loads(info.rstrip(';\n'))

        games_preloaded = int(info['games_loaded'])
        saved_positions = int(info['positions saved'])
        millions = (saved_positions + 100000 // 2) // 100000
        black_wins = int(info['bw'])
        black_losses = int(info['bl'])
        white_wins = int(info['ww'])
        white_losses = int(info['wl'])

        for dic_pts in range(1, 9):
            material_dict[dic_pts] = int(info[str(dic_pts)])

    # Initializing Stockfish engine
    engine = chess.engine.SimpleEngine.popen_uci("stockfish-10-win/Windows/stockfish_10_x64.exe")
    # If restarting the algorithm, quickly skipping already analyzed games
    if games_preloaded < games_load:
        while games_loaded < games_load:
            if games_loaded < games_preloaded:
                # Quick fetching
                chess.pgn.read_headers(pgn)
                games_loaded += 1
            else:
                next_game = chess.pgn.read_game(pgn)
                board = next_game.board()

                # Number of turns made between collected positions
                # Setting it high to start collecting positions immediately
                since_record = 10
                turn = 0

                for every_move in next_game.mainline_moves():
                    board.push(every_move)
                    turn += 1
                    since_record += 1

                    # Looking for good positions and saving them
                    # Positions with a low move count are all alike, positions with too many moves are too simple
                    if 14 < turn < 100 and since_record > 10:

                        # Simplifying position by cutting out repeated moves and total turns from fen notation
                        fen_ = str(board.fen())

                        spaces = 0
                        chars = 0

                        while spaces < 2:
                            len_fen = len(fen_)
                            if fen_[len_fen - 1 - chars] == ' ':
                                spaces += 1
                            chars += 1
                        fen_ = fen_[:len_fen - chars]
                        # Replacing with standard 0 repeated moves and 30 total turns made
                        fen_ += ' 0 30'
                        # Counting the material
                        uneven_m = uneven_material(fen_)
                        material_winner = uneven_m[0]
                        material_dif = uneven_m[1]

                        # Saving all info on games with high material difference together
                        if 13 > material_dif > 7:
                            material_dif = 8
                        # Dropping the games with too low/high material difference before analyzing
                        if 9 > material_dif > 0:

                            board_ = chess.Board(fen_)
                            # Analyzing the position
                            info = engine.analyse(board_, chess.engine.Limit(time=0.01))
                            # Interpreting Stockfish analysis
                            if "score" in info:
                                score = str(info["score"])

                                if score[0] == '#':  # check mate
                                    znak = score[1]
                                    score = znak + "9999"
                                else:
                                    znak = score[:1]

                                if score != '0':
                                    score = int(score[1:])
                                score = int(score)
                                good_position = True
                                # To avoid bias:
                                # Dropping the games with too much/low advantage on one side
                                # Taking into account the material and who is making next turn to avoid bias
                                if (900 > score > 250) and ((material_winner == 'white'
                                                             and turn % 2 == 1 and znak == '-')
                                                            or (material_winner == 'black'
                                                                and turn % 2 == 0 and znak == '-')
                                                            or (material_winner == 'white'
                                                                and turn % 2 == 0 and znak == '+')
                                                            or (material_winner == 'black'
                                                                and turn % 2 == 1 and znak == '+')):

                                    # Loading even number positions with wins and losses on their move for both sides
                                    if games_loaded > 100:
                                        if turn % 2 == 0 and znak == '+':
                                            if white_wins > white_losses:
                                                good_position = False
                                        if turn % 2 == 1 and znak == '-':
                                            if black_wins < black_losses:
                                                good_position = False

                                    # For better data quality analyzing position again with more time
                                    if good_position:
                                        info = engine.analyse(board_, chess.engine.Limit(time=0.2))
                                        score = str(info["score"])
                                        # Interpreting Stockfish analysis
                                        if score[0] == '#':  # check mate
                                            znak = score[1]
                                            score = znak + "9999"
                                        else:
                                            znak = score[:1]
                                        if score != '0':
                                            score = int(score[1:])
                                        score = int(score)

                                        # Filtering out the positions the same way second time
                                        if (600 > score > 149) and ((material_winner == 'white' and turn % 2 == 1 and znak == '-')
                                                                    or (material_winner == 'black' and turn % 2 == 0 and znak == '-')
                                                                    or (material_winner == 'white' and turn % 2 == 0 and znak == '+')
                                                                    or (material_winner == 'black' and turn % 2 == 1 and znak == '+')):
                                            material_dict[material_dif] += 1
                                            since_record = 0
                                            saved_positions += 1
                                            fen = fen_
                                            position = fentoonesandzeros(str(fen))

                                            positions_array[position_number] = position

                                            if turn % 2 == 0 and znak == '+':
                                                    white_wins += 1
                                            if turn % 2 == 1 and znak == '-':
                                                black_losses += 1
                                            if turn % 2 == 0 and znak == '-':
                                                    white_losses += 1
                                            if turn % 2 == 1 and znak == '+':
                                                black_wins += 1

                                            if znak == '+':
                                                scores_array[position_number] = 1
                                            else:
                                                scores_array[position_number] = 0

                                            # Extracting and showing relevant statistics every 100.000 games
                                            if saved_positions > 100000 * millions:
                                                games_preloaded_info = 0
                                                saved_positions_info = 0
                                                black_wins_info = 0
                                                black_losses_info = 0
                                                white_wins_info = 0
                                                white_losses_info = 0

                                                # Loading stats from save file
                                                for i in range(6):
                                                    with open("Labels/saves" + str(i) + ".txt", "r") as file:
                                                        info = file.read()
                                                        info = json.loads(info.rstrip(';\n'))
                                                        games_preloaded_info += int(info['games_loaded'])
                                                        saved_positions_info += int(info['positions saved'])
                                                        black_wins_info += int(info['bw'])
                                                        black_losses_info += int(info['bl'])
                                                        white_wins_info += int(info['ww'])
                                                        white_losses_info += int(info['wl'])

                                                print(process, ' games: ', games_preloaded_info, ' positions: ', saved_positions_info, ' black_wins: ', black_wins_info, ' black_losses ', black_losses_info, ' white_wins: ', white_wins_info, ' white_losses: ', white_losses_info)
                                                millions += 1

                                            if position_number == positions_dump - 1:
                                                if process == 111:
                                                    print('games_loaded:', games_loaded * 6, '   positionsSaved:',
                                                        saved_positions * 6, '   bw:', black_wins * 6,
                                                        '   bl:', black_losses * 6, '   ww:', white_wins * 6, '   wl:', white_losses * 6)

                                                stats = {"games_loaded": games_loaded, "positions saved": saved_positions,
                                                         "bw": black_wins, "bl": black_losses, "ww": white_wins,
                                                         "wl": white_losses}
                                                for i in range(1, 9):
                                                    stats[str(i)] = material_dict[i]

                                                #print('stats: ', stats)

                                                position_number = 0
                                                file_name_positions = 'Features/' + 'positions_' + str(process) + '.txt'
                                                file_name_scores = 'Labels/' + 'scores_' + str(process) + '.txt'
                                                file_name_saves = 'Labels/saves' + str(process) + '.txt'

                                                file = open(file_name_positions, 'a')
                                                np.savetxt(file, positions_array, delimiter=' ', fmt='%d')
                                                file.close()
                                                file = open(file_name_saves, 'w')
                                                json.dump(stats, file)
                                                file.close()
                                                file = open(file_name_scores, 'a')
                                                np.savetxt(file, scores_array, delimiter=' ', fmt='%d')
                                                file.close()
                                                positions_array = np.zeros(shape=(positions_dump, 778))
                                            else:
                                                position_number += 1

                games_loaded += 1

        # Saving positions in the last batch (which a fewer than the one batch holds)
        last_batch = saved_positions % positions_dump
        print('last_batch: ', last_batch)
        print('process: ', process, ' white wins: ', white_wins)
        print('process: ', process, 'black wins: ', black_wins)
        print('process: ', process, ' white losses: ', white_losses)
        print('process: ', process, 'black losses: ', black_losses)

        stats = {"games_loaded": games_loaded, "positions saved": saved_positions,
                 "bw": black_wins, "bl": black_losses, "ww": white_wins,
                 "wl": white_losses}
        for i in range(1, 9):
            stats[str(i)] = material_dict[i]
        scores_array = scores_array[:last_batch]
        positions_array = positions_array[:last_batch]
        file_name_positions = 'Features/' + 'positions_' + str(process) + '.txt'
        file_name_scores = 'Labels/' + 'scores_' + str(process) + '.txt'
        file_name_saves = 'Labels/saves' + str(process) + '.txt'

        file = open(file_name_positions, 'a')
        np.savetxt(file, positions_array, delimiter=' ', fmt='%d')
        file.close()
        file = open(file_name_scores, 'a')
        np.savetxt(file, scores_array, delimiter=' ', fmt='%d')
        file.close()
        file = open(file_name_saves, 'w')
        json.dump(stats, file)
        file.close()

    print('process: ', process, ' finished')

    engine.quit()

# List of all squares on the board
squares = ['a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8', 'c1',
           'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'e1', 'e2', 'e3',
           'e4',' e5', 'e6', 'e7', 'e8', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'g1', 'g2', 'g3', 'g4', 'g5',
           'g6', 'g7', 'g8', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7', 'h8']


def main():
    print(datetime.datetime.now())
    pr_list = []
    for i in range(6):
        p = multiprocessing.Process(target=savingdata, args= (i,))
        pr_list.append(p)
        p.start()

    if len(pr_list) == 6:
        for pr in pr_list:
            pr.join()

    if len(pr_list) == 6:
        for pr in pr_list:
            pr.terminate()


if __name__ == '__main__':
    main()
