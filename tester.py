from pandas.core.common import flatten
import tensorflow as tf
import numpy as np
import chess.pgn
from random import randint


def new_position():
    # Creating starting position in binary format 
    global board_list, pieces
    
    # List of all squares on the board
    squares = ['h1', 'g1', 'f1', 'e1', 'd1', 'c1', 'b1', 'a1', 'h2', 'g2', 'f2', 'e2', 'd2', 'c2', 'b2', 'a2',
               'h3', 'g3', 'f3', 'e3', 'd3', 'c3', 'b3', 'a3', 'h4', 'g4', 'f4', 'e4', 'd4', 'c4', 'b4', 'a4',
               'h5', 'g5', 'f5', 'e5', 'd5', 'c5', 'b5', 'a5', 'h6', 'g6', 'f6', 'e6', 'd6', 'c6', 'b6', 'a6',
               'h7', 'g7', 'f7', 'e7', 'd7', 'c7', 'b7', 'a7', 'h8', 'g8', 'f8', 'e8', 'd8', 'c8', 'b8', 'a8']
    
    # Starting position for every piece 
    start_position_white = {'a1': 'R', 'b1': 'N', 'c1': 'B', 'd1': 'Q', 'e1': 'K', 'f1': 'B', 'g1': 'N', 'h1': 'R',
                          'a2': 'P', 'b2': 'P', 'c2': 'P', 'd2': 'P', 'e2': 'P', 'f2': 'P', 'g2': 'P', 'h2': 'P'}
    start_position_black = {'a8': 'R', 'b8': 'N', 'c8': 'B', 'd8': 'Q', 'e8': 'K', 'f8': 'B', 'g8': 'N', 'h8': 'R',
                          'a7': 'P', 'b7': 'P', 'c7': 'P', 'd7': 'P', 'e7': 'P', 'f7': 'P', 'g7': 'P', 'h7': 'P'}

    for square in squares:
        if square in start_position_white:
            board_list.append(pieces[start_position_white[square]])
        else:
            board_list.append([0, 0, 0, 0, 0, 0])
    for square in squares:
        if square in start_position_black:
            board_list.append(pieces[start_position_black[square]])
        else:
            board_list.append([0, 0, 0, 0, 0, 0])

    conditions = [0, 0, 0, 0, 1]
    en_passant = [0, 0, 0, 0, 0]

    board_list.append(conditions + en_passant)


def fen_to_binary(fen):
    # Binary for each square
    squares = {'a': [0, 0, 0],
               'b': [1, 0, 0],
               'c': [0, 1, 0],
               'd': [0, 0, 1],
               'e': [1, 1, 0],
               'f': [1, 0, 1],
               'g': [0, 1, 1],
               'h': [1, 1, 1]}

    # Only 3 and 6 different because enpasant move can happen only on these rows
    numbers = {'1': [1, 1],
               '2': [1, 1],
               '3': [0, 1],
               '4': [1, 1],
               '5': [1, 1],
               '6': [1, 0],
               '7': [1, 1],
               '8': [1, 1]}

    # No en passant condition
    en_passant_letter = [0, 0, 0]
    en_passant_number = [0, 0]

    white_list = []
    black_list = []
    # List of all pieces as in fen notation
    black_pieces = ['r', 'n', 'b', 'q', 'k', 'p']
    white_pieces = ['R', 'N', 'B', 'Q', 'K', 'P']

    dividers = 0
    conditions = [0, 0, 0, 0, 0]

    # from fen to ones and zeros
    for letter in fen:
        # Position of pieces in the beginning of fen, no spaces yet
        if dividers == 0:
            if letter in numbers:
                letter = int(letter)
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
        # Position of info about castling and next move in fen
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
        # Position of info about en passant in fen
        if 6 > dividers > 3:
            if letter in squares:
                print('en_passant_letter ', en_passant_letter)
                en_passant_letter = squares[letter]
            if letter in numbers:
                en_passant_number = numbers[letter]
                print('en_passant_number ', en_passant_number)
            if letter == ' ':
                dividers += 1

    # Putting it all together
    pos = white_list + black_list + conditions + en_passant_letter + en_passant_number
    # Turning list into array
    pos = np.asarray([list(flatten(pos))])

    return pos


# Binary for every piece 
pieces = {'K': [1, 0, 0, 0, 0, 0],
          'Q': [0, 1, 0, 0, 0, 0],
          'R': [0, 0, 1, 0, 0, 0],
          'B': [0, 0, 0, 1, 0, 0],
          'N': [0, 0, 0, 0, 1, 0],
          'P': [0, 0, 0, 0, 0, 1],
          'none': [0, 0, 0, 0, 0, 0]
  }     

# Loading the trained model
new_model = tf.keras.models.load_model('RNN_Final-43-0.9265-0.9101--0.1882-0.2268.model')

# Creating beginning binary position and turning it into array
board_list = []
new_position()
board_list = np.asarray([list(flatten(board_list))])

# Initializing chessboard
board = chess.Board()

# Creating infinite loop to play the game forever
while False:
    # The user is prompted to type the move
    usersMove = input('What is your move?')
    usersMove = chess.Move.from_uci(usersMove)
    
    # Making the move picked by the user
    board.push(usersMove)
    
    # Initializing lists and dictionaries for the response moves and their quality
    candidates = []
    candidates_quality = []
    move_dic = {}
    
    # Going through all possible moves that opponent can make 
    for each_move in board.legal_moves:
        board.push(each_move)
        fen = str(board.fen())
        board.pop()
        position = fen_to_binary(fen)
        
        # Evaluating the quality of each move
        predictions = new_model.predict(position)
        
        # Making a list of qualities of each move
        candidates_quality.append(predictions[0][0]) 
        candidates.append(each_move)
        move_dic[each_move] = predictions[0][0]

    # Finding the best quality among all possible moves    
    best_quality = min(candidates_quality) 
    print('best_quality ', best_quality)
    index = 0
    best_quality_indexes = []
    
    for quality in candidates_quality:
        # Finding indexes of all the best moves in the list
        if quality == best_quality:
            best_quality_indexes.append(index)
        index += 1

    print('best_quality_indexes ', best_quality_indexes)
    print('The best moves are:')
    for indexes in best_quality_indexes:
        print(candidates[indexes])
    
    # Picking one move from the best ones on random 
    rand_ind = randint(0, len(best_quality_indexes) - 1)
    rand_ind = best_quality_indexes[rand_ind]
    # This is one of the best move
    random_best_move = candidates[rand_ind]  
    
    # Making the best random move
    board.push(random_best_move)
    print(board)
