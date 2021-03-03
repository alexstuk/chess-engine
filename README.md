# Chess Engine


This softwear allows to collect chess positions and train the model that can be used as a chess engine. 

It is made with TensorFlow and reinforcement machine learning. 

The idea is borrowed from the following research papers:

https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/ICPRAM_CHESS_DNN_2018.pdf

https://www.cs.tau.ac.il/~wolf/papers/deepchess.pdf

All positions used for training are labeled with Stockfish engine as wining or loosing and represented as a binary vector.

Selecter.py selelcts and saves in a file positions from games in a FEN format.

Training.py loads the positions from the file and trains the model.

Testing.py allows to play against the trained model. 
  
To avoid bias positions have to be picked:
1. Number of wining/loosing positions for both black/white sides is equal
2. Number of wininng postions with more material equals to number of winining positions with less material
3. Number of positions where the making next move side wins equals to number of positions where the making next move side loses

To avoid overfitting
1. All positions that are picked from the same are at least 5 turns apart 
2. Dropout regularization is used
3. Only informative positions are used. Positions where one side has too much/little advantage and positions with too few piecies are dropped 
