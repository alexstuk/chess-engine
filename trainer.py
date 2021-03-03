import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import tensor_board
from tensorflow.keras.callbacks import ModelCheckpoint, ModelCheckpoint


#loading from according to save files while counting move/win for black/white and training

def load_known_size(f_name, nrow, ncol):
    # Initializing empty numpy array
    x = np.empty((nrow, ncol), dtype=np.uint8)
    # Opening the file with positions and populating the array
    with open(f_name) as f:
        for irow, line in enumerate(f):
            x[irow, :] = line.split()
    return x


def load_big_file(f_name):
    position_number = 0
    # Unknown number of lines, so use list to store positions
    # Initializing lists for features and labels
    rows_f = []
    rows_l = []

    # Opening the file with positions
    with open(f_name) as f:
        for line in f:
            line = [float(s) for s in line.split()]
            # Adding positions to the features and labels array, line by line
            rows_f.append(np.array(line, dtype=np.uint8))
            rows_l.append(np.array([labels[position_number]][0], dtype=np.uint8))

            position_number += 1

    # Converting list of vectors rows_f to array before returning
    return np.vstack(rows_f), rows_l


def shuffle_in_unison(a, b):
    # Shuffling features and labels simultaneously, using the state of array
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    return a, b


positions_in_file = {0: 938669, 1: 938685, 2: 887826, 3: 813322, 4: 1038612, 5: 940463}

posInFile = {0: 4983263, 1: 5455042, 2: 6314536, 3: 4493998, 4: 4523274, 5: 4420516}

white_move_win = 0
white_move_loss = 0
black_move_win = 0
black_move_loss = 0

np.set_printoptions(linewidth=np.inf)

z = 0
o = 0

# Loading all the positions from each file/process
for i in range(6):
    positions_in_file = posInFile[i]

    file_name_positions = 'Features/positions_' + str(i) + '.txt'
    file_name_scores = 'Labels/scores_' + str(i) + '.txt'

    labels = load_known_size(file_name_scores, positions_in_file, 1)

    features, labels = load_big_file(file_name_positions)

    # After saving loaded positions to array, adding them to initialized array
    if i == 0:
        all_features = features
        all_labels = labels
    else:
        all_features = np.concatenate((all_features, features))
        all_labels = np.concatenate((all_labels, labels))

    print('process done: ', i)


shuffle_in_unison(all_features, all_labels)


# Setting of some positions for testing (3%)
training_len = int(round(len(all_labels) * 0.97))
features_testing = all_features[training_len:]
labels_testing = all_labels[training_len:]
features_training = all_features[:training_len]
labels_training = all_labels[:training_len]

# Getting the input dimensions of the array with features
input_dim = features_training.shape[1]

# Initializing sequential model
model = Sequential()

model.add(Dense(778, input_dim=input_dim, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.05))

model.add(Dense(1, activation='sigmoid'))

# Setup the optimizer and learning rate
opt = tf.keras.optimizers.Adam(lr=0.0004)

# Compile model
model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)
NAME = 'CHESS'
# Creating tensor board
tensor_board = tensor_board(log_dir="logs".format())

# Unique file name that will include the epoch and the validation acc for that epoch
file_path = "RNN_Final-{epoch:02d}-{acc:.4f}-{val_acc:.4f}--{loss:.4f}-{val_loss:.4f}" 

# Initializing the model checkpoint to save model weights at the end of every epoch, if it's the best seen so far
checkpoint = ModelCheckpoint("models/{}.model".format(file_path, monitor='val_acc', verbose=1, save_best_only=True,
                                                      mode='max'))

# Training the model
history = model.fit(
    features_training, labels_training,
    batch_size=512,
    epochs=500,
    validation_data=(features_testing, labels_testing),
    callbacks=[tensor_board, checkpoint],
)

# Use the estimator 'evaluate' method to evaluate the model
score = model.evaluate(features_testing, labels_testing, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Saving the model
model.save('epic_num_reader.model')
