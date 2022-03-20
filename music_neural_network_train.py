import music21
import glob
import numpy as np
from music21 import instrument, note, chord, converter
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, LSTM, Activation, Bidirectional
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import time

filepath = 'C:\\Users\\user\\Desktop\\midi songs/*.mid'
notes = []

name = "music_cnn_piano_prac-61-notes-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir = 'logs/{}'.format(name))

def train_network():
    notes = get_notes()
    print(sorted(notes))
    n_vocab = len(set(notes))
    # ~ print("n vocab: {}".format(n_vocab))
    network_input, network_output = prepare_sequence(notes, n_vocab)
    
    model = create_network(network_input, n_vocab)
    train(model, network_input, network_output)
    
def get_notes():
    """get all the notes from the filepath using glob"""
    for files in glob.glob(filepath):
        midi = converter.parse(files)
        # ~ print("Midi: {}".format(midi))
        note_to_parse = 0
        
        try: #file has instrument part
            s2 = instrument.partitionByInstrument(midi)
            # ~ print("S2: {}".format(s2))
            note_to_parse =     s2.parts[0].recurse()
            # ~ print("Note to parse: {}".format(note_to_parse))
        except: #file has a flat structure
            notes_to_parse = midi.flat.notes
            # ~ print("Note to parse: {}".format(note_to_parse))
            
        for element in note_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            if isinstance(element, chord.Chord):
                # ~ print("Element normalOrder: {}".format(element.normalOrder))
                notes.append('.'.join(str(n) for n in element.normalOrder))
                
        with open('notes_saved', 'wb') as f:
            pickle.dump(notes, f)
    # ~ with open('notes_saved', 'rb') as notes:
            # ~ notes = pickle.load(notes)
    # ~ print(notes)
    print("NOTES: {}".format(set(notes)))
    return notes
        # ~ print(notes)

#basically just to prepare the sequenc for the neural network to acquaint
def prepare_sequence(notes, n_vocab):
    """prepare the sequence used by the Neural Network"""
    sequence_length = 100
    
    #get all pitch name 
    pitchnames = sorted(set(item for item in notes))
    # ~ print("Pitchname: {}".format(pitchnames))
    # ~ print("Len of pitchnames: {}".format(len(pitchnames)))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
    # ~ print("Note_to_int: {}".format(note_to_int))
    network_input = []
    network_output = []
    
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i+ sequence_length]
        # ~ print("Sequence_in: {}".format(sequence_in))
        sequence_out = notes[i + sequence_length]
        # ~ print("Sequence_out: {}".format(sequence_out))
        network_input.append([note_to_int[char] for char in sequence_in])
        # ~ print("[note_to_int[char] for char in sequence_in]: {}".format([note_to_int[char] for char in sequence_in]))
        network_output.append([note_to_int[sequence_out]])
        # ~ print("([note_to_int[sequence_out]): {}".format([note_to_int[sequence_out]]))    
        
    n_patterns = len(network_input)
    # ~ print("Total network_input: {}".format(network_input))
    # ~ print("Total network_output: {}".format(network_output))
    
    # ~ print(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # ~ print("NI: {}".format(network_input))
    network_input = network_input/float(n_vocab)
    # ~ print("NI2: {}".format(network_input))
    
    # ~ print("network output before tocategorical: {}".format(network_output))
    print("network_output: {}".format(network_output))
    network_output = to_categorical(network_output)
    print("network_output[1] after to_categorical[1]: {}".format(network_output[1]))
    print("Output shape: {}".format(network_output.shape))
    print("input shape: {}".format(network_input.shape))
    # ~ print("NO: {}".format(network_output))
    # ~ print(network_input)
    # ~ print(network_input.shape)
    return (network_input, network_output)
    
#function to create an ANN (network)
def create_network(network_input, n_vocab):
    """create the architecture of neural network"""
     # ~ kernel_regularizer=tf.keras.regularizers.l2(0.001)
     # ~ activation  = tf.nn.relu
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model
    
def train(model, network_input, network_output):
    """train the neural network"""

    filepath =  "C:\\Users\\user\\Desktop\\MLmusic\\Weight1000epochs2\\weights-improvement-{epoch:02d}-{loss:4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
            filepath,
            monitor = 'loss',
            verbose = 0,
            mode='min'
        )
    callback_list = [checkpoint]
    
    start = time.time()
    model.fit(network_input, network_output, epochs=1000, batch_size =64, callbacks= callback_list)
    end = time.time()
    
    print("Total time used: {}".format(end))
    
if __name__ == '__main__':
    train_network()
    

