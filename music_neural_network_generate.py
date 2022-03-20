import music21
import glob
import numpy as np
from music21 import instrument, note, chord, converter,stream
from music21.midi import MidiTrack, MidiFile
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dropout, Dense, LSTM, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import time

#WHAT TO DO
#CHANGE THE FILEPATH TO THE NOTES U SAVED
#CHANGE THE WEIGHTS TO THE NOTES' WEIGHT

#function to get the notes inside those songs(a,b,c,d,e,f,g)

def generate():
    """generate a midi files"""
    filepath = "C:\\Users\\user\\Desktop\\python's_stuff\\geany_python\\notes_saved8"
    with open(filepath ,'rb') as f:
        notes = pickle.load(f)
    pitchnames = sorted(set(item for item in notes))
    #get len of pitch names
    n_vocab = len(set(notes))
    
    network_input, normalized_input = prepare_sequence(pitchnames,notes, n_vocab) 
    model = create_network(normalized_input, n_vocab) #all the functions above are excess
    prediction_output =  generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)

def prepare_sequence(pitchnames, notes, n_vocab):
    """prepare the sequence used by the Neural Network"""
    sequence_length = 100
    
    #get all pitch name 
   
    
    # ~ print("Notes: {}".format(a))
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
    print("Total network_input: {}".format(network_input))
    # ~ print("Total network_output: {}".format(network_output))
    
    # ~ print(network_input)
    # ~ print(network_input)
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # ~ print("NI: {}".format(network_input))
    normalized_input = normalized_input / float(n_vocab)
    # ~ print("NI2: {}".format(network_input))
    
    
    # ~ print("network output before tocategorical: {}".format(network_output))
    # ~ network_output = to_categorical(network_output)
    # ~ print("NO: {}".format(network_output))
    # ~ print(network_input)
    # ~ print(network_input.shape)
    return (network_input,  normalized_input)
    
#function to create an ANN (network)
def create_network(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM( #classic
        64,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.9)) #0.9 is baddddddddddddddddddddddddddddddddddd
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.9))
    model.add(LSTM(64))
    model.add(Dense(32))
    model.add(Dropout(0.9))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
                                 metrics =['accuracy'])
    #will try to understand different kind of loss, optimizer and metrics in the future as i alr know some basic one
    filepath = "C:\\Users\\user\\Desktop\\MLmusic\\songs for carnival\\EternalHarvest\\weights-improvement-4865-loss0.0364-acc0.992180-bigger.hdf5"
    model.load_weights(filepath) #we can simply load the weight from the another file or just copy the other 2 needed function into the file
    
    return model
    
def generate_notes(model, network_input, pitchnames, n_vocab):
    """generate notes from the neural network based on a sequence of notes"""
    start = np.random.randint(0, len(network_input)-1)
    
    int_to_note = dict((number, note) for number,note in enumerate(pitchnames))
    print("Int to note: {}".format(int_to_note))
    
    pattern = network_input[start]
    prediction_output = []  
    print("Pattern: {}".format(pattern))
    #generate 500 notes
    for note_index in range(500):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        print("prediction_input2: {}".format(prediction_input))
        prediction_input = prediction_input / float(n_vocab)
        print("prediction_input3: {}".format(prediction_input))
        prediction = model.predict(prediction_input, verbose =0)
        print("Prediction :{}".format(prediction))
        index = np.argmax(prediction)
        print("Index: {}".format(index))
        # ~ print("Index: {}".format(index))    
        result = int_to_note[index]
        print("Result: {}".format(result))
        prediction_output.append(result)
        # ~ print("Prediction_output: {}".format(prediction_output))
        pattern.append(index)
        # ~ print("Len of pattern: {}".format(len(pattern)))
        # ~ print("Pattern appending index: {}".format(pattern))
        pattern = pattern[1:len(pattern)]
        # ~ print("Pattern : {}".format(pattern))
        print(prediction_output)
    return prediction_output
    
def create_midi(prediction_output):
    offset = 0
    output_notes = []
    
    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                print("current_note: {}".format(current_note))
                new_note = note.Note(int(current_note))
                print("New_note: {}".format(new_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
                print("Notes: {}".format(notes))
            new_chord = chord.Chord(notes)
            new_chord.offset = offset 
            output_notes.append(new_chord)
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset 
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
            print(output_notes)
            
        offset+=0.5
    
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('musicxml', fp ='test_output_mid9')
    midi_stream.show('musicxml')
    
if __name__ == '__main__':
    generate()    

