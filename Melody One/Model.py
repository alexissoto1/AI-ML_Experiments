# Based on https://github.com/hexahedria/biaxial-rnn-music-composition

import numpy as np
import glob
from tqdm import tqdm #Create loading bar to show progress!!!!
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops

import midi_manipulation 


import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_songs(path):
    files = glob.glob('{}/*.mid*'.format(path))
    songs = [] #Empty array with songs
    for f in tqdm(files):
        try:
            song = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            if np.array(song).shape[0] > 50:
                songs.append(song)
        except Exception as e:
            raise e
            print("Error at loading MIDI files")
    return songs


##################################################################################################################################################################

# Model parameters

lowest_note = midi_manipulation.lowerBound  #Lowest note (Index)
highest_note = midi_manipulation.upperBound  #Highest note (Index)
note_range = highest_note - lowest_note 

num_timesteps = 18  # This is the number of timesteps that we will create at a time
n_visible = 2 * note_range * num_timesteps  # Visible layers
n_hidden = 60  # This is the size of the hidden layer

num_epochs = 150  # Number of times going through data.
batch_size = 8  # The number of training examples that we are going to send through the RBM at a time.
lr = tf.constant(0.009, tf.float32)  # The learning rate of model

# Variables:

# The placeholder variable that holds our data
x = tf.placeholder(tf.float32, [None, n_visible], name="x")
# The weight matrix that stores the edge weights
W = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name="W")
# The bias vector for the hidden layer
bh = tf.Variable(tf.zeros([1, n_hidden], tf.float32, name="bh"))
# The bias vector for the visible layer
bv = tf.Variable(tf.zeros([1, n_visible], tf.float32, name="bv"))


# Sampling from a vector of probabilities
def sample(probs):
    # Takes in a vector of probabilities, and returns a random vector of 0s and 1s sampled from the input vector
    return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))


# This function runs the gibbs chain. We will call this function in two places:
#    - When we define the training update step
#    - When we sample our music segments from the trained RBM
def gibbs_sample(k):
    # Runs a k-step gibbs chain to sample from the probability distribution of the RBM defined by W, bh, bv
    def gibbs_step(count, k, xk):
        # Runs a single gibbs step. The visible values are initialized to xk
        hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh))  # Propagate the visible values to sample the hidden values
        xk = sample(
            tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv))  # Propagate the hidden values to sample the visible values
        return count + 1, k, xk

    # Run gibbs steps for k iterations
    ct = tf.constant(0)  # counter
    [_, _, x_sample] = control_flow_ops.while_loop(lambda count, num_iter, *args: count < num_iter,
                                                   gibbs_step, [ct, tf.constant(k), x])
    # This is not strictly necessary in this implementation,
    # but if you want to adapt this code to use one of TensorFlow's
    # optimizers, you need this in order to stop tensorflow from propagating gradients back through the gibbs step
    x_sample = tf.stop_gradient(x_sample)
    return x_sample


# Training Update Code

# First, we get the samples of x and h from the probability distribution
# The sample of x
x_sample = gibbs_sample(1)
# The sample of the hidden nodes, starting from the visible state of x
h = sample(tf.sigmoid(tf.matmul(x, W) + bh))
# The sample of the hidden nodes, starting from the visible state of x_sample
h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh))

# Next, we update the values of W, bh, and bv,
# based on the difference between the samples that we drew and the original values
size_bt = tf.cast(tf.shape(x)[0], tf.float32)
W_adder = tf.multiply(lr / size_bt,
                      tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
bv_adder = tf.multiply(lr / size_bt, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
bh_adder = tf.multiply(lr / size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))
# When we do sess.run(updt), TensorFlow will run all 3 update steps
updt = [W.assign_add(W_adder), bv.assign_add(bv_adder), bh.assign_add(bh_adder)]

#Running session and graphics (tqdm)

def Run(path): 
    
    songs = get_songs(path) #Folder with MIDI files. Path needs to be entered here!
    print("{} songs processed".format(len(songs)))
        
    with tf.Session() as sess:
        
        # initialize the variables of the model
        init = tf.global_variables_initializer() #Look up for this extension
        sess.run(init) #Init variables on tf
        print("Test")
        # Run through all of the training data num_epochs times
        for epoch in tqdm(range(num_epochs)):
            for song in songs:
                # The songs are stored in a time x notes format. The size of each song is timesteps_in_song x 2*note_range
                # Here we reshape the songs so that each training example
                # is a vector with num_timesteps x 2*note_range elements
                song = np.array(song)
                song = song[:int(np.floor(song.shape[0] // num_timesteps) * num_timesteps)]
                song = np.reshape(song, [song.shape[0] // num_timesteps, song.shape[1] * num_timesteps])
                # Train the RBM on batch_size examples at a time
                #print(len(song), batch_size)
                for i in range(1, len(song), batch_size):
                    tr_x = song[i:i + batch_size]
                    sess.run(updt, feed_dict={x: tr_x})
        
        # Making music after model training!
        # Run a gibbs chain where the visible nodes are initialized to 0
        sample = gibbs_sample(1).eval(session=sess, feed_dict={x: np.zeros((10, n_visible))})
        for i in range(sample.shape[0]):
            if not any(sample[i, :]):
                continue
            # Here we reshape the vector to be time x notes, and save as midi file
            S = np.reshape(sample[i, :], (num_timesteps, 2 * note_range))
            midi_manipulation.noteStateMatrixToMidi(S, "Sample_{}".format(i))
