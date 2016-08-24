'''
Created on Jul 14, 2016

@author: tylerbonnell
'''

import numpy as np
import tempfile
import tensorflow as tf
from controller import DiscreteDeepQ, HumanController
from simulation import KarpathyGame
from tf_rl import simulate
from tf_rl.models import MLP
from tf_rl.simulation.MovementMatching_game import MovementGame
import csv


#LOG_DIR = tempfile.mkdtemp()
LOG_DIR = "/tmp/mnist_logs"
print(LOG_DIR)
SAVE_DIR = "/Users/tylerbonnell/Documents/RL_trained_agent/rla_03_test/model.ckpt"    #this loads the trained rla 

#define the context from which we would like to get predictions from the rla
current_settings = {
    'objects': [
        'groupMate1',
        'groupMate2',
        'groupMate3',
        'groupMate4',
        'groupMate5',
        'groupMate6',
        'groupMate7',
        'groupMate8',
        'groupMate9',
        'groupMate10',
        'groupMate11',
        'groupMate12',
        'groupMate13',
        'groupMate14',
        'groupMate15',
        'groupMate16',
        'groupMate17',
        'groupMate18'
    ],
    'colors': {
        'hero':   'red',
        'groupMate1': 'green',
        'groupMate2': 'yellow',
        'groupMate3': 'blue',
        'groupMate4': 'orange',
        'groupMate5': 'black',
        'groupMate6': 'grey',
        'groupMate7': 'cyan',
        'groupMate8': 'magenta',
        'groupMate9': 'violet',
        'groupMate10': 'cyan',
        'groupMate11': 'magenta',
        'groupMate12': 'magenta',
        'groupMate13': 'magenta',
        'groupMate14': 'magenta',
        'groupMate15': 'magenta',
        'groupMate16': 'magenta',
        'groupMate17': 'magenta',
        'groupMate18': 'magenta'
    },
    'object_reward': {
        'friend': 0.1,
    },
    'min_offset': 0.785,
    'hero_bounces_off_walls': False,
    'world_size': (3200,1550),
    'hero_initial_position': [826.7389, 761.1064],
    'hero_initial_speed':    [1,   0],
    "maximum_speed":         [50, 50],
    "object_radius": 20.0,
    "num_objects": {
        "groupMate1" : 1,
        "groupMate2" : 1,
        "groupMate3" : 1,
        "groupMate4" : 1,
        "groupMate5" : 1,
        "groupMate6" : 1,
        "groupMate7" : 1,
        "groupMate8" : 1,
        "groupMate9" : 1,
        "groupMate10" : 1,
        "groupMate11" : 1,
        "groupMate12" : 1,
        "groupMate13" : 1,
        "groupMate14" : 1,
        "groupMate15" : 1,
        "groupMate16" : 1,
        "groupMate17" : 1,
        "groupMate18" : 1
    },
    "column_ID": {
        "groupMate1" : 3,
        "groupMate2" : 5,
        "groupMate3" : 7,
        "groupMate4" : 9,
        "groupMate5" : 11,
        "groupMate6" : 13,
        "groupMate7" : 15,
        "groupMate8" : 17,
        "groupMate9" : 19,
        "groupMate10" : 21,
        "groupMate11" : 23,
        "groupMate12" : 25,
        "groupMate13" : 27,
        "groupMate14" : 29,
        "groupMate15" : 31,
        "groupMate16" : 33,
        "groupMate17" : 35,
        "groupMate18" : 37
        
    },                
    "num_observation_lines" : 32,
    "observation_line_length": 100.,
    "tolerable_distance_to_wall": 50,
    "wall_distance_penalty":  -0.0,
    "delta_v": 0.5,
    'max_rewards': 4,
    "negative_reward":0.0,
    "positive_reward":1,
    "movement_penalty":-0.001,
    "deltaT":120,
    "stopped_distance":0.01
}

#import observed movement data (GPS)
gpsdata = []
with open ('track12h_03_stand_sub.csv', newline='') as csvfile:
    gpsreader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
    next(gpsreader)
    for row in gpsreader:
        gpsdata.append(row)
#2hourTrack, track12h_03_stand, track12h_03_stand_sub
        
gpsdata_validation = []
with open ('track12h_04_stand.csv', newline='') as csvfile: 
    gpsreader_val = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
    next(gpsreader_val)
    for row in gpsreader_val:
        gpsdata_validation.append(row)




print("Validation starting")

#start a new Game
g_validation = MovementGame(current_settings, gpsdata_validation)

#initialize and load new session and graph
tf.reset_default_graph() #resets the tf graph to allow for the definition of a new structure around the restored variables
session_val = tf.InteractiveSession()

#build controller
brain_val = MLP([g_validation.observation_size,], [200, 200, g_validation.num_actions], 
    [tf.tanh, tf.tanh, tf.identity])

optimizer_val = tf.train.RMSPropOptimizer(learning_rate= 0.000, decay=0.0)

#DiscreteDeepQ object
current_controller_val = DiscreteDeepQ(g_validation.observation_size, g_validation.num_actions, brain_val, optimizer_val, session_val,
                                    discount_rate=0.99, exploration_period=0, max_experience=7000, 
                                    store_every_nth=1, train_every_nth=9999999) #i.e. never train
saver_val = tf.train.Saver()
saver_val.restore(session_val, SAVE_DIR)
print(session_val.run(tf.all_variables()))


FPS          = 30
ACTION_EVERY = 1
    
fast_mode = True
if fast_mode:
    WAIT, VISUALIZE_EVERY = False, 20
else:
    WAIT, VISUALIZE_EVERY = True, 1



iterations_val = 1
rewards_val = [None]*iterations_val

for i in range(iterations_val):    
    try:
        #for d in ['/cpu:1', '/cpu:2', '/cpu:3']:
            with tf.device("/gpu:0"):
            #with tf.device(d):
                simulate(simulation=g_validation,
                        controller=current_controller_val,
                        fps=FPS,
                        visualize_every=VISUALIZE_EVERY,
                        action_every=ACTION_EVERY,
                        wait=WAIT,
                        disable_training=True,
                        simulation_resolution=0.1, #0.001
                        save_path="/Users/tylerbonnell/Documents/RL_gif/val",
                        validationStep=True)
    except IndexError: #end of GPS file
        print("Interrupted")
        g_validation.return_to_start()
        rewards_val[i]=g_validation.get_total_rewards()

print("Validation rewards") 
print(rewards_val)

print("Validation completed")
#do_eval(session)
#g.plot_reward(smoothing=100)