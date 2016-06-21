
from __future__ import print_function
import numpy as np
import tempfile
import tensorflow as tf
from controller import DiscreteDeepQ, HumanController
from simulation import KarpathyGame
from tf_rl import simulate
from tf_rl.models import MLP
from tf_rl.simulation.MovementMatching_game import MovementGame
import csv




LOG_DIR = tempfile.mkdtemp()
print(LOG_DIR)

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
        'groupMate18',
        'groupMate19',
        'groupMate20',
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
        'groupMate9': 'magenta',
        'groupMate10': 'magenta',
        'groupMate11': 'magenta',
        'groupMate12': 'magenta',
        'groupMate13': 'magenta',
        'groupMate14': 'magenta',
        'groupMate15': 'magenta',
        'groupMate16': 'magenta',
        'groupMate17': 'magenta',
        'groupMate18': 'magenta',
        'groupMate19': 'magenta',
        'groupMate20': 'magenta',
    },
    'object_reward': {
        'friend': 0.1,
    },
    'min_offset': 10,
    'max_rewards': 5,
    'hero_bounces_off_walls': False,
    'world_size': (1000,1000),
    'hero_initial_position': [826.7389, 761.1064],
    'hero_initial_speed':    [10,   0],
    "maximum_speed":         [50, 50],
    "object_radius": 5.0,
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
        "groupMate18" : 1,
        "groupMate19" : 1,
        "groupMate20" : 1,
        
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
        "groupMate18" : 37,
        "groupMate19" : 39,
        "groupMate20" : 41,
        
    },                
    "num_observation_lines" : 32,
    "observation_line_length": 200.,
    "tolerable_distance_to_wall": 50,
    "wall_distance_penalty":  -0.0,
    "delta_v": 50,
    "negative_reward":-0.1,
    "positive_reward":0.6,
    "deltaT":120
}

#import observed movement data (GPS)
gpsdata = []
with open ('2hourTrack.csv', newline='') as csvfile:
    gpsreader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
    next(gpsreader)
    for row in gpsreader:
        gpsdata.append(row)

#print(gpsdata[0][1]) #time step, then the column number

# create the game simulator
g = MovementGame(current_settings, gpsdata)

human_control = False

if human_control:
    # WSAD CONTROL (requires extra setup - check out README)
    current_controller = HumanController({b"w": 3, b"d": 0, b"s": 1,b"a": 2,}) 
else:
    # Tensorflow business - it is always good to reset a graph before creating a new controller.
    tf.reset_default_graph()
    session = tf.InteractiveSession()

    # This little guy will let us run tensorboard
    #      tensorboard --logdir [LOG_DIR]
    journalist = tf.train.SummaryWriter(LOG_DIR)

    # Brain maps from observation to Q values for different actions.
    # Here it is a done using a multi layer perceptron with 2 hidden
    # layers
    brain = MLP([g.observation_size,], [200, 200, g.num_actions], 
                [tf.tanh, tf.tanh, tf.identity])
    
    # The optimizer to use. Here we use RMSProp as recommended
    # by the publication
    optimizer = tf.train.RMSPropOptimizer(learning_rate= 0.001, decay=0.9)

    # DiscreteDeepQ object
    current_controller = DiscreteDeepQ(g.observation_size, g.num_actions, brain, optimizer, session,
                                       discount_rate=0.99, exploration_period=5000, max_experience=10000, 
                                       store_every_nth=4, train_every_nth=4,
                                       summary_writer=journalist)
    
    session.run(tf.initialize_all_variables())
    session.run(current_controller.target_network_update)
    # graph was not available when journalist was created  
    journalist.add_graph(session.graph_def)
    
FPS          = 30
ACTION_EVERY = 3
    
fast_mode = False
if fast_mode:
    WAIT, VISUALIZE_EVERY = False, 20
else:
    WAIT, VISUALIZE_EVERY = True, 1

    
try:
    with tf.device("/cpu:0"):
        simulate(simulation=g,
                 controller=current_controller,
                 fps=FPS,
                 visualize_every=VISUALIZE_EVERY,
                 action_every=ACTION_EVERY,
                 wait=WAIT,
                 disable_training=False,
                 simulation_resolution=0.1, #0.001
                 save_path="/Users/tylerbonnell/Documents/gitRepro/tensorflow-deepq/data/testData")
except KeyboardInterrupt:
    print("Interrupted")
    
session.run(current_controller.target_network_update)

current_controller.q_network.input_layer.Ws[0].eval()

current_controller.target_q_network.input_layer.Ws[0].eval()

g.plot_reward(smoothing=100)


print("test completed")