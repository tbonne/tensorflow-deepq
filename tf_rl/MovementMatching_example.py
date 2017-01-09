
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


#LOG_DIR = tempfile.mkdtemp()
LOG_DIR = "/tmp/mnist_logs"
print(LOG_DIR)
SAVE_DIR = "/Users/tylerbonnell/Documents/RL_trained_agent/rla_03_DM/DM_model.ckpt"
LOAD_DIR = "/Users/tylerbonnell/Documents/RL_trained_agent/rla_03_DM/DM_model.ckpt"

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
        'groupMate15'
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
        'groupMate15': 'magenta'
    },
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
        "groupMate15" : 1
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
        "groupMate15" : 31
        
    },
    "ID_number": {
        "groupMate1" : 1,
        "groupMate2" : 2,
        "groupMate3" : 3,
        "groupMate4" : 4,
        "groupMate5" : 5,
        "groupMate6" : 6,
        "groupMate7" : 7,
        "groupMate8" : 8,
        "groupMate9" : 9,
        "groupMate10" : 10,
        "groupMate11" : 11,
        "groupMate12" : 12,
        "groupMate13" : 13,
        "groupMate14" : 14,
        "groupMate15" : 15
        
    },
    'world_size': (3200,1550),
    "object_radius": 2.0,                
    'pos_rewards': 0.01,
    'neg_rewards': 0.0,
    "deltaT":1,
    "moveThreshold":0.02,
    "angleThreshold":3.14/8,
    "number_of_closest_neigh":15
}

#import observed movement data (GPS)
gpsdata = []
with open ('track12h_03_stand2.csv', newline='') as csvfile:
    gpsreader = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
    next(gpsreader)
    for row in gpsreader:
        gpsdata.append(row)
#2hourTrack, track12h_03_stand, track12h_03_stand_sub
        
gpsdata_validation = []
with open ('track12h_07_stand2.csv', newline='') as csvfile: 
    gpsreader_val = csv.reader(csvfile, delimiter=",", quoting=csv.QUOTE_NONNUMERIC)
    next(gpsreader_val)
    for row in gpsreader_val:
        gpsdata_validation.append(row)


# create the game simulator
g = MovementGame(current_settings, gpsdata)


human_control = False

if human_control:
    # WSAD CONTROL (requires extra setup - check out README)
    current_controller = HumanController({b"w": 3, b"d": 0, b"s": 1,b"a": 2,}) 
else:
    # Tensorflow business - it is always good to reset a graph before creating a new controller.
    tf.reset_default_graph()
    NUM_CORES = 4
    session = tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=NUM_CORES,
                   intra_op_parallelism_threads=NUM_CORES))

    # This little guy will let us run tensorboard
    #      tensorboard --logdir [LOG_DIR]
    summary_op = tf.merge_all_summaries()
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
                                       discount_rate=0.0, exploration_period=3500, max_experience=7000, 
                                       store_every_nth=1, train_every_nth=1,
                                       summary_writer=journalist)
    
    session.run(tf.initialize_all_variables())
    session.run(current_controller.target_network_update)
    
    # graph was not available when journalist was created  
    journalist.add_graph(session.graph_def)
    saver = tf.train.Saver()
    
    
FPS          = 30
ACTION_EVERY = 1
    
fast_mode = True
if fast_mode:
    WAIT, VISUALIZE_EVERY = False, 20
else:
    WAIT, VISUALIZE_EVERY = True, 1


iterations = 0 #if this is greater than one then we need to fix how the model is being saved
selfT = 0
rewardsList = []
rewards = [None]*iterations

for i in range(iterations):    
    try:
        for d in ['/cpu:0', '/cpu:1', '/cpu:2', '/cpu:3']:
        #   with tf.device("/cpu:0"):
            #with tf.device(d):
                simulate(simulation=g,
                        controller=current_controller,
                        fps=FPS,
                        visualize_every=VISUALIZE_EVERY,
                        action_every=ACTION_EVERY,
                        wait=WAIT,
                        disable_training=False,
                        simulation_resolution=None, #0.001
                        save_path="/Users/tylerbonnell/Documents/RL_gif",
                        validationStep=False)
    except IndexError: #end of GPS file
        print("Interrupted")
        g.return_to_start(i*selfT)
        rewards[i]=g.get_total_rewards()
        rewardsList=g.get_rewardList()
    
    session.run(current_controller.target_network_update)

    current_controller.q_network.input_layer.Ws[0].eval()

    current_controller.target_q_network.input_layer.Ws[0].eval()

    g.plot_reward(smoothing=10)
    
    
    #saver.save(session, LOG_DIR, global_step=i)
    #summary_str = session.run(summary_op, feed_dict=feed_dict)
    #journalist.add_summary(summary_str, i)

    save_path = saver.save(session, SAVE_DIR)
    print("Model saved in file: %s" % save_path) #this should be dropped down (remove tab) when running more than one iteration...

session.close()
print("Training iterations completed")
print("")






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
                                    discount_rate=0.0, exploration_period=0, max_experience=7000, 
                                    store_every_nth=1, train_every_nth=9999999) #i.e. never train
saver_val = tf.train.Saver()
saver_val.restore(session_val, LOAD_DIR)
#print(session_val.run(tf.all_variables()))

iterations_val = 1
rewards_val = [None]*iterations_val

for i in range(iterations_val):    
    try:
        #for d in ['/cpu:1', '/cpu:2', '/cpu:3']:
            with tf.device("/cpu:0"):
            #with tf.device(d):
                simulate(simulation=g_validation,
                        controller=current_controller_val,
                        fps=FPS,
                        visualize_every=VISUALIZE_EVERY,
                        action_every=ACTION_EVERY,
                        wait=WAIT,
                        disable_training=True,
                        simulation_resolution=None, #0.001
                        save_path="/Users/tylerbonnell/Documents/RL_gif/val",
                        validationStep=True)
    except IndexError: #end of GPS file
        print("Interrupted")
        g_validation.return_to_start(i*selfT)
        rewards_val[i]=g_validation.get_total_rewards()
        #xyout = g.get_xylist()
        #if iterations_val - i <= 2:
        xyout = g_validation.get_xylist()
        #else:
            #g.clear_xylist()
        

print("iterations_val = ",i)

if iterations_val > 0:    
    with open('xyout_test.csv', 'w', newline='') as csvfile:
        writerOUT = csv.writer(csvfile, delimiter=' ',
                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for item in xyout:
            writerOUT.writerow([item,])
    
print("Training rewards") 
print(rewardsList)
print(rewards) 
print("Validation rewards") 
print(rewards_val)

print("Validation completed")
#do_eval(session)
#g.plot_reward(smoothing=100)

