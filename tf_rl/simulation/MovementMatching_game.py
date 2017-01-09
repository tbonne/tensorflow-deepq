import math
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import euclid
import csv

from collections import defaultdict
from euclid import Circle, Point2, Vector2, LineSegment2

import tf_rl.utils.svg as svg

class GameObject(object):
    def __init__(self, position, direction, obj_type, settings, colID,idNumb):
        """Esentially represents circles of different kinds, which have
        position and directions of travel."""
        self.settings = settings
        self.radius = self.settings["object_radius"]
        self.obj_type = obj_type
        self.position = position
        self.direction    = direction
        self.distance = 0
        #self.obsSpeed = Vector2(0,0)
        #self.bounciness = 1.0
        self.colID = colID
        self.idNumb = idNumb
        self.timeStep=self.settings["deltaT"];
        self.deltaT = self.settings["deltaT"];

    
    def update_position_and_direction(self,GPS,timeS):
        self.direction = self.unit_vector(Vector2(GPS[timeS][self.colID]-GPS[timeS-self.deltaT][self.colID],GPS[timeS][self.colID+1]-GPS[timeS-self.deltaT][self.colID+1]))
        newPos = Point2(GPS[timeS][self.colID],GPS[timeS][self.colID+1])
        self.distance = self.calculate_distance(self.position,newPos)
        self.position = newPos
    
    
    def step(self, dt,GPS, timeS):
        """Update position and direction of travel based on GPS data"""
        #if self.idNumb == 0:
            #print("performing agent update now: ")
        self.update_position_and_direction(GPS,timeS)
        
        
    def as_circle(self):
        return Circle(self.position, float(self.radius))

    def draw(self):
        """Return svg object for this item."""
        color = self.settings["colors"][self.obj_type]
        return svg.Circle(self.position + Point2(10, 10), self.radius, color=color)
    
    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        #return vector / np.linalg.norm(vector)
        vecU = [0,0]
        try:
            vecU = Vector2(vector[0]/vector.magnitude(),vector[1]/vector.magnitude())
        except ZeroDivisionError:
            print ()#"division by zero")
            
        return vecU 

    def calculate_distance(self, p1, p2):
        return ( (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


class MovementGame(object):
    def __init__(self, settings,gpsFile):
        """Initiallize game simulator with settings"""
        self.settings = settings
        self.GPS = gpsFile
        self.timeStep = self.settings["deltaT"]
        self.previousOffset = 0
        self.size  = self.settings["world_size"]
        self.hero = GameObject(Point2(self.GPS[0][0],self.GPS[0][1]),
                               self.unit_vector(Vector2(self.GPS[self.timeStep][0]-self.GPS[0][0],self.GPS[self.timeStep][1]-self.GPS[0][1])),
                               "hero",
                               self.settings,0,0)
        self.objects = []
        count = 0
        id=0;
        for obj_type, number in settings["num_objects"].items():
            count=count+2
            id = id+1
            for _ in range(number):
                self.spawn_object(obj_type,count,id,gpsFile,self.timeStep)
                
                
        #self.observation_lines = self.generate_observation_lines()
        self.object_reward = 0
        self.collected_rewards = []
        self.xylist = []
        self.hero.xdist = []
        self.hero.ydist = []
        self.hero.xypos = Vector2(-1,-1)
        self.hero.prediction = [0,0]
        self.count = 0
        self.rewardCounter = 0
        self.rewardList = []
 
        # observation size: 6 for each group member, additionally there are two numbers representing agents own speed and position, and one for the distance traveled during the step.
        self.observation_size = 6 * len(self.settings["objects"]) + 3 #+ 2 

        self.actions = [Vector2(*d) for d in [[1,0], [0,1], [-1,0], [0,-1], [0.0,0.0], [-1,-1], [1,1],[-1,1], [1,-1]]]
        #self.actions = [Vector2(*d) for d in [[1,0],[0.707,0.707], [0,1],[-0.707,0.707], [-1,0],[-0.707,-0.707],[0,-1],[0.707,-0.707],[0.0,0.0]]]
        self.num_actions = len(self.actions)

    def spawn_object(self, obj_type, colID, idNumb,GPS,timeStep):
        """Spawn object of a given type and add it to the objects array"""
        position  = Point2(GPS[1][colID],GPS[0][colID+1])
        direction     = self.unit_vector(Vector2(GPS[timeStep][colID]-GPS[0][colID],GPS[timeStep][colID+1]-GPS[0][colID+1]))
        self.objects.append(GameObject(position, direction, obj_type, self.settings,colID,idNumb))    

    def perform_action(self, action_id):
        """Change direction of travel of hero"""
        assert 0 <= action_id < self.num_actions
        
        #update my travel direction based on controller (Speed based: direction and magnitude)
        #print("3: performing action now: ")
        self.hero.prediction = self.actions[action_id]
                    
        #update my travel based on controller (Speed based: direction and magnitude)
        #self.hero.speed *= (1-self.settings["friction"]) #tendency to slow down (friction/effort)
        #self.hero.speed += self.directions[action_id] * self.settings["delta_v"]
        #self.hero.xdist.append(self.hero.speed[0])
        #self.hero.ydist.append(self.hero.speed[1])
        
        #update my travel direction randomly
        #action_id = random.randint(0, self.num_actions - 1) #used to quantify the range reward accumulation due to random chance
        #self.hero.prediction = self.actions[action_id]
       
        #print(action_id, end="")     

    def step(self, dt):
        """Simulate all the objects for a given amount of time"""
        #print("4: stepping agents now: ")
        
        for obj in self.objects + [self.hero] :
            obj.step(dt, self.GPS, self.timeStep)

        #keep track of time
        self.timeStep=self.timeStep+self.settings["deltaT"]
        
        #monitor rewards
        self.count+=1
        if(self.count>1000):
            self.rewardList.append(self.rewardCounter)
            self.rewardCounter=0
            self.count=0
            
        print(self.timeStep)
        

    def directionMatching(self):
        """Reward function based on match with observed direction of travel"""
        #print("2: performing rewards now: ")
        currentRewards=0;
        
        #traveled greater than cutoff
        if self.hero.distance < self.settings["moveThreshold"]:
            #reward if action predicted not to move
            if self.hero.prediction == [0,0]:
                self.object_reward += self.settings["pos_rewards"]
                currentRewards=self.settings["pos_rewards"]
        else:
            #reward if action predicted is within X of the observed travel direction
            if self.angle_between1(self.hero.prediction, self.hero.direction) < self.settings["angleThreshold"]:
                self.object_reward += self.settings["pos_rewards"]
                currentRewards=self.settings["pos_rewards"]
        
        #keep track of rewards (diagnostics)
        self.rewardCounter += currentRewards 
        
        #record: x,y, obs direction, pred direction
        self.xylist.append([self.GPS[self.timeStep-self.settings["deltaT"]][0], self.GPS[self.timeStep-self.settings["deltaT"]][1], self.hero.direction[0],self.hero.direction[1],self.hero.prediction[0],self.hero.prediction[1],self.hero.distance,currentRewards] )       
                
    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        #return vector / np.linalg.norm(vector)
        return Vector2(vector[0]/vector.magnitude(),vector[1]/vector.magnitude()) 

    def angle_between2(self, v1, v2):
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    def angle_between1(self, v1, v2):
        v1_u = v1
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    def distance_between(self, v1, v2):
        hypo = 0
        if (v2[0]-v1[0]==0 and v2[1]-v1[1]==0):
            hypo = 0
        else:
            hypo = math.hypot(v2[0]-v1[0], v2[1]-v1[1])
        return hypo
    

    def observe(self):
        """Return observation vector. Returns relative positions, direction of travel, and 
        proximity of the closest X objects to the hero. 
        """
        
        #print("1: performing observation now: ",self.timeStep)
        num_obj_types = len(self.settings["objects"]) 
        #max_speed_x, max_speed_y = self.settings["maximum_speed"]

        relevant_objects = [obj for obj in self.objects
                            if obj.idNumb != self.hero.idNumb]
        # objects sorted from closest to furthest
        relevant_objects.sort(key=lambda x: x.position.distance(self.hero.position))

        #initialize all observations to 0
        observation        = np.zeros(self.observation_size)
        observation_offset = 0
        
        #consider only the x closest neighbours
        for i in range(self.settings["number_of_closest_neigh"]):
            
            #get neighbour attributes
            #object_type_id = self.settings["objects"].index(relevant_objects[i].obj_type)
            dir_x, dir_y = tuple(relevant_objects[i].direction)
            relative_position_x = relevant_objects[i].position[0]-self.hero.position[0]
            relative_position_y = relevant_objects[i].position[1]-self.hero.position[1]
            rel_vec = self.unit_vector(Vector2(relative_position_x,relative_position_y))
            rel_x = rel_vec[0]
            rel_y = rel_vec[1]
            proximity = ( 1 / (1 + (math.pow( math.pow(relative_position_x,2) + math.pow(relative_position_y,2),0.5)/10) ) ) #10 meters is where proximity reaches 0.5 
            if proximity > 1: 
                proximity=1
            
            
            #record attributes in observation
            observation_offset = (relevant_objects[i].idNumb-1)*6
            observation[observation_offset]     = 1.0                                 #individual is one of the X closest
            observation[observation_offset + 1] = proximity                           #individual is X distance away
            observation[observation_offset + 2] = rel_x                               #individual is moving in the x direction by X
            observation[observation_offset + 3] = rel_y                               #individual is moving in the y direction by X
            observation[observation_offset + 4] = dir_x                               #individual is moving in the x direction by X
            observation[observation_offset + 5] = dir_y                               #individual is moving in the y direction by X
            observation_offset=0
            #assert num_obj_types + 2 == self.eye_observation_size
            #observation_offset += self.eye_observation_size
            

        #record hero attributes
        observation_offset = ((num_obj_types-1)*6)+6
        #observation[observation_offset]     = self.hero.speed[0]                        #this is my predicted speed (action taken by the agent) at this time point
        #observation[observation_offset + 1] = self.hero.speed[1] 
        observation[observation_offset]      = self.hero.direction[0]                     #this is the observed speed leading to this time point (previous direction of travel)
        observation[observation_offset + 1]  = self.hero.direction[1] 
        #observation[observation_offset + 2] = self.hero.position[0] / 1600.0 - 1.0      #this is the location of the animal normalized to the extent of the landscape
        #observation[observation_offset + 3] = self.hero.position[1] / 775.0 - 1.0
        observation[observation_offset + 2] = ( 1 / (1 + self.hero.distance/10) )
        observation_offset += 3
        #print(observation)
        
        assert observation_offset == self.observation_size

        return observation


    def collect_reward(self):
        """Return accumulated object eating score + current distance to walls score"""
        #wall_reward =  self.settings["wall_distance_penalty"] * \
        #               np.exp(-self.distance_to_walls() / self.settings["tolerable_distance_to_wall"])
        #assert wall_reward < 1e-3, "You are rewarding hero for being close to the wall!"
        self.directionMatching() #reward for being close to focal animal location 
        total_reward =  self.object_reward  #wall_reward +
        self.object_reward = 0
        self.collected_rewards.append(total_reward)
        return total_reward

    def plot_reward(self, smoothing = 30):
        """Plot evolution of reward over time."""
        plottable = self.collected_rewards[:]
        while len(plottable) > 1000:
            for i in range(0, len(plottable) - 1, 2):
                plottable[i//2] = (plottable[i] + plottable[i+1]) / 2
            plottable = plottable[:(len(plottable) // 2)]
        x = []
        for  i in range(smoothing, len(plottable)):
            chunk = plottable[i-smoothing:i]
            x.append(sum(chunk) / len(chunk))
        plt.plot(list(range(len(x))), x)
        print(x)
        
    def get_total_rewards(self):
        sumL = sum(self.collected_rewards) 
        del self.collected_rewards[:] #reinitialize the rewards list
        return sumL
    
    def get_xylist(self):
        return self.xylist
    
    def get_rewardList(self):
        return self.rewardList
    
    def clear_xylist(self):
        del self.xylist[:]
    
    def return_to_start(self,iteration):
        self.timeStep=self.settings["deltaT"]+iteration
        self.hero.update_position_and_direction(self.GPS, self.timeStep+iteration)

    def _repr_html_(self):
        return self.to_html()

    def to_html(self, stats=[]):
        """Return svg representation of the simulator"""

        stats = stats[:]
        recent_reward = self.collected_rewards[-100:] + [0]
        #objects_eaten_str = ', '.join(["%s: %s" % (o,c) for o,c in self.objects_eaten.items()])
        stats.extend([
            #"nearest wall = %.1f" % (self.distance_to_walls(),),
            "reward       = %.1f" % (sum(recent_reward)/len(recent_reward),),
            #"objects eaten => %s" % (objects_eaten_str,),
        ])

        scene = svg.Scene((self.size[0] + 20, self.size[1] + 20 + 20 * len(stats)))
        scene.add(svg.Rectangle((10, 10), self.size))


        #for line in self.observation_lines:
        #    scene.add(svg.Line(line.p1 + self.hero.position + Point2(10,10),
        #                       line.p2 + self.hero.position + Point2(10,10)))

        for obj in self.objects + [self.hero] :
            scene.add(obj.draw())

        offset = self.size[1] + 15
        for txt in stats:
            scene.add(svg.Text((10, offset + 20), txt, 15))
            offset += 20

        return scene

