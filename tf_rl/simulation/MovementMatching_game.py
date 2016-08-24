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
    def __init__(self, position, speed, obj_type, settings, colID):
        """Esentially represents circles of different kinds, which have
        position and speed."""
        self.settings = settings
        self.radius = self.settings["object_radius"]

        self.obj_type = obj_type
        self.position = position
        self.speed    = speed
        self.obsSpeed = Vector2(0,0)
        self.bounciness = 1.0
        self.colID = colID

    
    def update_position(self,GPS,timeS):
        self.position = Point2(GPS[timeS][self.colID],GPS[timeS][self.colID+1])
    
    def startConditions(self):
        self.timeStep=0;

    def step(self, dt,GPS, timeS):
        #"""Move and bounce of walls."""
        #self.wall_collisions()
        
        #group mates move as observed, agent attempts to match focal animal movements
        #if(self.colID==0):
        #self.move(dt)
        #else:
        """Update position based on GPS data"""
        self.update_position(GPS,timeS)
        
    def as_circle(self):
        return Circle(self.position, float(self.radius))

    def draw(self):
        """Return svg object for this item."""
        color = self.settings["colors"][self.obj_type]
        return svg.Circle(self.position + Point2(10, 10), self.radius, color=color)

class MovementGame(object):
    def __init__(self, settings,gpsFile):
        """Initiallize game simulator with settings"""
        self.settings = settings
        self.GPS = gpsFile
        self.timeStep = 0
        self.previousOffset = 0
        self.size  = self.settings["world_size"]
        self.walls = [LineSegment2(Point2(0,0),                        Point2(0,self.size[1])),
                      LineSegment2(Point2(0,self.size[1]),             Point2(self.size[0], self.size[1])),
                      LineSegment2(Point2(self.size[0], self.size[1]), Point2(self.size[0], 0)),
                      LineSegment2(Point2(self.size[0], 0),            Point2(0,0))]

        self.hero = GameObject(Point2(*self.settings["hero_initial_position"]),
                               Vector2(*self.settings["hero_initial_speed"]),
                               "hero",
                               self.settings,0)
        if not self.settings["hero_bounces_off_walls"]:
            self.hero.bounciness = 0.0

        self.objects = []
        count = 0
        for obj_type, number in settings["num_objects"].items():
            count=count+2
            for _ in range(number):
                self.spawn_object(obj_type,count)
                
                
        self.observation_lines = self.generate_observation_lines()

        self.object_reward = 0
        self.collected_rewards = []
        self.xylist = []
        self.hero.xdist = []
        self.hero.ydist = []
        self.hero.xypos = Vector2(-1,-1)
        self.hero.lastRewards = 0
 
        # every observation_line sees one of objects or wall and
        # two numbers representing speed of the object (if applicable)
        self.eye_observation_size = len(self.settings["objects"]) + 3
        # additionally there are two numbers representing agents own speed and position.
        self.observation_size = self.eye_observation_size * len(self.observation_lines) + 2 + 2 + 2

        self.directions = [Vector2(*d) for d in [[1,0], [0,1], [-1,0], [0,-1], [0.0,0.0], [-1,-1], [1,1]]]
        #self.directions = [Vector2(*d) for d in [[1,0],[0.707,0.707], [0,1],[-0.707,0.707], [-1,0],[-0.707,-0.707],[0,-1],[0.707,-0.707],[0.0,0.0]]]
        #self.directions = [Vector2(*d) for d in [[1],[-1], [0]]]
        self.num_actions      = len(self.directions)

        self.objects_eaten = defaultdict(lambda: 0)
        
    def spawn_object(self, obj_type, colID):
        """Spawn object of a given type and add it to the objects array"""
        radius = self.settings["object_radius"]
        position = np.random.uniform([radius, radius], np.array(self.size) - radius)
        position = Point2(float(position[0]), float(position[1]))
        max_speed = np.array(self.settings["maximum_speed"])
        speed    = np.random.uniform(-max_speed, max_speed).astype(float)
        speed = Vector2(float(speed[0]), float(speed[1]))

        self.objects.append(GameObject(position, speed, obj_type, self.settings,colID))    

    def perform_action(self, action_id):
        """Change speed to one of hero vectors"""
        assert 0 <= action_id < self.num_actions
        
        #update my travel direction based on controller
        self.hero.speed *= (1-self.settings["friction"]) #tendency to slow down (friction/effort)
        self.hero.speed += self.directions[action_id]*self.settings["delta_v"]
        self.hero.xdist.append(self.hero.speed[0])
        self.hero.ydist.append(self.hero.speed[1])
        
        #update my travel direction randomly
        #action_id = random.randint(0, self.num_actions - 1) #used to quantify the range reward accumulation due to random chance
                    
        #record: x,y, obs direction, pred direction
        self.xylist.append([self.GPS[self.timeStep][0], self.GPS[self.timeStep][1], self.hero.obsSpeed[0],self.hero.obsSpeed[1],self.hero.speed[0],self.hero.speed[1]] )            

    def step(self, dt):
        """Simulate all the objects for a given amount of time.
        Also resolve collisions with the hero"""
        for obj in self.objects + [self.hero] :
            obj.step(dt, self.GPS, self.timeStep)

        #record observed travel of hero, update reference xypos to be used when updating rewards
        obs_t0_hero = [self.GPS[self.timeStep][0],self.GPS[self.timeStep][1]]
        obs_t1_hero = [self.GPS[self.timeStep-1][0],self.GPS[self.timeStep-1][1]]
        self.hero.obsSpeed = Vector2(obs_t1_hero[0]-obs_t0_hero[0],obs_t1_hero[1]-obs_t0_hero[1])
        if self.hero.xypos[0]==-1 and self.hero.xypos[0] == -1:
                self.hero.xypos = obs_t0_hero
                self.hero.lastRewards = 0    

        #keep track of time
        self.timeStep=self.timeStep+1
        self.hero.lastRewards+=1
        print(self.timeStep)
        

    def matchingMovements(self):
        """new reward function based on match with observed travel"""
        
        if self.hero.lastRewards==self.settings["deltaT"]:
            pos_x=self.hero.xypos[0]
            pos_y=self.hero.xypos[1]
            self.hero.xypos[0]=-1
            self.hero.xypos[1]=-1
        
            magnitude_dist = 0
        
            #predicted position
            if self.hero.xdist.__sizeof__()>0:
                for item in self.hero.xdist:
                    pos_x += item
                for item in self.hero.ydist:
                    pos_y += item
                
                del self.hero.xdist[:]
                del self.hero.ydist[:]
        
                #current position
                obs_t0 = [self.GPS[self.timeStep][0],self.GPS[self.timeStep][1]]
        
                #difference between the current and predicted positions
                magnitude_dist = pow( (pow(pos_x-obs_t0[0],2)+pow(pos_y-obs_t0[1],2)),0.5) #given speed over the same amount of time
        
                if magnitude_dist < self.settings["withinR"]:
                    self.object_reward += self.settings["max_rewards"]*pow(1-magnitude_dist/self.settings["withinR"],2)
        
        
            print("Total offset in distance = ", magnitude_dist,"  --  Rewards given ",self.object_reward)
            print("Obs = ", obs_t0,"  pred = [",pos_x, ", ",pos_y,"]")
                        
        """self.xypos = Vector2(0,0)
        
        
        diff_angle = 9999
        #sim = [self.hero.position[0], self.hero.position[1]]
        obs_t0 = [self.GPS[self.timeStep][0],self.GPS[self.timeStep][1]]
        obs_t1 = [self.GPS[self.timeStep+1][0],self.GPS[self.timeStep+1][1]]
        obsSpeed_plusOne = Vector2(obs_t1[0]-obs_t0[0],obs_t1[1]-obs_t0[1])
        pred_direction = self.hero.speed
        
        #if self.hero.speed.magnitude()>self.settings["stopped_distance"]:
        #    pred_direction = self.unit_vector(self.hero.speed)
        #else:
        #    pred_direction = self.hero.speed*0
        
        #rewards based on distance between vectors (allows for magnitude of speed to play a role, as opposed to just their angles)
        #dist = self.distance_between(obs_direction,pred_direction)
        #if dist< self.settings["min_offset"]:
        #    self.object_reward += self.settings["max_rewards"] * (1 - (dist / self.settings["min_offset"]))
        
        if obsSpeed_plusOne.magnitude()<self.settings["stopped_distance"]:
            if pred_direction[0]==0 and pred_direction[1]==0:
                self.object_reward += self.settings["positive_reward"]
            else:
                self.object_reward += self.settings["negative_reward"]
        else:
            if pred_direction.magnitude()>self.settings["stopped_distance"]:
                diff_angle = abs(self.angle_between2(obsSpeed_plusOne, pred_direction))
                
        #reward based on offset from focal animals actual path
        #if sim[0]==0. and sim[1]==0.:
        #   nothing        
        #else: 
        #    totalOffset = self.distance_between(sim, obs) 
        
        if diff_angle< self.settings["min_offset"]:
            self.object_reward += self.settings["max_rewards"] * (1 - (diff_angle / self.settings["min_offset"]))
        else:
            self.object_reward += self.settings["negative_reward"]
            
        #if diff_angle < self.previousOffset:
        #    self.object_reward += self.settings["positive_reward"]
        
        #speedNow = [self.hero.speed[0], self.hero.speed[1]]
        #velocityNow = self.total_speed(speedNow)
        #self.object_reward += velocityNow*self.settings["movement_penalty"]
        
        
        #update hero to have the same speed and position as the observed data
        #if(self.timeStep>1 and self.deltaT>self.settings["deltaT"]):
        #   self.hero.position = Point2(obs[0],obs[1])
        #    self.hero.speed = Vector2(self.GPS[self.timeStep][0]-self.GPS[self.timeStep-1][0], self.GPS[self.timeStep][1]-self.GPS[self.timeStep-1][1])
        #   self.deltaT=0
        """
                
    def squared_distance(self, p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    
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
    
    def total_speed(self, v1):
        hypo = 0
        if (v1[0]-0==0 and v1[1]-0==0):
            hypo = 0
        else:
            hypo = math.hypot(v1[0]-0, v1[1]-0)
        return hypo
    
    def resolve_collisions(self):
        
        """If hero touches, hero eats. Also reward gets updated."""
        #collision_distance = 2 * self.settings["object_radius"]
        #collision_distance2 = collision_distance ** 2
        #to_remove = []
        #for obj in self.objects:
        #    if self.squared_distance(self.hero.position, obj.position) < collision_distance2:
        #        to_remove.append(obj)
        #for obj in to_remove:
        #    self.objects.remove(obj)
        #    self.objects_eaten[obj.obj_type] += 1
        #    self.object_reward += self.settings["object_reward"][obj.obj_type]
        #    self.spawn_object(obj.obj_type)

    def inside_walls(self, point):
        """Check if the point is inside the walls"""
        EPS = 1e-4
        return (EPS <= point[0] < self.size[0] - EPS and
                EPS <= point[1] < self.size[1] - EPS)

    def observe(self):
        """Return observation vector. For all the observation directions it returns representation
        of the closest object to the hero - might be nothing, another object or a wall.
        Representation of observation for all the directions will be concatenated.
        """
        num_obj_types = len(self.settings["objects"]) + 1 # and wall
        max_speed_x, max_speed_y = self.settings["maximum_speed"]

        observable_distance = self.settings["observation_line_length"]

        relevant_objects = [obj for obj in self.objects
                            if obj.position.distance(self.hero.position) < observable_distance]
        # objects sorted from closest to furthest
        relevant_objects.sort(key=lambda x: x.position.distance(self.hero.position))

        observation        = np.zeros(self.observation_size)
        observation_offset = 0
        for i, observation_line in enumerate(self.observation_lines):
            # shift to hero position
            observation_line = LineSegment2(self.hero.position + Vector2(*observation_line.p1),
                                            self.hero.position + Vector2(*observation_line.p2))

            observed_object = None
            # if end of observation line is outside of walls, we see the wall.
            if not self.inside_walls(observation_line.p2):
                observed_object = "**wall**"
            for obj in relevant_objects:
                if observation_line.distance(obj.position) < self.settings["object_radius"]:
                    observed_object = obj
                    break
            object_type_id = None
            speed_x, speed_y = 0, 0
            proximity = 0
            if observed_object == "**wall**": # wall seen
                object_type_id = num_obj_types - 1
                # a wall has fairly low speed...
                speed_x, speed_y = 0, 0
                # best candidate is intersection between
                # observation_line and a wall, that's
                # closest to the hero
                best_candidate = None
                for wall in self.walls:
                    candidate = observation_line.intersect(wall)
                    if candidate is not None:
                        if (best_candidate is None or
                                best_candidate.distance(self.hero.position) >
                                candidate.distance(self.hero.position)):
                            best_candidate = candidate
                if best_candidate is None:
                    # assume it is due to rounding errors
                    # and wall is barely touching observation line
                    proximity = observable_distance
                else:
                    proximity = best_candidate.distance(self.hero.position)
            elif observed_object is not None: # agent seen
                object_type_id = self.settings["objects"].index(observed_object.obj_type)
                speed_x, speed_y = tuple(observed_object.speed)
                intersection_segment = obj.as_circle().intersect(observation_line)
                assert intersection_segment is not None
                try:
                    proximity = min(intersection_segment.p1.distance(self.hero.position),
                                    intersection_segment.p2.distance(self.hero.position))
                except AttributeError:
                    proximity = observable_distance
            for object_type_idx_loop in range(num_obj_types):
                observation[observation_offset + object_type_idx_loop] = 1.0
            if object_type_id is not None:
                observation[observation_offset + object_type_id] = proximity / observable_distance
            observation[observation_offset + num_obj_types] =     speed_x   / max_speed_x
            observation[observation_offset + num_obj_types + 1] = speed_y   / max_speed_y
            assert num_obj_types + 2 == self.eye_observation_size
            observation_offset += self.eye_observation_size

        observation[observation_offset]     = self.hero.speed[0]     #this is my predicted speed (action taken by the agent) at this time point
        observation[observation_offset + 1] = self.hero.speed[1] 
        observation[observation_offset + 2] = self.hero.obsSpeed[0]  #this is the observed speed leading to this time point (previous direction of travel)
        observation[observation_offset + 3] = self.hero.obsSpeed[1] 
        observation_offset += 4
        
        # add normalized location of the hero in environment        
        observation[observation_offset]     = self.hero.position[0] / 350.0 - 1.0
        observation[observation_offset + 1] = self.hero.position[1] / 250.0 - 1.0
        
        assert observation_offset + 2 == self.observation_size

        return observation

    def distance_to_walls(self):
        """Returns distance of a hero to walls"""
        res = float('inf')
        for wall in self.walls:
            res = min(res, self.hero.position.distance(wall))
        return res - self.settings["object_radius"]

    def collect_reward(self):
        """Return accumulated object eating score + current distance to walls score"""
        #wall_reward =  self.settings["wall_distance_penalty"] * \
        #               np.exp(-self.distance_to_walls() / self.settings["tolerable_distance_to_wall"])
        #assert wall_reward < 1e-3, "You are rewarding hero for being close to the wall!"
        self.matchingMovements() #reward for being close to focal animal location 
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
    
    def clear_xylist(self):
        del self.xylist[:]
    
    def return_to_start(self):
        self.timeStep=0
        self.hero.update_position(self.GPS, 0)

    def generate_observation_lines(self):
        """Generate observation segments in settings["num_observation_lines"] directions"""
        result = []
        start = Point2(0.0, 0.0)
        end   = Point2(self.settings["observation_line_length"],
                       self.settings["observation_line_length"])
        for angle in np.linspace(0, 2*np.pi, self.settings["num_observation_lines"], endpoint=False):
            rotation = Point2(math.cos(angle), math.sin(angle))
            current_start = Point2(start[0] * rotation[0], start[1] * rotation[1])
            current_end   = Point2(end[0]   * rotation[0], end[1]   * rotation[1])
            result.append( LineSegment2(current_start, current_end))
        return result

    def _repr_html_(self):
        return self.to_html()

    def to_html(self, stats=[]):
        """Return svg representation of the simulator"""

        stats = stats[:]
        recent_reward = self.collected_rewards[-100:] + [0]
        objects_eaten_str = ', '.join(["%s: %s" % (o,c) for o,c in self.objects_eaten.items()])
        stats.extend([
            "nearest wall = %.1f" % (self.distance_to_walls(),),
            "reward       = %.1f" % (sum(recent_reward)/len(recent_reward),),
            "objects eaten => %s" % (objects_eaten_str,),
        ])

        scene = svg.Scene((self.size[0] + 20, self.size[1] + 20 + 20 * len(stats)))
        scene.add(svg.Rectangle((10, 10), self.size))


        for line in self.observation_lines:
            scene.add(svg.Line(line.p1 + self.hero.position + Point2(10,10),
                               line.p2 + self.hero.position + Point2(10,10)))

        for obj in self.objects + [self.hero] :
            scene.add(obj.draw())

        offset = self.size[1] + 15
        for txt in stats:
            scene.add(svg.Text((10, offset + 20), txt, 15))
            offset += 20

        return scene

