# Reinforcement Learning

# Here we have a simple agent in a simple world.  It has three actions
# (go forward, turn left, and turn right), and its only sense are three

# range finders (radar).



# Initially, its basal ganglia action selection system is set up to have

# fixed utilities for the three actions (moving forward has a utility of 0.8,

# which is larger than turning left or right (0.7 and 0.6), so it should always

# go forward.



# The reward system is set up to give a positive value when moving forward, and

# a negative value if the agent crashes into a wall.  In theory, this should

# cause it to learn to avoid obstacles.  In particular, it will start to turn

# before it hits the obstacle, and so the reward should be negative much less

# often.



# The error signal in this case is very simple: the difference between the

# computed utility and the instantaneous reward.  This error signal should

# only be applied to whatever action is currently being chosen (although it

# isn't quite perfect at doing this).  Note that this means it cannot learn

# to do actions that will lead to *future* rewards.



import logging

import time

import random



import nengo

import numpy as np

from grid import Cell as GridCell

from grid import ContinuousAgent, GridNode

from grid import World as GridWorld



from nengo_fpga.networks import FpgaPesEnsembleNetwork



# Note: Requires the "keyboard_state" branch of nengo_gui for full

#       interactive functionality





# Set the nengo logging level to 'info' to display all of the information

# coming back over the ssh connection.

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)



# ---------------- BOARD SELECT ----------------------- #

# Change this to your desired device name

board = "de1"

# ---------------- BOARD SELECT ----------------------- #





# ----------- WORLD CONFIGURATION ---------------------------------------------

class Cell(GridCell):

    def color(self):
        if self.wall:
            return "black"
        elif self.destination:
            return "pink"
        else:
            return None

    def load(self, char):
        if char == "#":
            self.wall = True
        elif char == "?":
            #self.destination_x = int(self.x)
            #self.destination_y = int(self.y)
            global temp
            temp = [self.x, self.y]
            self.destination = True
        else:
            self.destination = False
            self.wall = False



class WorldConfig:

    curr_ind = -1

    world_maps = [
"""
#################
#---------------#
#--?------------#
#---------------#
#---------------#
#---------------#
#---------------#
#---------------#
#---------------#
#---------------#
#---------------#
#---------------#
#---------------#
#################
""",

"""
#################
#---------------#
###############-#
#---------------#
#-###############
#---------------#
###############-#
#---------------#
#-###############
#---------------#
###############-#
#---------------#
#-?-------------#
#################""",
"""
#################
##############--#
############---##
###########---###
##########---####
#########---#####
########-?-######
#######---#######
######---########
#####---#########
####---##########
###---###########
##---############
#################""",
"""
#################
#---------------#
#-----###-------#
#-----###-------#
#---------------#
#---------------#
#######---#######
#---------------#
#---------------#
#-----###-------#
#-----###-------#
#-----###----?--#
#---------------#
#################"""
    ]

    init_pos = [(1, 3, 2), (1, 1, 2), (2, 12, 1), (1, 1, 1)]



    world = None

    agent = None



    def get_init_pos(self):

        return self.init_pos[self.curr_ind]



    def get_map(self):

        return self.world_maps[self.curr_ind]



    def set_ind(self, new_ind):

        if 0 <= new_ind < len(self.world_maps):

            self.curr_ind = new_ind

            lines = self.get_map().splitlines()



            if (len(lines[0])) == 0:

                del lines[0]

            lines = [x.rstrip() for x in lines]

            for j, _ in enumerate(lines):

                for i, _ in enumerate(lines[0]):

                    self.world.get_cell(i, j).load(lines[j][i])



    def reset_pos(self):

        self.agent.x = self.get_init_pos()[0]

        self.agent.y = self.get_init_pos()[1]

        self.agent.dir = self.get_init_pos()[2]

        self.agent.cell = self.world.get_cell(self.agent.x, self.agent.y)





world_cfg = WorldConfig()

world = GridWorld(Cell, map=world_cfg.get_map(), directions=4)

agent = ContinuousAgent()

world_cfg.world = world

world_cfg.agent = agent

world_cfg.set_ind(0)

world_cfg.world.add(

    agent,

    x=world_cfg.get_init_pos()[0],

    y=world_cfg.get_init_pos()[1],

    dir=world_cfg.get_init_pos()[2],

)



# ----------- LEARNING & MODEL PARAMETERS -------------------------------------

learn_rate = 1e-4

learn_synapse = 0.030

learn_timeout = 5000

radar_dim = 5

turn_bias = 0.25

action_threshold = 0.1

init_transform = [0.8, 0.6, 0.7]



# ----------- MODEL SEED CONFIGURATION ----------------------------------------

seed = int(time.time())

print("USING SEED: {0}".format(seed))



# ----------- MODEL PROPER ----------------------------------------------------

if "__page__" in locals():

    # Additional options for keyboard-state branch of nengo_gui

    # Allows the user to control the status of the learning, and to change the

    # map being used by the agent.

    print("Press 'q' to enable exploration and reset agent position.")

    print("Press 'e' to disable exploration and reset agent position.")

    print("Press 'w' to reset agent position.")

    print("Press 1-{0} to change maps.".format(len(world_cfg.world_maps)))



model = nengo.Network(seed=seed)

with model:

    # Create the environment

    env = GridNode(world_cfg.world, dt=0.005)



    # Handle the movement of the agent, and generate the movement

    # "goodness" grade

    

    def move(t, x, my_world=world_cfg):

        speed, rotation = x
        dt = 0.001
        max_speed = 10.0  # 10.0
        max_rotate = 10.0  # 10.0
        success = my_world.agent.in_destination(temp)
        if success:
            my_world.reset_pos()
        #my_world.agent.turn(rotation * dt * max_rotate)
        can_go_forward = my_world.agent.go_forward(speed * dt * max_speed)
        if not can_go_forward:
            my_world.agent.color = "red"
            #my_world.reset_pos()
            return 0
        else:
            my_world.agent.color = "blue"
            #my_world.reset_pos()
            go_toawrd = my_world.agent.go_towards(my_world.world.get_cell(temp[0], temp[1]))
            if go_toawrd:
                my_world.agent.turn(rotation * dt * max_rotate)
                my_world.agent.go_forward(speed * dt * max_speed)
            return turn_bias + speed
        
        
        
        
        """#attempting to add movement functions to make agen move forward and backward
        #my_world.agent.turn(rotation * dt * max_rotate)
        #my_world.agent.go_forward(speed * dt * max_speed)
        go_toawrd = my_world.agent.go_towards(my_world.world.get_cell(destination[0], destination[1]))
        print(go_toawrd)
        if go_toawrd:
            my_world.agent.turn(rotation * dt * max_rotate)
            my_world.agent.go_forward(speed * dt * max_speed)
        #my_world.agent.go_in_direction(my_world.agent.dir)
        #print(destination)
        can_go_forward = my_world.agent.go_forward(speed * dt * max_speed)
        #success = my_world.agent.go_towards("14", "1")
        
        #success = my_world.agent.go_towards(str(random.randint(1, len(my_world.get_map().splitlines()[1]))), str(random.randint(1, len(my_world.get_map().splitlines())-1)))
        success = my_world.agent.in_destination(destination)

        if success:
            my_world.reset_pos()
        #success = my_world.agent.turn_left()
        #if speed >= 0:
            #success = my_world.agent.go_backward(speed * dt * max_speed)
        #else:
            #my_world.agent.turn(rotation * dt * max_rotate)
            #my_world.agent.turn_around()
            #my_world.agent.turn_right()
            #my_world.agent.go_forward(speed * dt * max_speed) 
            #success = my_world.agent.go_forward(speed * dt * max_speed)  
            # Use -speed for backward
        if not can_go_forward:
            my_world.agent.color = "pink"
            #my_world.reset_pos()
            return 0
        else:
            my_world.agent.color = "blue"
            #my_world.reset_pos()
            return turn_bias + speed"""



    movement = nengo.Ensemble(n_neurons=100, dimensions=2, radius=1.4)

    movement_node = nengo.Node(move, size_in=2, label="reward")

    nengo.Connection(movement, movement_node)



    # Gener

    # ate the context (radar distance to walls front, left, right)

    def detect(t):

        angles = (np.linspace(-0.5, 0.5, radar_dim) + agent.dir) % world.directions

        return [agent.detect(d, max_distance=4)[0] for d in angles]



    stim_radar = nengo.Node(detect)



    # Create the action selection networks

    bg = nengo.networks.actionselection.BasalGanglia(3)

    thal = nengo.networks.actionselection.Thalamus(3)

    nengo.Connection(bg.output, thal.input)

    #print("Ball output: ", bg.output[0])

    #print("Thal input: ", thal.input[0])



    # Convert the selection actions to movement transforms

    nengo.Connection(thal.output[0], movement, transform=[[1], [0]])

    nengo.Connection(thal.output[1], movement, transform=[[0], [1]])

    nengo.Connection(thal.output[2], movement, transform=[[0], [-1]])

    #print("Thal output: ", thal.output[0])



    # Generate the training (error) signal

    def error_func(t, x):

        #print(x)

        actions = np.array(x[:3])

        utils = np.array(x[3:6])

        r = x[6]

        activate = x[7]



        max_action = max(actions)

        actions[actions < action_threshold] = 0

        actions[actions != max_action] = 0

        actions[actions == max_action] = 1



        z =  activate * (

            np.multiply(actions, (utils - r) * (1 - r) ** 5)

            + np.multiply((1 - actions), (utils - 1) * (1 - r) ** 5)

        )

        #print(z)

        return z



    errors = nengo.Node(error_func, size_in=8, size_out=3)

    #print(errors[:3])

    nengo.Connection(thal.output, errors[:3])

    nengo.Connection(bg.input, errors[3:6])

    nengo.Connection(movement_node, errors[6])



    # the learning is done on the board

    adapt_ens = FpgaPesEnsembleNetwork(

        board,

        n_neurons=100 * radar_dim,

        dimensions=radar_dim,

        learning_rate=learn_rate,

        function=lambda x: init_transform,

        seed=1524081122,

        label="pes ensemble",

    )

    adapt_ens.ensemble.radius = 4



    nengo.Connection(stim_radar, adapt_ens.input, synapse=learn_synapse)

    nengo.Connection(errors, adapt_ens.error)

    nengo.Connection(adapt_ens.output, bg.input)



    class LearnActive:

        """Class to store persistent learning state"""



        def __init__(self, my_world, page_data=None):

            self.my_world = my_world

            self.page = page_data



            self._is_learning = 2

            # _is_learning values:

            # < 0: no learning

            # 1: learning, will stop at learn_timeout

            # 2: continuous learning



        def __call__(self, t):

            if self.page is not None:

                init_agent_pos = False

                # Create a dictionary instead of if/else

                # "<key press>": (<learning>, <init_agent_pos>)

                keyboard_dict = {

                    "q": (2, True),

                    "e": (-1, True),

                    "w": (self._is_learning, True),

                }



                for k in self.page.keys_pressed:

                    if k.isdigit():

                        new_map_ind = int(k) - 1

                        if new_map_ind != self.my_world.curr_ind:

                            self.my_world.set_ind(new_map_ind)

                            init_agent_pos = True

                    elif k in list(keyboard_dict.keys()):

                        self._is_learning, init_agent_pos = keyboard_dict[k]



                learning = (

                    (t <= learn_timeout) or (self._is_learning == 2)

                ) and self._is_learning > 0



                self._nengo_html_ = """

                <svg width="100%" height="100%" viewbox="0 0 200 75">

                    <text x="50%" y="50%" fill="{0}" text-anchor="middle"

                     alignment-baseline="middle" font-size="50">{1}</text>

                </svg>

                """.format(

                    "red" if learning else "grey",

                    "Explore: ON" if learning else "Explore: Off",

                )



                if not learning and self._is_learning == 1:

                    init_agent_pos = True

                    self._is_learning = -1



                if init_agent_pos:

                    self.my_world.reset_pos()



                return int(learning)

            else:

                # Keyboard state branch not detected. Default to continuous learning

                self._nengo_html_ = """

                <svg width="100%" height="100%" viewbox="0 0 200 75">

                    <text x="50%" y="50%" fill="red" text-anchor="middle"

                     alignment-baseline="middle" font-size="50">

                     Explore: ON</text>

                </svg>

                """



                # Return 1, to turn learning on permanently

                return 1



    # Need to pass in keyboard handler since it's not local to the class

    learn_on = nengo.Node(LearnActive(world_cfg, locals().get("__page__")))

    nengo.Connection(learn_on, errors[7])