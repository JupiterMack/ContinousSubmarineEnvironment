import os
import sys
import numpy as np

import pygame
from pygame.constants import K_w
from ple.games import base
from ple.games.base.pygamewrapper import PyGameWrapper

GROUND_LEVEL = 1-.20 #change seafloor level
PIPE_GROUP = 5

class BirdPlayer(pygame.sprite.Sprite):
    """
    The player of the game
    """
    MAX_VELOCITY = 5  # Define a maximum velocity
    

    def __init__(self,
                 SCREEN_WIDTH, SCREEN_HEIGHT, init_pos,
                 image_assets, rng, color="red", scale=1.0, ground_level=GROUND_LEVEL):

        """
        Initialize the bird player.

        Parameters
        ----------
        SCREEN_WIDTH : int
        SCREEN_HEIGHT : int
        init_pos : [int, int]
        image_assets : pygame.image
        rng :
        color : str {"red", "blue", "yellow"}
        scale : float
        """
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        self.image_order = [0, 1, 2, 1]
        # done image stuff

        pygame.sprite.Sprite.__init__(self)

        self.image_assets = image_assets

        self.init(init_pos, color)

        self.height = self.image.get_height()  # 24
        self.width = self.image.get_width()    # 34
        self.scale = scale
        self.ground_level = ground_level
        self.vel = 0

        self.pos_y_limits = [0, self.SCREEN_HEIGHT * self.ground_level - self.height]
        self.trajectory_index = 0

        # New attributes for ascent and descent trajectories
        self.ascent_trajectory = self.calculate_ascent_trajectory(ground_level)
        self.descent_trajectory = self.calculate_descent_trajectory(ground_level)

        self.is_ascending = False  # Tracks whether the bird is ascending
        self.last_action = None  # This line is added

        

        self.timesteps_since_last_action_change = 0  # Initialize the counter
        self.action_change_interval = 30  # Set the minimum interval for action change

        self.flap_history = []
        self.flap_counter = 0

        self.rng = rng

        self._StartPos()  # makes the direction and position random
        self.constant_altitude = self.pos_y

    def init(self, init_pos, color):
        """
        Set up the surface we draw the bird too

        Parameters
        ----------
        init_pos : [int, int]
        color : str
        """
        #self.flapped = True  # start off w/ a flap
        self.current_image = 0
        self.color = color
        self.image = self.image_assets[self.color][self.current_image]
        self.rect = self.image.get_rect()
        self.thrust_time = 0.0
        self.game_tick = 0
        self.pos_x = init_pos[0]
        self.pos_y = init_pos[1]

    """ def decide_target_position(self, action):
        if action == 0:  # Ascend
            index = self.find_closest_index(self.pos_y, self.ascent_trajectory)
            if index is not None and index < len(self.ascent_trajectory):
                return self.ascent_trajectory[index]
        elif action == 1:  # Descend
            index = self.find_closest_index(self.pos_y, self.descent_trajectory)
            if index is not None and index < len(self.descent_trajectory):
                return self.descent_trajectory[index]
        return self.constant_altitude
    
    def find_closest_index(self, altitude, trajectory):
        if trajectory is not None and len(trajectory) > 0:
            return np.argmin(np.abs(trajectory - altitude))
        return None """
    
    def _StartPos(self):
        """
        Set the initial y position at ground level.
        """
        # Calculate the ground level position
        ground_position = self.SCREEN_HEIGHT * self.ground_level

        # Set the bird's initial position to just above the ground
        # Subtract the height of the bird to place it above the ground
        self.pos_y = ground_position - self.height

        self.rect.center = (self.pos_x, self.pos_y)
    
    def find_closest_index(self, altitude, trajectory):
        """
        Find the index of the point in the trajectory closest to the current altitude.

        Parameters:
        altitude: Current altitude of the bird.
        trajectory: The trajectory (ascent or descent) to follow.

        Returns:
        int: Index of the closest point on the trajectory.
        """
        return np.argmin(np.abs(trajectory - altitude))

    @staticmethod
    def calc_curve(a1: float, a2: float, w1: float, w2: float, junction_x: float, nsteps: int = 100) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the stitched sigmoid curve for the bird's trajectory.
        The function now finds a junction point where derivatives of both curves are equal.

        Parameters:
        a1, a2: Amplitudes of the left and right sigmoids.
        w1, w2: Widths of the left and right sigmoids.
        junction_x: Approximate junction point (x-value).
        nsteps: Number of steps for linspace.
        """
        # Adjusting the end points of np.linspace calls
        left_x = np.linspace(-10, junction_x, nsteps // 2)
        right_x = np.linspace(junction_x, 10, nsteps // 2)

        left = a1 / (1 + np.exp(-w1 * (left_x - junction_x)))
        right = a2 / (1 + np.exp(-w2 * (right_x - junction_x)))
        # Adjust for the height at the junction point
        hzero_l = left[-1]
        hzero_r = right[0]
        height_adjustment = hzero_l - hzero_r
        right += height_adjustment

        x_values = np.concatenate([left_x, right_x])
        y_values = right #np.concatenate([right, left])       #swap left and right for steep vs gradual ascent

        return x_values, y_values
    
    def calculate_ascent_trajectory(self, current_y):
        a1, a2, w1, w2 = self.get_ascent_parameters()
        _, y_values = self.calc_curve(a1, a2, w1, w2, 5, nsteps=100)
        # Map y_values to the range between the bird's current position and the ceiling
        y_min, y_max = y_values.min(), y_values.max()
        scaled_y_values = np.interp(y_values, (y_min, y_max), (current_y, self.pos_y_limits[0]))

        # Ensure that the trajectory is within the screen limits
        scaled_y_values = np.clip(scaled_y_values, self.pos_y_limits[0], self.pos_y_limits[1])

        return scaled_y_values
    
    def get_ascent_parameters(self):
        """
        Get the parameters for the ascent trajectory's stitched sigmoid function.

        Returns:
        tuple: (a1, a2, w1, w2)
        """
        a1 = 5  # Lower amplitude of the left sigmoid for a less steep ascent
        a2 = 60  # Lower amplitude of the right sigmoid
        w1 = .5  # Wider width for a smoother transition
        w2 =  5 # Wider width for the right sigmoid
        return a1, a2, w1, w2

    def calculate_descent_trajectory(self, current_y):
        a, w, y_min, y_max = self.get_descent_parameters()
        x_values = np.linspace(-10, 10, 100)
        y_values = self.descent_sigmoid(x_values, a, w, y_min, y_max)
        # Adjust the y-values to start from current_y and scale towards the ground
        scaled_y_values = np.interp(y_values, (y_values.min(), y_values.max()), (current_y, self.pos_y_limits[1]))

        # Ensure that the trajectory is within the screen limits
        scaled_y_values = np.clip(scaled_y_values, self.pos_y_limits[0], self.pos_y_limits[1])

        return scaled_y_values
    
    def get_descent_parameters(self):
        """
        Get the parameters for the descent trajectory's sigmoid function.

        Returns:
        tuple: (a, w, y_min, y_max)
        """
        a = 2  # Lower amplitude for a less steep descent
        w = .4  # Wider width for a smoother descent
        y_min = self.pos_y_limits[0]  # Minimum y-value (e.g., ground level)
        y_max = self.pos_y_limits[1]  # Maximum y-value (e.g., ceiling level)
        return a, w, y_min, y_max

    
    def descent_sigmoid(self, x, a, w, y_min, y_max):
        """
        Sigmoid function for descent trajectory.
        x: Current position or time
        a: Amplitude of the sigmoid
        w: Width of the sigmoid
        y_min: Minimum y-value (sea floor level)
        y_max: Maximum y-value (ceiling level)
        """
        return y_min + (y_max - y_min) / (1 + np.exp(-w * (x - a)))
    
    def scale_trajectory(self, y_values):
        # Scale the y-values to cover the entire vertical range
        y_range = self.pos_y_limits[1] - self.pos_y_limits[0]
        return self.pos_y_limits[0] + y_range * ((y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values)))
    
    def follow_trajectory(self, trajectory, action):
        # Use game_tick to cycle through the trajectory
        if action == 2:  # Maintain altitude
            # Logic for maintaining altitude
            pass
        else:
            # Smoothly follow the trajectory
            target_y = trajectory[self.trajectory_index]

            move_step = 5  # Smaller move_step for smoother movement
            self.pos_y = self.pos_y + move_step * np.sign(target_y - self.pos_y)

            # Clamp position within screen limits
            self.pos_y = max(min(self.pos_y, self.pos_y_limits[1]), self.pos_y_limits[0])
            self.rect.center = (self.pos_x, self.pos_y)

            # Increment trajectory index
            if self.game_tick % 5 == 0:
                self.trajectory_index = (self.trajectory_index + 1) % len(trajectory)

       
        return self.trajectory_index
    
    def update(self, action):  # dt = 1/fps
        """
        Update the bird's position and speed

        Parameters
        ----------
        dt : float
            Time fraction
        """
        self.game_tick += 1

        # Check if the action has changed
        if self.last_action != action:
            self.last_action = action
            self.trajectory_index = 0  # Reset the trajectory index on action change

            # Choose the trajectory based on the action
            if action == 0:  # Ascend
                self.current_trajectory = self.calculate_ascent_trajectory(self.pos_y)
            elif action == 1:  # Descend
                self.current_trajectory = self.calculate_descent_trajectory(self.pos_y)
            else:  # Maintain altitude
                self.current_trajectory = np.array([self.pos_y] * self.action_change_interval)

        # Update the bird's position based on the current trajectory
        if self.trajectory_index < len(self.current_trajectory):
            self.pos_y = self.current_trajectory[self.trajectory_index]
            self.trajectory_index += 1
        else:
            # If the end of the trajectory is reached, maintain the last position
            self.pos_y = self.current_trajectory[-1]

        
        # Ensure bird's position is within the screen limits
        self.pos_y = np.clip(self.pos_y, self.pos_y_limits[0], self.pos_y_limits[1])

        # Image cycle for the flapping bird
        if (self.game_tick + 1) % 15 == 0:
            self.current_image += 1
            if self.current_image >= 3:
                self.current_image = 0

            # Set the image to draw with.
            self.image = self.image_assets[self.color][self.current_image]
            self.rect = self.image.get_rect()

        self.rect.center = (self.pos_x, self.pos_y)
        self.timesteps_since_last_action_change += 1

    
    def draw(self, screen):
        """
        Draw the bird onto the game world

        Parameters
        ----------
        screen : pygame.display
        """
        screen.blit(self.image, self.rect.center)


class Pipe(pygame.sprite.Sprite):

    def __init__(self,
                 SCREEN_WIDTH, SCREEN_HEIGHT, gap_start, gap_size, image_assets, scale, speed,
                 offset=0, color="sub"):

        """
        Initialize single Pipe object

        Parameters
        ----------
        SCREEN_WIDTH : int
        SCREEN_HEIGHT : int
        gap_start : int
        gap_size : int
        image_assets : pygame.image
        scale : float
        speed : float
        offset : int (default 0)
        color : str {"green", "red"}
        """
        self.speed = speed * scale  # x-distance per dt
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        self.image_assets = image_assets
        # done image stuff

        self.width = self.image_assets["sub"]["lower"].get_width()  # 52
        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.Surface((self.width, self.SCREEN_HEIGHT))
        self.image.set_colorkey((0, 0, 0))

        self.init(gap_start, gap_size, offset, color)

    def init(self, gap_start, gap_size, offset, color):
        """
        Set up the surface we draw the upper and lower pipe too

        Parameters
        ----------
        gap_start
        gap_size
        offset
        color
        """
        self.image.fill((0, 0, 0))
        self.gap_start = gap_start
        self.x = self.SCREEN_WIDTH + offset  # + self.width

        self.lower_pipe = self.image_assets["sub"]["lower"]
        self.upper_pipe = self.image_assets["sub"]["upper"]

        top_bottom = gap_start - self.upper_pipe.get_height()
        bottom_top = gap_start + gap_size

        self.image.blit(self.upper_pipe, (0, top_bottom))
        self.image.blit(self.lower_pipe, (0, bottom_top))

        self.rect = self.image.get_rect()
        self.rect.center = (self.x, self.SCREEN_HEIGHT / 2)

    def update(self, dt):
        """
        Update the pipe position

        Parameters
        ----------
        dt : float
            Time fraction
        """
        self.x -= self.speed
        self.rect.center = (self.x, self.SCREEN_HEIGHT / 2)


class Backdrop():

    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT,
                 image_background, image_base, scale, speed, ground_level=GROUND_LEVEL):
        """
        Initialize the game world background

        Parameters
        ----------
        SCREEN_WIDTH : int
            Screen width in pixels.
        SCREEN_HEIGHT : int
            Screen height in pixels.
        image_background : pygame.image
        image_base : pygame.image
        scale : float
        speed : float
        """
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        self.background_image = image_background
        self.base_image = image_base

        self.x = 0
        self.speed = speed * scale
        self.ground_level=ground_level
        self.max_move = self.base_image.get_width() - self.background_image.get_width()

    def update_draw_base(self, screen, dt):
        # the extra is on the right
        """
        Update the backgrounds position

        Parameters
        ----------
        screen : pygame.display
        dt : float
            Time fraction
        """
        if self.x > -1 * self.max_move:
            self.x -= self.speed
        else:
            self.x = 0

        screen.blit(self.base_image, (self.x, self.SCREEN_HEIGHT * self.ground_level))

    def draw_background(self, screen):
        """
        Draw the background

        Parameters
        ----------
        screen : pygame.display
        """
        screen.blit(self.background_image, (0, 0))

class ContFlappyBird(PyGameWrapper):
    """
    Used physics values from sourabhv's `clone`_.

    .. _clone: https://github.com/sourabhv/FlapPyBird


    Parameters
    ----------
    width : int (default: 288)
        Screen width. Consistent gameplay is not promised for different widths or heights, therefore the width and
        height should not be altered.

    height : inti (default: 512)
        Screen height.

    pipe_gap : int (default: 100)
        The gap in pixels left between the top and bottom pipes.

    """

    def __init__(self, width=288, height=512, pipe_gap=100, ground_level=GROUND_LEVEL):

        """
        Initialize ContinousFlappyBird

        Set up all parameters, the images and the screen. This also defines the reward structure. Thus, if you want to
        add further levels of reward, add them here.

        Parameters
        ----------
        width
        height
        pipe_gap
        """

        actions = {
            "up": K_w
        }

        fps = 30
        self.speed = 4.0

        base.PyGameWrapper.__init__(self, width, height, actions=actions)  #self.width = 286

        self.scale = 30.0 / fps
        self.allowed_fps = 30  # restrict the fps
        self.pipe_gap = pipe_gap
        self.ground_level = ground_level
        self.pipe_color = "red"
        self.images = {}

        # so we can preload images
        pygame.display.set_mode((1, 1), pygame.NOFRAME)

        # setup image paths
        self._dir_ = os.path.dirname(os.path.abspath(__file__))
        self._asset_dir = os.path.join(self._dir_, "assets/")
        self._load_images()

        # Set up the postion of pipes and the bird
        self.pipe_width = 52
        self.pipe_offsets = [int(-0.60*self.width + i*self.pipe_width) for i in range(7)]
        self.init_pos = (
            int(self.width * 0.2),
            int(height * ground_level-20)               #set to height*ground_level-20 for bottom, set to 5 for top.
        )

        # Set limits of the pipe gap.
        self.pipe_min = int(self.pipe_gap / 4)
        self.pipe_max = int(self.height * 0.79 * 0.6 - self.pipe_gap / 2)

        self.backdrop = None
        self.player = None
        self.pipe_group = None

        # New attributes for controlling pipe alignment
        self.num_pipes_same_gap = PIPE_GROUP  # Number of pipes to maintain the same gap raise to make more stable, lower to reduce stability
        self.current_pipe_gap = self.pipe_min  # Initial pipe gap
        self.pipes_generated = 0  # Counter for the number of pipes generated
        self.current_action = 0

        self.rewards = {
            "positive": 0.1,
            "negative": -0.4,
            "tick": 0,
            "loss": -0.5,
            "win": 5.0
        }

    def set_flap_power(self, power):
        """
        Set the flap power of the bird.

        Parameters
        ----------
        power : float
            The power of the flap, typically a continuous value.
        """
        if self.player:  # Ensure the player object exists
            self.player.set_flap_power(power)

    def _load_images(self):
        """
        preload and convert all the images so its faster when we reset
        """

        self.images["player"] = {}
        for c in ["sub"]:
            image_assets = [
                os.path.join(self._asset_dir, "sub-upflap.png"),
                os.path.join(self._asset_dir, "sub-midflap.png"),
                os.path.join(self._asset_dir, "sub-downflap.png"),
            ]

            self.images["player"][c] = [pygame.image.load(im).convert_alpha() for im in image_assets]

        self.images["background"] = {}
        for b in ["sub"]:
            path = os.path.join(self._asset_dir, "background-sub.png")
            self.images["background"][b] = pygame.image.load(path).convert()

        self.images["pipes"] = {}
        for c in ["sub"]:
            path = os.path.join(self._asset_dir, "pipe-sub.png")

            self.images["pipes"][c] = {}
            self.images["pipes"][c]["lower"] = pygame.image.load(path).convert_alpha()
            self.images["pipes"][c]["upper"] = pygame.transform.rotate(self.images["pipes"][c]["lower"], 180)

        path = os.path.join(self._asset_dir, "base-sub.png")
        self.images["base"] = pygame.image.load(path).convert()


    def init(self):
        """
        Initialize background, player and pipes.
        """
        if self.backdrop is None:
            self.backdrop = Backdrop(
                self.width,
                self.height,
                self.images["background"]["sub"],
                self.images["base"],
                self.scale,
                speed = self.speed, #!
                ground_level = self.ground_level
            )

        if self.player is None:
            self.player = BirdPlayer(
                self.width,
                self.height,
                self.init_pos,
                self.images["player"],
                self.rng,
                color="sub",
                scale=self.scale
            )

        if self.pipe_group is None:
            self.pipe_group = pygame.sprite.Group([
                self._generatePipes(offset=-75),
                self._generatePipes(offset=-75 + self.width / 2),
                self._generatePipes(offset=-75 + self.width * 1.5),
                self._generatePipes(offset=-75 + self.width * 2),
                self._generatePipes(offset=-75 + self.width * 2.5),
                self._generatePipes(offset=-75 + self.width * 3),
                self._generatePipes(offset=-75 + self.width * 4)
            ])

        # Set initial background type
    
        self.backdrop.background_image = self.images["background"]["sub"]

        # instead of recreating
        color = "sub"
        self.player.init(self.init_pos, color)

        self.pipe_color = "sub"
        for i, p in enumerate(self.pipe_group):
            self._generatePipes(offset=self.pipe_offsets[i], pipe=p)

        self.score = 0.0
        self.lives = 1
        self.game_tick = 0

    def getGameState(self):
        """
        Gets a non-visual state representation of the game.

        Returns
        -------

        dict
            * player y position.
            * players velocity.
            * current pipe top y position
            * current pipe bottom y position
            * next pipe distance to player
            * next pipe top y position
            * next pipe bottom y position

            See code for structure.

        """
        pipes = []
        for p in self.pipe_group:
            # If end of pipe is not yet passed by bird center
            if p.x + 10 >= self.player.pos_x:
                pipes.append((p, p.x - self.player.pos_x))

        # Sort pipes according to distance of end of pipe to the player
        pipes.sort(key=lambda p: p[1])

        current_pipe = pipes[1][0]
        next_pipe = pipes[0][0]

        if next_pipe.x < current_pipe.x:
            current_pipe, next_pipe = next_pipe, current_pipe

        state = {
            "player_y": self.player.pos_y,
            "player_vel": self.player.vel,

            "curr_pipe_top_y": current_pipe.gap_start,
            "curr_pipe_bottom_y": current_pipe.gap_start + self.pipe_gap,

            "next_pipe_dist_to_player": next_pipe.x - next_pipe.width/2 - self.player.pos_x,
            "next_pipe_top_y": next_pipe.gap_start,
            "next_pipe_bottom_y": next_pipe.gap_start + self.pipe_gap
        }

        return state

    def getScore(self):
        """
        Return the game score.

        Returns
        -------
        self.score : int

        """
        return self.score

    def _generatePipes(self, offset=0, pipe=None):
        """

        Parameters
        ----------
        offset : int
            Offset of pipe in x-direction in pixels
        pipe : Pipe instance

        Returns
        -------
        pipe : Pipe instance
        """

        # Change the gap only after the specified number of pipes
        if self.pipes_generated % self.num_pipes_same_gap == 0:
            # Randomly select a new gap start
            self.current_pipe_gap = self.rng.randint(self.pipe_min, self.pipe_max)

        if pipe is None:
            pipe = Pipe(
                self.width,
                self.height,
                self.current_pipe_gap,
                self.pipe_gap,
                self.images["pipes"],
                self.scale,
                color=self.pipe_color,
                offset=offset,
                speed=self.speed
            )

            return pipe
        else:
            pipe.init(self.current_pipe_gap, self.pipe_gap, offset, self.pipe_color)
        # Increment the pipe generation counter
        self.pipes_generated += 1

    def game_over(self):
        """
        Return the game state - game over?

        Returns
        -------
        bool
            True if self.lives <= 0, else False

        """
        return self.lives <= 0
    
    """ def handleAction(self, action):
        print(f"Received action: {action}")
        self.current_action = action
        self._handle_player_events(self.current_action)  # Handle the action immediately """

    def _handle_player_events(self):
        """
        Process keyboard events
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    def step(self, dt, action):  # dt comes in as ms (1000/ fps)
        """
        Move the game objects.

        Parameters
        ----------
        dt : float
            Time fraction
        action: int
            Action decided by agent (0 for descent, 1 for ascent)
        """
        
        self.game_tick += 1
        dt = dt / 1000.0   # so dt = 1/fps
        self._handle_player_events()
        state = self.getGameState()
        #print(f"Game Tick: {self.game_tick}, Action: {action} , Pos_y: {state}")
        hit_pipe = False
        for p in self.pipe_group:
            if self.player.rect.colliderect(p.rect):
                if self.player.pos_x < (p.x + 10):
                    top_pipe_check = ((self.player.pos_y - self.player.height/2) <= p.gap_start)
                    bot_pipe_check = ((self.player.pos_y + self.player.height/2) >= (p.gap_start + self.pipe_gap))
                    if top_pipe_check or bot_pipe_check:
                        hit_pipe = True
        if hit_pipe:
            self.score += self.rewards["negative"]
        else:
            self.score += self.rewards["positive"]

        # move the pipes
        for p in self.pipe_group:
            # is fully out of the screen within the next action?
            if p.x < -p.width + p.speed:
                self._generatePipes(offset=int(0.5*self.pipe_width), pipe=p)

        # check whether the bird fell on the ground
        if self.player.pos_y >= self.player.pos_y_limits[1]:
            self.score += self.rewards["loss"]

        # check whether the bird went above the screen
        if self.player.pos_y <= self.player.pos_y_limits[0]:
            self.score += self.rewards["loss"]

        # Use the provided action to update the player's state
        self.player.update(action)
        self.pipe_group.update(dt)

        # draw background, pipes and player
        self.backdrop.draw_background(self.screen)
        self.pipe_group.draw(self.screen)
        self.backdrop.update_draw_base(self.screen, dt)
        self.player.draw(self.screen)


    def set_speed(self, speed):
        """
        Set background speed parameter, i.e. how fast the bird flies through the game world. Consistent gameplay is not
        promised for different settings.

        Parameters
        ----------
        speed : float
            The speed with which the world (the pipes) moves relative to the bird player
        """

        for p in self.pipe_group:
            p.speed = speed
        self.backdrop.speed = speed

    def set_pipe_position(self, value):
        """
        Adjust the position of the pipes based on the provided value.

        Parameters
        ----------
        value : float
            The value by which to adjust the position of the pipes.
        """
        for pipe in self.pipe_group:
            # Adjust the gap_start position of each pipe
            pipe.gap_start += value
            # Ensure the pipe's position remains within the allowed limits
            pipe.gap_start = max(self.pipe_min, min(self.pipe_max, pipe.gap_start))
            # Reinitialize the pipe with the new gap_start position
            pipe.init(pipe.gap_start, self.pipe_gap, pipe.x, self.pipe_color)

