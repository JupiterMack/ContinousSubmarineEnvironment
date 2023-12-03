import os
import sys
import numpy as np

sys.path.append("FlappyBird_agents_upgraded")
sys.path.append("FlappyBird_environment_upgraded")
import pygame
from pygame.constants import K_w
from .. import base
from ..base.pygamewrapper import PyGameWrapper


class BirdPlayer(pygame.sprite.Sprite):
    """
    The player of the game
    """

    def __init__(self,
                 SCREEN_WIDTH, SCREEN_HEIGHT, init_pos,
                 image_assets, rng, color="red", scale=1.0):

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

        # all in terms of y
        self.vel = 0
        self.FLAP_POWER = 9 * self.scale
        self.MAX_DROP_SPEED = 10.0
        self.GRAVITY = 1.0 * self.scale

        self.rng = rng

        self._oscillateStartPos()  # makes the direction and position random
        self.rect.center = (self.pos_x, self.pos_y)  # could be done better

        self.pos_y_limits = [0, self.SCREEN_HEIGHT * 0.79 - self.height]

    def init(self, init_pos, color):
        """
        Set up the surface we draw the bird too

        Parameters
        ----------
        init_pos : [int, int]
        color : str
        """
        self.flapped = True  # start off w/ a flap
        self.current_image = 0
        self.color = color
        self.image = self.image_assets[self.color][self.current_image]
        self.rect = self.image.get_rect()
        self.thrust_time = 0.0
        self.game_tick = 0
        self.pos_x = init_pos[0]
        self.pos_y = init_pos[1]

    def _oscillateStartPos(self):
        """
        Randomly generate the initial y position
        """
        offset = 8 * np.sin(self.rng.rand() * np.pi)
        self.pos_y += offset

    def flap(self):
        """
        Set parameters, such that bird flaps in next update() call
        """

        if self.pos_y > -2.0 * self.image.get_height():
            self.vel = 0.0
            self.flapped = True

    def update(self, dt):  # dt = 1/fps
        """
        Update the bird's position and speed

        Parameters
        ----------
        dt : float
            Time fraction
        """
        self.game_tick += 1

        # image cycle of the flapping bird
        if (self.game_tick + 1) % 15 == 0:
            self.current_image += 1

            if self.current_image >= 3:
                self.current_image = 0

            # set the image to draw with.
            self.image = self.image_assets[self.color][self.current_image]
            self.rect = self.image.get_rect()

        # update vertical speed of the bird:
        if self.vel < self.MAX_DROP_SPEED and self.thrust_time == 0.0:
            self.vel += self.GRAVITY  # dv/dt = a = self.GRAVITY

        # the whole point is to spread this out over the same time it takes in 30fps.
        if self.thrust_time + dt <= (1.0 / 30.0) and self.flapped:
            self.thrust_time += dt
            self.vel += -1.0 * self.FLAP_POWER + self.GRAVITY # dv/dt = a = F / m = - FLAP_POWER / 1
        else:
            self.thrust_time = 0.0
            self.flapped = False

        # bird cannot move out of the visible area
        self.pos_y += self.vel  # s = s_old + ds/dt * dt = s_old + v*dt
        if self.pos_y > self.pos_y_limits[1]:
            self.pos_y = self.pos_y_limits[1]
        elif self.pos_y < self.pos_y_limits[0]:
            self.pos_y = self.pos_y_limits[0]
        self.rect.center = (self.pos_x, self.pos_y)

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
                 offset=0, color="green"):

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

        self.width = self.image_assets["green"]["lower"].get_width()  # 52
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

        self.lower_pipe = self.image_assets[color]["lower"]
        self.upper_pipe = self.image_assets[color]["upper"]

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
                 image_background, image_base, scale, speed):
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

        screen.blit(self.base_image, (self.x, self.SCREEN_HEIGHT * 0.79))

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

    def __init__(self, width=288, height=512, pipe_gap=100):

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
            int(self.height / 2)
        )

        # Set limits of the pipe gap.
        self.pipe_min = int(self.pipe_gap / 4)
        self.pipe_max = int(self.height * 0.79 * 0.6 - self.pipe_gap / 2)

        self.backdrop = None
        self.player = None
        self.pipe_group = None

        # New attributes for controlling pipe alignment
        self.num_pipes_same_gap = 5  # Number of pipes to maintain the same gap
        self.current_pipe_gap = self.pipe_min  # Initial pipe gap
        self.pipes_generated = 0  # Counter for the number of pipes generated

        self.rewards = {
            "positive": 0.1,
            "negative": -0.4,
            "tick": 0,
            "loss": -0.5,
            "win": 5.0
        }

    def _load_images(self):
        """
        preload and convert all the images so its faster when we reset
        """

        self.images["player"] = {}
        for c in ["red", "blue", "yellow"]:
            image_assets = [
                os.path.join(self._asset_dir, "%sbird-upflap.png" % c),
                os.path.join(self._asset_dir, "%sbird-midflap.png" % c),
                os.path.join(self._asset_dir, "%sbird-downflap.png" % c),
            ]

            self.images["player"][c] = [pygame.image.load(
                im).convert_alpha() for im in image_assets]

        self.images["background"] = {}
        for b in ["day", "night"]:
            path = os.path.join(self._asset_dir, "background-%s.png" % b)

            self.images["background"][b] = pygame.image.load(path).convert()

        self.images["pipes"] = {}
        for c in ["red", "green"]:
            path = os.path.join(self._asset_dir, "pipe-%s.png" % c)

            self.images["pipes"][c] = {}
            self.images["pipes"][c]["lower"] = pygame.image.load(
                path).convert_alpha()
            self.images["pipes"][c]["upper"] = pygame.transform.rotate(
                self.images["pipes"][c]["lower"], 180)

        path = os.path.join(self._asset_dir, "base.png")
        self.images["base"] = pygame.image.load(path).convert()

    def init(self):
        """
        Initialize background, player and pipes.
        """
        if self.backdrop is None:
            self.backdrop = Backdrop(
                self.width,
                self.height,
                self.images["background"]["day"],
                self.images["base"],
                self.scale,
                speed = self.speed #!
            )

        if self.player is None:
            self.player = BirdPlayer(
                self.width,
                self.height,
                self.init_pos,
                self.images["player"],
                self.rng,
                color="red",
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
        color = self.rng.choice(["day", "night"])
        self.backdrop.background_image = self.images["background"][color]

        # instead of recreating
        color = self.rng.choice(["red", "blue", "yellow"])
        self.player.init(self.init_pos, color)

        self.pipe_color = self.rng.choice(["red", "green"])
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

    def _handle_player_events(self):
        """
        Process keyboard events
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == self.actions['up']:
                    self.player.flap()

    def game_over(self):
        """
        Return the game state - game over?

        Returns
        -------
        bool
            True if self.lives <= 0, else False

        """
        return self.lives <= 0

    def step(self, dt):  # dt comes in as ms (1000/ fps)
        """
        Move the game objects.

        Parameters
        ----------
        dt : float
            Time fraction
        """
        self.game_tick += 1
        dt = dt / 1000.0   # so dt = 1/fps

        # handle player movement
        self._handle_player_events()

        # check whether the bird hit a pipe
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

        # update the player, pipe and backgournd position
        self.player.update(dt)
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

    def set_gravity(self, gravity):
        """
        Set the gravity parameter to generate non-stationary dynamcis.

        Parameters
        ----------
        gravity : float
        """
        self.player.GRAVITY = gravity

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

