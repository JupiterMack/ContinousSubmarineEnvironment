import os, sys
import gym
import gym.spaces as gspc
import numpy as np
sys.path.append("ContinuousSubmarine_agents")
sys.path.append("ContinuousSubmarineEnvironment")
from ple.ple import PLE
from envs.random_trajectories import Nonstationarity

PIPE_GAP = 90

# configure the temporal pattern of CFB-hNS, i.e. the wash-out ratio and the maximum duration.
WASH_OUT_RATIO = 3
MAX_WASHOUT = 300

# Number of state features in CFB
NUM_FEAT = 7

def process_state(state):
    # Ensure that the state is a dictionary and all values can be converted to a float
    if isinstance(state, dict):
        try:
            processed_state = np.array([float(value) for value in state.values()])
        except ValueError:
            raise ValueError("Non-numeric state values encountered in state dictionary.")
    else:
        raise TypeError("State is expected to be a dictionary.")

    return processed_state


class PLEEnv_state(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name='FlappyBird', display_screen=True, nonstationary=None, nrandfeatures=0,
                 noise_level=0, phase='train'):
        # set headless mode
        """

        Parameters
        ----------
        game_name : str {'FlappyBird', 'ContFlappyBird'}
            The baseline game
        display_screen : bool
            Whether or not to visualize the game
        nonstationary : str {"gfNS", "gsNS", "hNS"}
            Which non-stationarity to apply: gfNS - fast and discrete switching of the gravity parameter
                                             gsNS - slow and continuous switching of the gravity parameter
                                             hNS  - non-stationarity similar to wash-out effect
            Add further options for additional types of NS or for using several types of NS in a single environment.
        nrandfeatures : int
            Number of additional uninformative random features
        noise_level : float
            The standard deviation of the additive noise on the state representation is computed as
            `noise_level` * mean(state value interval)
        phase : str {"train", "test"}
            Whether it is a training or test environment. Has consequences on which pre-generated non-stationary
            trajectory is loaded
        """

        os.environ['SDL_VIDEODRIVER'] = 'dummy'

        # open up a game state to communicate with emulator
        import importlib
        game_module_name = ('ple.games.%s' % game_name).lower()
        game_module = importlib.import_module(game_module_name)
        self.game_name = game_name
        self.game = getattr(game_module, game_name)(width=286, pipe_gap=PIPE_GAP)
        self.game_state = PLE(self.game, fps=30, display_screen=display_screen, state_preprocessor=process_state)
        self.game_state.init()

        # get the action and state set of the game
        self._action_set = self.game_state.getActionSet()
        self.action_space = gspc.Discrete(3)  # 0 for descent, 1 for ascent
        self.screen_height, self.screen_width = self.game_state.getScreenDims()
        if game_name == 'FlappyBird':
            self.observation_space = gspc.Box(low=-np.inf, high=np.inf, shape=(8+nrandfeatures,), dtype=np.float32)
        else:
            self.observation_space = gspc.Box(low=-np.inf, high=np.inf, shape=(7+nrandfeatures,), dtype=np.float32)
        self.viewer = None

        # initialize parameters of non-stationarities
        # CFB-hNS
        self.hNS = False
        self.noflap_cnt = 0
        self.flap_cnt = 1
        self.decayed_thrust = 9  # initial thust parameter
        

        # which NS to apply
        self.nonstationary = nonstationary
        self.phase = phase

        # n rand features
        self.nrandfeat = nrandfeatures

        # Additive noise on state representation
        self.noise = noise_level
        self.noise_gen = self.generate_noise_sources(noise_level)  # initialize noise generator

    def step(self, action):
        """
        Perform a step with a continuous action.

        Parameters
        ----------
        action : float
            Continuous action value representing flap power.
        """
        # Pass continuous action directly to the game's act method
        reward = self.game_state.act(action)  # Pass the action to act method
        state = self.game_state.getGameState()
        terminal = self.game_state.game_over()

        return state, reward, terminal, {}


    def _get_image(self):
        """
        Get the generated game world returned by PLE

        Returns
        -------
        image_rotated
        """
        image_rotated = np.fliplr(
            np.rot90(self.game_state.getScreenRGB(), 3))  # Hack to fix the rotated image returned by ple
        return image_rotated

    @property
    def _n_actions(self):
        """
        Return the number of valid actions

        Returns
        -------
        int
        """
        return len(self._action_set)

    def reset(self):
        """
        Reset the game.

        Returns
        -------
        state : numpy array
        """
        if self.game_name == 'FlappyBird':
            self.observation_space = gspc.Box(low=-np.inf, high=np.inf, shape=(8+self.nrandfeat,), dtype=np.float32)
        else:
            self.observation_space = gspc.Box(low=-np.inf, high=np.inf, shape=(NUM_FEAT+self.nrandfeat,), dtype=np.float32)
        self.game_state.reset_game()
        state = self.get_state()
        return state

    def render(self, mode='human', close=False):
        """
        Visualize the current game state

        Parameters
        ----------
        mode : str {"rgb_array", "human"}
        close : bool

        Returns
        -------
        img: numpy array
            Contains RGB values of the image
        """
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def seed(self, seed=None):
        """
        Initialize random number generator and non-stationarities as well as uncertainties in the environment
        such as random features.

        Load the pre-generated temporal patterns of the gravity parameter and of the values of random features.
        Add new non-stationarities here.

        Parameters
        ----------
        seed : int
            Seed for the random number generator. `seed` also defines which pre-generated temporal pattern is loaded.
        """
        rng = np.random.RandomState(seed)
        self.game_state.rng = rng
        self.game_state.game.rng = self.game_state.rng
        self.game_state.init()

        # --- NON-STATIONARITIES ---
        # Set the parameter update function and load the pre-generated sequences of parameter values.
        # A new non-stationarity needs to define the update function of a parameter (`update_param`)
        # and specify the temporal pattern `param_traj`.

        # non-stationary effect of action
        if self.nonstationary == 'hNS':
            self.hNS = True
            self.nonstationary = None       # as no dynamics parameter is updated.

        # fast non-stationarity of internal state -> update gravity
        elif self.nonstationary == 'gfNS':
            self.update_param = self._update_gravity  # which parameter shall be updated?
            self.param_traj = Nonstationarity(self.nonstationary, seed%20, self.phase) # temporal pattern of the parameter

        # slow non-stationarity of internal state -> update gravity
        elif self.nonstationary == 'gsNS':
            self.update_param = self._update_gravity # which parameter shall be updated?
            self.param_traj = Nonstationarity(self.nonstationary, seed%20, self.phase) # temporal pattern of the parameter

        elif self.nonstationary == 'random_waveform':
            self.update_param = self._update_random_waveform
            self.param_traj = Nonstationarity(self.nonstationary, seed%20, self.phase)


        # --- RANDOM FEATURES ---
        if self.nrandfeat == 0:
            self.get_state = self.game_state.getGameState
        # random features with same initial statistics than y position of the bird
        # i.e. similar std, mean and step-wise change-rate.
        elif self.nrandfeat > 0:
            self.feat_traj = [Nonstationarity('rand_feat', (seed + i) % 20, self.phase) for i in range(self.nrandfeat)]
            self.get_state = self.get_extended_state

    def get_extended_state(self):
        """
        Get game state that includes random features.

        Random feature values are drawn from the pre-genenated sequence of random values: `feat_traj`.

        Returns
        -------
        state : numpy array
            The extended state, i.e. the original game state plus `self.nrandfeat` random features
        """
        state = self.game_state.getGameState()
        extended_vals = [traj.get_next_value()*200 for traj in self.feat_traj]
        state = np.concatenate([state, extended_vals])
        return state

    def generate_noise_sources(self, noise_level):
        """
        Initialize the noise generator for additive noise on the state representation.

        The noise is Gaussian distributed with zero-mean and standard deviation:
        std = `noise_level` * (half of each state feature value range)

        Parameters
        ----------
        noise_level : float
            Defines the standard deviation of the additive noise.

        Returns
        -------
        noise_gen : lambda function that returns an array with random values. One for each state feature.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from envs.nenvironment import PLEEnv_state
        >>> p = PLEEnv_state()
        >>> noise_generator = p.generate_noise_sources(0.5)
        >>> noise = []
        >>> for i in range(20):
        >>>     n = noise_generator()
        >>>     noise.append(n)
        >>> plt.figure()
        >>> plt.plot(noise)
        >>> plt.show()
        """
        feat_val_interval = [350, 17, 200, 200, 50, 200, 200]  # difference between the minimum and maximum value
                                                               # of each feature.
        feat_val_interval.extend([140] * self.nrandfeat)
        noise_std = [1/2 * noise_level * f_int for f_int in feat_val_interval]
        noise_gen = lambda: [np.random.normal(0, noise_std[i]) for i in range(len(feat_val_interval))]
        return noise_gen

    def add_noise(self, state, noise_level):
        """
        Add noise to state representation if noise_level is not 0.

        Parameters
        ----------
        state : numpy array
            Current state signal
        noise_level : float
            The number by which the half of a state feature's value range is scaled.

        Returns
        -------
        numpy array
            Updated state signal
        """
        if noise_level == 0:
            return state
        else:
            noise = self.noise_gen()
            state = [s+n for s,n in zip(state, noise)]
            return np.asarray(state)

    def decay_thrust(self, action):
        """
        Compute decayed flap power after a sequence of N "flap"-actions.

        The decayed flap power depends on the duration of the previous sequence of "flap"-actions (`self.flap_cnt`) and
        the constants `WASH_OUT_RATIO` and `MAX_WASHOUT` which define the decay rate of the flap power.

        Parameters
        ----------
        action : int {0,1}
            The type of action which shall be executed: 0 -> "flap"; 1 -> "no flap"

        Returns
        -------
        action = 0
            Return a "flap"-action (action  = 0) in any case.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> from envs.nenvironment import PLEEnv_state
        >>> p = PLEEnv_state()
        >>> thrust_history = []
        >>> for act in [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,1,0,1,0,1,0,0,0,0,0,1,1,0,1,0,1,1,1,1,1,1,0,0,0]:
        >>>     _ = p.decay_thrust(act)
        >>>     thrust_history.append(p.game.player.FLAP_POWER)
        >>> plt.figure()
        >>> plt.plot(thrust_history)
        >>> plt.show()
        """
        if action == 1:  # NO FLAP
            self.noflap_cnt += 1
            self.decayed_thrust -= 9. * WASH_OUT_RATIO/min(self.flap_cnt, MAX_WASHOUT)
        elif action == 0:  # FLAP
            self.decayed_thrust = 9
            # if it's the first "flap"-action after a sequence of "no flap"-actions
            if not self.noflap_cnt == 0:
                self.noflap_cnt = 0
                self.flap_cnt = 0
            self.flap_cnt += 1

        # update flap power
        self.game.player.FLAP_POWER = max(0, min(10, self.decayed_thrust))
        return 0

    def _update_gravity(self, gravity):  # Either this or flap power
        """
        Set gravity constant of the baseline game to a new value.

        Parameters
        ----------
        gravity : float
            New value of the gravity constant.
        """
        self.game.set_gravity(gravity)

    def _update_background_speed(self, speed):
        """
        Set speed of pipes in the baseline game to a new value.

        Parameters
        ----------
        speed : float
            New value of the speed constant.
        """
        self.game.set_speed(speed)
        
    def _update_random_waveform(self, value):
        """
        Set the oscillation pattern of the pipes' position in the baseline game.

        This method adjusts the waveform pattern that determines the vertical position 
        of the pipes. By updating this waveform based on the provided value, the game 
        can introduce variations in the pipe positions, adding an element of unpredictability 
        and challenge to the gameplay.

        Parameters
        ----------
        value : float
            New value to adjust the waveform pattern for pipe positions.
        """
        self.game.set_pipe_position(value)




