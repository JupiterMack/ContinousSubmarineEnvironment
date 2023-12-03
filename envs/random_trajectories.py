import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import simplejson
sys.path.append("FlappyBird_agents_upgraded")
sys.path.append("FlappyBird_environment_upgraded")
PLOTTING = True

class Nonstationarity():
    """
    Class to load and return pre-generated sequences of values for implemented non-stationarities and RF in CFB

    Attributes
    ----------
    points_list : list of floats
        List that contains the sequence of paramter values
    ns_type : str
        String that determines the type of pre-generated no-stationarity that shall be loaded.
    phase : str
        String that determines whether the loaded pre-generated no-stationarity is a training or test version. The
        version differ in the random seed that are used during generation and in the length of the sequence.
    seed : int
        ID of the loaded trajectory - represents the seed, that has been used during generation of a sequence of values.

    Methods
    -------
    load_trajectory()
        Load a pre-generated sequence of values from a .txt file
    get_next_value()
        Return the next value from the sequence of values `points_list`

    Examples
    --------
    >>> ns = Nonstationarity(ns_type='gfNS', seed=1, phase='train')
    >>> ns.load_trajectory()
    >>> val = ns.get_next_value()
    >>> print(val)
    """
    def __init__(self, ns_type, seed, phase):
        """
        Initialize sequence of parameter values that are used to specify non-stationary dynamics in CFB.

        Parameters
        ----------
        ns_type : str {"gfNS", "gsNS", "hNS"}
            All parameters define the path to the pre-generated sequence of values
        seed : int
        phase : str {"train", "test"}
        """
        self.points_list = []
        self.ns_type = ns_type
        self.phase = phase  # train or test
        if phase == 'test':
            self.seed = 100+seed
        else:
            self.seed = seed
        self.load_trajectory()

    def load_trajectory(self):
        """
        Load a pre-generated sequence of random values.
        """

        #path = os.getcwd()
        path = 'C:/Users/mackj/OneDrive/Desktop/ThesisFB/FlappyBird_environment_upgraded'
        print('Load {}/envs/{}_envs/{}_{}.txt'.format(path, self.phase, self.ns_type, self.seed))
        with open(path + '/envs/{}_envs/{}_{}.txt'.format(self.phase, self.ns_type, self.seed), 'r') as f:
            self.points_list = simplejson.load(f)

    def get_next_value(self):
        """
        Get next value from the pre-generated list of parameter values.

        Returns
        -------
        float
        """
        if len(self.points_list) <= 1:
            self.load_trajectory()
        return self.points_list.pop()


class Overlayed_RandomSines():
    """
    Generate continuous sequence of 'random' values.

    The sequence is a superposition of random sines

    Attributes
    ----------
    nsamples : int
        Number of samples that shall be generated in a sequence
    nsines : int
        Number of sine waves that are superpositioned
    points_list : list of floats
        Sequence of random values that represent a smooth random line.
    fband : (2,) numpy array of floats
        Array with minimum and maximum allowed frequency of a sine wave.
    amplitude : float
        Maximum amplitude of the smooth random line. Each single sine waves gets an amplitude of `amplitude`/`nsines`.
    offset : float
        Offset of the sequence of values from 0.

    Methods
    -------
    add_values()
        Generate the sequence of values.
    get_next_value()
        Return the next value of `points_list`. Is only used for testing puposes, but not during the game play.
    get_len()
        Return the length of `points_list`.

    Examples
    --------
    >>> RS = Overlayed_RandomSines(nsamples=300, offset=4., amplitude=2., fband=[0.0002, 0.0004])
    >>> sequence_length = RS.get_len()
    >>> print('Length of the pre-generated sequence after initialization: {}'.format(sequence_length))
    >>> RS.add_values()
    >>> sequence_length = RS.get_len()
    >>> print('Length of the pre-generated sequence after adding 300 values: {}'.format(sequence_length))
    >>> val = RS.get_next_value()
    >>> print(val)
    >>> sequence_length = RS.get_len()
    >>> print('Length of the pre-generated sequence after returning the first value: {}'.format(sequence_length))

    """
    def __init__(self, nsamples, offset, amplitude, fband):
        self.nsamples = nsamples
        self.nsines = 30
        self.points_list = []
        self.fband = np.array(fband, dtype=float)
        self.amplitude = amplitude / self.nsines
        self.offset = offset

    def add_values(self):
        """
        Superposition `nsines` sine waves. Each of them has a random phase and frequency.
        """
        sample = np.linspace(0, self.nsamples)
        phases = np.random.rand(self.nsines)
        frequencies = self.fband[0] + np.random.rand(self.nsines) * (self.fband[1] - self.fband[0])
        amplitudes = (np.random.rand(self.nsines) - 0.5) * self.amplitude
        # y = A * sin(2ft * pi - phase_rad)
        sin_func = lambda x, ampl, frq, ph: ampl * np.sin(2. * np.pi * (frq * x - ph))

        sine_waves = []
        plt.figure()
        for i in range(self.nsines):
            wave = [sin_func(x, amplitudes[i], frequencies[i], phases[i]) for x in
                    np.linspace(0, self.nsamples, self.nsamples)]
            sine_waves.append(wave)

        self.points_list = list(np.sum(sine_waves, axis=0))
        self.points_list = [p + self.offset for p in self.points_list]
        if PLOTTING:
            plt.figure()
            plt.plot(self.points_list)
            plt.ylabel('gravity')
            plt.xlabel('# changes')
            plt.show()

    def get_next_value(self):
        """
        Return the next value from the sequence of generated values.

        Returns
        -------
        float
        """
        if self.get_len() < 1:
            self.add_values()
        return self.points_list.pop()

    def get_len(self):
        """
        Return the length of the sequence of generated values, that have not been used during an experiment yet.

        Returns
        -------
        float
        """
        return len(self.points_list)


class RandomIntSteps():
    """
        Generate sequence of integer values with discrete value switches.

        The value and the duration of constant value are randomly chosen from the intervals `value_interval` and
        `time_interval`

        Attributes
        ----------
        nsamples : int
            Number of samples that shall be generated in a sequence
        points_list : list of floats
            Sequence of random values that represent a smooth random line.
        time_interval : (3,) list of int
            The list defines the number of time steps, after which the parameter value changes.
            - `time_interval`[0] -> minimum number of time steps between two switches
            - `time_interval`[1] -> maximum number of time steps between two switches
            - `time_interval`[2] -> difference of linearly spaced values in the interval
                                    [`time_interval`[0], `time_interval`[1]]
        value_interval : (2,) list of int
            The list defines the number of allowed parameter values: [min_value, max_value]. The upper bound is
            excluded.

        Methods
        -------
        add_values()
            Generate the sequence of values.
        get_next_value()
            Return the next value of a generated sequence. Is only used for testing puposes, but not during the game play.

        get_len()


        Examples
        --------
        >>> RS = RandomIntSteps(nsamples=300, time_interval=[10,20,5], value_interval=[2,5])
        >>> sequence_length = RS.get_len()
        >>> print('Length of the pre-generated sequence after initialization: {}'.format(sequence_length))
        >>> RS.add_values()
        >>> sequence_length = RS.get_len()
        >>> print('Length of the pre-generated sequence after adding 300 values: {}'.format(sequence_length))
        >>> val = RS.get_next_value()
        >>> print(val)
        >>> sequence_length = RS.get_len()
        >>> print('Length of the pre-generated sequence after returning the first value: {}'.format(sequence_length))

        """
    def __init__(self, nsamples, time_interval, value_interval):
        self.nsamples = nsamples
        self.points_list = []
        self.time_interval = time_interval
        self.value_interval = value_interval

    def add_values(self):
        # create random switching time points (switch after 5 to 20 episodes)
        switching_time = list(np.arange(self.time_interval[0], self.time_interval[1], self.time_interval[2]))
        repetitions = list(np.random.choice(switching_time, size=int(self.nsamples / np.mean(self.time_interval))))

        # create values of each step
        values = list(np.random.randint(low=self.value_interval[0], high=self.value_interval[1],
                                        size=int(self.nsamples / np.mean(self.time_interval))))

        points = []
        for i in range(len(repetitions)):
            points += [int(values[i]) for _ in range(repetitions[i])]
        self.points_list = points
        if PLOTTING:
            plt.figure()
            plt.plot(self.points_list, 'r')
            plt.ylabel('background speed')
            plt.xlabel('episode index')
            plt.show()

    def get_next_value(self):
        if self.get_len() < 1:
            self.add_values()
        return self.points_list.pop()

    def get_len(self):
        return len(self.points_list)


class RandomFloatSteps():
    """
        Generate sequence of float values with discrete value switches.

        The value and the duration of constant value are randomly chosen from the intervals `value_interval` and
        `time_interval`

        Attributes
        ----------
        nsamples : int
            Number of samples that shall be generated in a sequence
        points_list : list of floats
            Sequence of random values that represent a smooth random line.
        time_interval : (3,) list of int
            The list defines the number of time steps, after which the parameter value changes.
            - `time_interval`[0] -> minimum number of time steps between two switches
            - `time_interval`[1] -> maximum number of time steps between two switches
            - `time_interval`[2] -> difference of linearly spaced values in the interval
                                    [`time_interval`[0], `time_interval`[1]]
        value_interval : (2,) list of int
            The list defines the number of allowed parameter values: [min_value, max_value]. The upper bound is
            excluded.

        Methods
        -------
        add_values()
            Generate the sequence of values.
        get_next_value()
            Return the next value of a generated sequence. Is only used for testing puposes, but not during the game play.

        get_len()


        Examples
        --------
        >>> RS = RandomIntSteps(nsamples=300, time_interval=[10,20,5], value_interval=[2,5])
        >>> sequence_length = RS.get_len()
        >>> print('Length of the pre-generated sequence after initialization: {}'.format(sequence_length))
        >>> RS.add_values()
        >>> sequence_length = RS.get_len()
        >>> print('Length of the pre-generated sequence after adding 300 values: {}'.format(sequence_length))
        >>> val = RS.get_next_value()
        >>> print(val)
        >>> sequence_length = RS.get_len()
        >>> print('Length of the pre-generated sequence after returning the first value: {}'.format(sequence_length))

    """
    def __init__(self, nsamples, time_interval, value_interval):
        self.nsamples = nsamples
        self.points_list = []
        self.time_interval = time_interval
        self.amplitude = abs(value_interval[0] - value_interval[1])
        self.offset = np.mean(value_interval)

    def add_values(self):
        """
        Create samples of the sequence of random steps
        """
        # Create samples at random switching time points. Switch values after `time_interval[0]` and `time_interval[1]`
        # time steps.
        switching_time = list(np.arange(self.time_interval[0],self.time_interval[1]+1,self.time_interval[2]))
        repetitions = list(np.random.choice(switching_time, size=int(self.nsamples / np.mean(self.time_interval))))

        # create float values in [`value_interval[0]`, `value_interval[1]`] for each step
        values = list(np.random.rand(int(self.nsamples / np.mean(self.time_interval))))
        values = [round((self.amplitude * (v - 0.5) + self.offset), 1) for v in values]
        points = []
        for i in range(len(repetitions)):
            points += [values[i] for _ in range(repetitions[i])]
        self.points_list = points

        if PLOTTING:
            plt.figure()
            plt.plot(self.points_list, 'r')
            plt.ylabel('background speed')
            plt.xlabel('episode index')
            plt.show()

    def get_next_value(self):
        """
        Return a sample from `points_list`

        Returns
        -------
        float
        """
        if self.get_len() < 1:
            self.add_values()
        return self.points_list.pop()

    def get_len(self):
        return len(self.points_list)

class RandomWaveform():
    """
    Generate continuous sequence of 'random' values based on a combination of sine and cosine functions.

    Attributes
    ----------
    nsamples : int
        Number of samples that shall be generated in a sequence
    points_list : list of floats
        Sequence of random values.
    amplitude : float
        Maximum amplitude of the waveform.
    frequency : float
        Frequency of the waveform.

    Methods
    -------
    add_values()
        Generate the sequence of values.
    get_next_value()
        Return the next value of `points_list`.
    get_len()
        Return the length of `points_list`.
    """
    def __init__(self, nsamples, amplitude, frequency):
        self.nsamples = nsamples
        self.points_list = []
        self.amplitude = amplitude
        self.frequency = frequency

    def add_values(self):
        """
        Generate the sequence based on a combination of sine and cosine functions.
        """
        t = np.linspace(0, self.nsamples, self.nsamples)
        waveform = self.amplitude * (np.sin(self.frequency * t) + np.cos(self.frequency * t))
        self.points_list = list(waveform)

    def get_next_value(self):
        """
        Return the next value from the sequence of generated values.

        Returns
        -------
        float
        """
        if self.get_len() < 1:
            self.add_values()
        return self.points_list.pop()

    def get_len(self):
        """
        Return the length of the sequence of generated values.

        Returns
        -------
        int
        """
        return len(self.points_list)

def load_ns_trajectory(ns_name, seed):
    """
    Load a pre-generated trajectory of values

    Parameters
    ----------
    ns_name : str
        Name of the non-stationarity that is represented by the loaded sequence of parameter values.
    seed : int
        Seed with which the sequence was genrated originally. Functions as ID here.

    Returns
    -------
    list of floats or ints

    """
    with open('train_envs/{}_{}.txt'.format(ns_name, seed), 'r') as f:
        ns = simplejson.load(f)
    return ns

def generate_trajectories(ns_type, seed_list, train_steps, test_steps):

    # In the episodic version for meta learning, each episode takes 60 steps ~ 4 tunnels ~ 30s
    # It is important that the discrete nonstationarities switch values at time steps, that are
    # a multiple of T_{episode} = 60 steps.
    # The mapping from #steps to time interval assumes dt=500ms


    """
    Generate the sequences of samples, that are then loaded during the game.

    So far we can generate sequences of the
    Add new types of non-stationarities here if further temporal patterns shall be utilized in experiments.

    Parameters
    ----------
    ns_type : str {"gfNS", "gsNS", "rand_feat"}
    seed_list
    train_steps
    test_steps
    """
    for seed in seed_list:
        for phase in ['train','test']:
            if phase == 'train':
                s = seed
                np.random.seed(seed)
                nsteps = train_steps
                path = 'train_envs'
            elif phase == 'test':
                s = seed + 100
                np.random.seed(s)
                nsteps = test_steps
                path = 'test_envs'
            else:
                print('The phase {} is not implemented yet.'.format(phase))

            # Generate sequences
            if ns_type == "gfNS":
                # --- gfNS ---
                # values:            [0.1, 0.2, ..., 1.4, 1.5]
                # time_interval:     [120 steps, 180steps, 240 steps]
                # which is equivalent to 1, 1.5 and 2 mins (dt = 500ms) or about 8, 12, 16 tunnels
                gfNS = RandomFloatSteps(nsamples=nsteps, time_interval=[120, 240, 60],
                                        value_interval=[0.5, 1.5])  # upper bound is excluded
                gfNS.add_values()
                with open(os.path.join(path, 'gfNS_{}.txt'.format(path,s)), 'w') as f:
                    simplejson.dump(gfNS.points_list, f)

            elif ns_type == "gsNS":
                # --- gsNS ---
                # values:            [0.1, ..., 1.5]
                # freqs of sines:    [0.000012, ..., 0.000023]
                # equiv. period T:   [43200 steps, ..., 86400 steps] ~ [6h - 12h]
                gsNS = Overlayed_RandomSines(nsamples=nsteps, offset=1., amplitude=5, fband=[0.000012, 0.000023])
                gsNS.add_values()
                with open(os.path.join(path, 'gsNS_{}.txt'.format(s)), 'w') as f:
                    simplejson.dump(gsNS.points_list, f)

            elif ns_type == "rand_featf":
                # --- random state feature ---
                # values:            [0.1, ..., 1.]
                # freqs of sines:    [0.00167, ..., 0.05]
                # equiv. period T:   [20 steps, ..., 600 steps] ~ [10s - 5mins]
                rand_feat = Overlayed_RandomSines(nsamples=nsteps, offset=0.5, amplitude=3., fband=[0.0017, 0.05])
                rand_feat.add_values()
                with open(os.path.join(path, 'rand_feat_{}.txt'.format(s)), 'w') as f:
                    simplejson.dump(rand_feat.points_list, f)
            elif ns_type == "random_waveform":
                # --- RandomWaveform ---
                waveform = RandomWaveform(nsamples=nsteps, amplitude=5., frequency=0.05)
                waveform.add_values()
                with open(os.path.join(path, 'random_waveform_{}.txt'.format(s)), 'w') as f:
                    simplejson.dump(waveform.points_list, f)
            else:
                print("The requested type of sequence ({}) has not been implemented yet.".format(ns_type))


if __name__ == '__main__':

    # Generate the sequences for thesis experiments.
    seed_list = np.linspace(1,20)
    total_train_steps = 2e6 + 3
    total_test_steps = 5e5 + 3
    for ns in ["gfNS", "gsNS", "rand_feat", "random_waveform"]:
        generate_trajectories(ns_type=ns, seed_list=seed_list,
                              train_steps=total_train_steps, test_steps=total_test_steps)
