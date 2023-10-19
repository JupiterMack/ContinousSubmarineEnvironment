from gym.envs.registration import register

"""
Register versions of ContinuousFlappyBird (CFB) as gym envs.

Each CFB version is registered as a 'nonterminating' version (time limit = 10M interactions) and as a clipped 
version (600 interactions). 

Here only envs that include a single modification from the set of possible modifications have been registered:
-Original game, 
-CFB + random features, 
-CFB with random features and additve noise on state representation and 
-CFB with random features and non-stationary model dynamics
Add further envs as needed here. 
Naming convention so far is <game>-<non-stationarity>-nl<noise_level>-nrf<nrandfeat>-<phase>-v0

A remark for future work: All combinations of modifications other than gfNS + gsNS are possible as both 
non-stationarities change the internal gravity parameter and the action of one non-stationarity would be overwritten 
by the other. 
"""

for game in ['ContFlappyBird']:
    nondeterministic = False
    register(
        id='{}-v1'.format(game),
        entry_point='envs.nenvironment:PLEEnv_state',
        kwargs={'game_name': game, 'display_screen': False},
        #tags={'wrapper_config.TimeLimit.max_episode_steps': 10000000},
        max_episode_steps = 10000000 ,
        nondeterministic=nondeterministic,)

    # CLIPPED ENVIRONMENT
    register(
        id='{}-v3'.format(game),
        entry_point='envs.nenvironment:PLEEnv_state',
        kwargs={'game_name': game, 'display_screen': False},
        #tags={'wrapper_config.TimeLimit.max_episode_steps': 3000},
        max_episode_steps = 3000 ,
        nondeterministic=nondeterministic, )

    for p in ['train', 'test']:
        # Random features
        for nrf in [0, 1, 2, 3, 4]:
            register(
                id='{}-nrf{}-{}-v0'.format(game, nrf, p),
                entry_point='envs.nenvironment:PLEEnv_state',
                kwargs={'game_name': game, 'display_screen': False, 'nrandfeatures': nrf, 'phase': p},
                #tags={'wrapper_config.TimeLimit.max_episode_steps': 10000000},
                max_episode_steps = 10000000 ,
                nondeterministic=nondeterministic, )

            register(
                id='{}-clip-nrf{}-{}-v0'.format(game, nrf, p),
                entry_point='envs.nenvironment:PLEEnv_state',
                kwargs={'game_name': game, 'display_screen': False, 'nrandfeatures': nrf, 'phase': p},
                #tags={'wrapper_config.TimeLimit.max_episode_steps': 600},
                max_episode_steps = 600 ,
                nondeterministic=nondeterministic, )

            # Noise
            for nl in [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3,
                       0.35, 0.4, 0.5]:
                register(
                    id='{}-nl{}-nrf{}-{}-v0'.format(game, nl, nrf, p),
                    entry_point='envs.nenvironment:PLEEnv_state',
                    kwargs={'game_name': game, 'display_screen': False, 'noise_level': nl},
                    #tags={'wrapper_config.TimeLimit.max_episode_steps': 10000000},
                    max_episode_steps = 10000000 ,
                    nondeterministic=nondeterministic, )

                register(
                    id='{}-clip-nl{}-nrf{}-{}-v0'.format(game, nl, nrf, p),
                    entry_point='envs.nenvironment:PLEEnv_state',
                    kwargs={'game_name': game, 'display_screen': False, 'noise_level': nl},
                    #tags={'wrapper_config.TimeLimit.max_episode_steps': 600},
                    max_episode_steps = 600 ,
                    nondeterministic=nondeterministic, )

            # nonstationary
            for ns in ['gfNS', 'gsNS', 'hNS', 'random_waveform']:
                register(
                    id='{}-{}-nrf{}-{}-v0'.format(game, ns, nrf, p),
                    entry_point='envs.nenvironment:PLEEnv_state',
                    kwargs={'game_name': game, 'display_screen': False, 'nonstationary': ns, 'nrandfeatures': nrf, 'phase': p},
                    #tags={'wrapper_config.TimeLimit.max_episode_steps': 10000000},
                    max_episode_steps = 10000000 ,
                    nondeterministic=nondeterministic, )

                register(
                    id='{}-clip-{}-nrf{}-{}-v0'.format(game, ns, nrf, p),
                    entry_point='envs.nenvironment:PLEEnv_state',
                    kwargs={'game_name': game, 'display_screen': False, 'nonstationary': ns, 'nrandfeatures': nrf, 'phase': p},
                    #tags={'wrapper_config.TimeLimit.max_episode_steps': 600},
                    max_episode_steps = 600 ,
                    nondeterministic=nondeterministic, )

nondeterministic = False
register(
    id='{}-v1'.format('FlappyBird'),
    entry_point='envs.environment:PLEEnv_state',
    kwargs={'game_name': game, 'display_screen': False},
    #tags={'wrapper_config.TimeLimit.max_episode_steps': 10000000},
    max_episode_steps = 10000000 ,
    nondeterministic=nondeterministic, )
