REWARD_BOUNDS = {
    'coinrun': (0., 10.),
    'starpilot': (0., 64.),
    'caveflyer': (0., 12.),
    'dodgeball': (0., 19.),
    'fruitbot': (-5., 32.4),
    'chaser': (0., 13.),
    'miner': (0., 13.),
    'jumper': (0., 10.),
    'leaper': (0., 10.),
    'maze': (0., 10.),
    'bigfish': (0., 32.),  # (0, 40.)
    'heist': (0., 10.),
    'climber': (0., 12.6),
    'plunder': (0., 30.),
    'ninja': (0., 10.),
    'bossfight': (0., 13.),
}

ALL_COMBOS = [
    ("LEFT", "DOWN"),
    ("LEFT",),
    ("LEFT", "UP"),
    ("DOWN",),
    (),
    ("UP",),
    ("RIGHT", "DOWN"),
    ("RIGHT",),
    ("RIGHT", "UP"),
    ("D",),
    ("A",),
    ("W",),
    ("S",),
    ("Q",),
    ("E",),
]

COMBO_STRINGS = ['+'.join(combo) for combo in ALL_COMBOS]

META_ALLOWED_COMBOS = {
    'bigfish': ['LEFT+DOWN', 'LEFT', 'LEFT+UP', 'DOWN',
                'UP', 'RIGHT+DOWN', 'RIGHT', 'RIGHT+UP'],
    'coinrun': ['LEFT', 'RIGHT', 'UP', ''],
    'fruitbot': ['LEFT', 'RIGHT', 'D', ''],
    'jumper': ['LEFT', 'RIGHT', 'UP', '', 'LEFT+UP', 'RIGHT+UP'],
    'climber': ['LEFT', 'RIGHT', 'UP', ''],
    'plunder': ['LEFT', 'RIGHT', 'UP', ''],
    'chaser': ['LEFT', 'RIGHT', 'UP', 'DOWN'],
    'bossfight': ['LEFT+DOWN', 'LEFT', 'LEFT+UP', 'DOWN',
                  'UP', 'RIGHT+DOWN', 'RIGHT', 'RIGHT+UP',
                  '', 'D'],
    'ninja': ['LEFT', 'LEFT+UP', 'UP', 'RIGHT', 'RIGHT+UP',
              'A', 'W' 'D', ''],
    'caveflyer': ['LEFT', 'RIGHT', 'UP', '', 'DOWN', 'D'],
    'dodgeball': COMBO_STRINGS
}
