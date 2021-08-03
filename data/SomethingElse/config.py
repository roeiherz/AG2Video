action_to_number_of_instances = ['Putting [something] on a surface',
                                 'Moving [something] up',
                                 'Pushing [something] from left to right',
                                 'Moving [something] down',
                                 'Pushing [something] from right to left',
                                 'Covering [something] with [something]',
                                 'Uncovering [something]',
                                 'Taking [one of many similar things on the table]',
                                 "__padding__"]
action_to_num_objects = {
    'Putting [something] on a surface': 2,
    'Moving [something] up': 2,
    'Pushing [something] from left to right': 2,
    'Moving [something] down': 2,
    'Pushing [something] from right to left': 2,
    'Covering [something] with [something]': 3,
    'Uncovering [something]': 2,
    'Taking [one of many similar things on the table]': 2,
}

# action not in that list is dropped
valid_actions = [
    "Putting [something] on a surface",
    "Moving [something] up",
    "Pushing [something] from left to right",
    "Moving [something] down",
    "Pushing [something] from right to left",
    "Covering [something] with [something]",
    "Uncovering [something]",
    "Taking [one of many similar things on the table]",
    "__padding__"]
