MAP_PERIODICITY_TO_SEASONALITY = {
    "daily": {
        "daily_seasonality": True
    },
    "weekly": {
        "weekly_seasonality": True
    },
    "monthly": {
        "monthly_seasonality": True
    }
}

MAP_PERIODICITY_TO_SEASONALITY_PARAMETERS = {
    'monthly': {
        'period': 30.5,
        'fourier_order': 5
    },
    'weekly': {
        'period': 7,
        'fourier_order': 3
    },
    'daily': {
        'period': 1,
        'fourier_order': 1
    },
}

MAP_PERIODICITY_TO_FREQUENCY = {
    'monthly': {
        'freq': 'M'
    },
    'weekly': {
        'freq': 'W'
    },
    'daily': {
        'freq': 'D'
    },
}


def get_seasonality(periodicity):
    if periodicity not in MAP_PERIODICITY_TO_SEASONALITY:
        raise ValueError("Invalid periodicity")
    return MAP_PERIODICITY_TO_SEASONALITY[periodicity]


def get_seasonality_parameters(periodicity):
    if periodicity not in MAP_PERIODICITY_TO_SEASONALITY_PARAMETERS:
        raise ValueError("Invalid periodicity")
    return MAP_PERIODICITY_TO_SEASONALITY_PARAMETERS[periodicity]


def get_frequency_parameters(periodicity):
    if periodicity not in MAP_PERIODICITY_TO_FREQUENCY:
        raise ValueError("Invalid periodicity")
    return MAP_PERIODICITY_TO_FREQUENCY[periodicity]
