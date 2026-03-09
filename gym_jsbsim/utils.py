import functools
import math
import operator
from typing import Tuple
from gym_jsbsim.aircraft import cessna172P, a320, f15, f16
from typing import Dict, Iterable


class AttributeFormatter(object):
    """
    Replaces characters that would be illegal in an attribute name

    Used through its static method, translate()
    """
    ILLEGAL_CHARS = '\-/.'
    TRANSLATE_TO = '_' * len(ILLEGAL_CHARS)
    TRANSLATION_TABLE = str.maketrans(ILLEGAL_CHARS, TRANSLATE_TO)

    @staticmethod
    def translate(string: str):
        return string.translate(AttributeFormatter.TRANSLATION_TABLE)


def get_env_id(task_type, aircraft, shaping, enable_flightgear) -> str:
    """
    Creates an env ID from the environment's components

    :param task_type: Task class, the environment's task
    :param aircraft: Aircraft namedtuple, the aircraft to be flown
    :param shaping: HeadingControlTask.Shaping enum, the reward shaping setting
    :param enable_flightgear: True if FlightGear simulator is enabled for visualisation else False
     """
    if enable_flightgear:
        fg_setting = 'FG'
    else:
        fg_setting = 'NoFG'
    return f'JSBSim-{task_type.__name__}-{aircraft.name}-{shaping}-{fg_setting}-v0'


def get_env_id_kwargs_map() -> Dict[str, Tuple]:
    """ Returns all environment IDs mapped to tuple of (task, aircraft, shaping, flightgear) """
    # lazy import to avoid circular dependencies
    from gym_jsbsim.tasks import Shaping, HeadingControlTask, TurnHeadingControlTask

    map = {}
    for task_type in (HeadingControlTask, TurnHeadingControlTask):
        for plane in (cessna172P, a320, f15, f16):
            for shaping in (Shaping.STANDARD, Shaping.EXTRA, Shaping.EXTRA_SEQUENTIAL):
                for enable_flightgear in (True, False):
                    id = get_env_id(task_type, plane, shaping, enable_flightgear)
                    assert id not in map
                    map[id] = (task_type, plane, shaping, enable_flightgear)
    return map


def product(iterable: Iterable):
    """
    Multiplies all elements of iterable and returns result

    ATTRIBUTION: code provided by Raymond Hettinger on SO
    https://stackoverflow.com/questions/595374/whats-the-function-like-sum-but-for-multiplication-product
    """
    return functools.reduce(operator.mul, iterable, 1)


def reduce_reflex_angle_deg(angle: float) -> float:
    """ Given an angle in degrees, normalises in [-179, 180] """
    # ATTRIBUTION: solution from James Polk on SO,
    # https://stackoverflow.com/questions/2320986/easy-way-to-keeping-angles-between-179-and-180-degrees#
    new_angle = angle % 360
    if new_angle > 180:
        new_angle -= 360
    return new_angle


def offset_geodetic_position(latitude_deg: float,
                             longitude_deg: float,
                             north_m: float = 0.0,
                             east_m: float = 0.0) -> Tuple[float, float]:
    """
    Applies a small local north/east offset to a geodetic position.

    This uses an equirectangular approximation, which is sufficient for the
    short formation offsets used by the shared-world multi-aircraft wrapper.
    """
    metres_per_deg_lat = 111_320.0
    metres_per_deg_lon = max(1e-9, metres_per_deg_lat * math.cos(math.radians(latitude_deg)))
    next_lat_deg = latitude_deg + (north_m / metres_per_deg_lat)
    next_lon_deg = longitude_deg + (east_m / metres_per_deg_lon)
    return next_lat_deg, next_lon_deg
