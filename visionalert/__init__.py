from collections import namedtuple

from visionalert.config import load_config, Config

Rectangle = namedtuple("Rectangle", ["start_x", "start_y", "end_x", "end_y"])
Object = namedtuple("Object", ["name", "confidence", "coordinates"])