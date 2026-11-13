import sys
import inspect
import os


current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
great_grandparent_dir = os.path.dirname(grandparent_dir)
great_great_grandparent_dir = os.path.dirname(great_grandparent_dir)

sys.path[:0] = [current_dir, parent_dir, grandparent_dir, great_grandparent_dir, great_great_grandparent_dir]

from .test import *
from .vis import *
from.test_utils import *
from test_files.Fluid_change_detection import *