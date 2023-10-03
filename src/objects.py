from typing import TypedDict
from enum import Enum


class CollisionLocation(Enum):
    LEFT = 1
    RIGHT = 2
    CENTER = 3


class Collision(TypedDict):
    detected: bool
    location: CollisionLocation
    distance: float
