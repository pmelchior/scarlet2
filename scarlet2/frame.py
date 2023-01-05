from dataclasses import dataclass
from .bbox import Box
@dataclass
class Frame:
    bbox: Box

    def __hash__(self):
        return hash(self.bbox)
