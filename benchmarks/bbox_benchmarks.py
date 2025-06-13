from scarlet2.bbox import Box


def time_bbox_creation():
    """Basic timing benchmark for Box creation"""
    _ = Box((15, 15))


def mem_bbox_creation():
    """Basic memory benchmark for Box creation"""
    _ = Box((15, 15))


class BoxSuite:
    """Suite of benchmarks for methods of the Box class"""

    params = [2, 16, 256, 2048]

    def setup(self, edge_length):
        """Create a Box, for each of the different edge_lengths defined in the
        `params` list."""
        self.bb = Box((edge_length, edge_length))

    def time_bbox_contains(self, edge_length):
        """Timing benchmark for `contains` method. Note that `edge_length` is
        unused by this method."""
        this_point = (6, 7)
        self.bb.contains(this_point)

    def mem_bbox_contains(self, edge_length):
        """Memory benchmark for `contains` method. Note that `edge_length` is
        unused by this method."""
        this_point = (6, 7)
        self.bb.contains(this_point)
