from dataclasses import dataclass
from typing import List
import numpy as np
import math

@dataclass
class Point3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_array(self) -> np.ndarray:
        """Return the point as a NumPy array [x, y, z]."""
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def to_list(self) -> List[float]:
        """Return the point as a list [x, y, z]."""
        return [self.x, self.y, self.z]

    def distance_to(self, other: "Point3D") -> float:
        """Compute Euclidean distance to another Point3D."""
        return math.sqrt((self.x - other.x) ** 2 +
                         (self.y - other.y) ** 2 +
                         (self.z - other.z) ** 2)

    def __add__(self, other: "Point3D") -> "Point3D":
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Point3D") -> "Point3D":
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> "Point3D":
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)

    def __repr__(self) -> str:
        return f"Point3D(x={self.x:.3f}, y={self.y:.3f}, z={self.z:.3f})"
