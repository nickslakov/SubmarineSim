import math
class Vector:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dot(self, vec2):
        return self.x * vec2.x + self.y * vec2.y

    def cross(self, vec2):
        return self.x * vec2.y - self.y * vec2.x

    def length(self):
        return math.sqrt(self.x**2 + self.y**2)

    def rotate(self, theta):
        self.x = math.cos(theta) * self.x - math.sin(theta) * self.y
        self.y = math.sin(theta) * self.x + math.cos(theta) * self.y
