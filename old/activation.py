import math

def sigmoid( x: float ) -> float:
    return 1 / (1 + math.e ** ( -x ) )

def treshold( x: float ) -> float:
    return 1 if x >= 0 else 0

def relu( x: float ) -> float:
    return x if x >= 0 else 0
