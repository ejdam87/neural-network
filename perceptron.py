import random
from vector import Vector


class Perceptron:

    def __init__( self, dim: int ) -> None:
        self.dim = dim
        self.weights = Vector( [ random.random( ) for _ in range( dim + 1 ) ] )

    def set_weights( self, v: Vector ) -> None:
        self.weights = v

    def classify( self, vector: Vector ) -> int:
        return 1 if self.weights * vector >= 0 else 0


train = [ ( Vector( [ -1, 0 ] ), 1 ), ( Vector( [ 0, 1 ] ), 1 ), ( Vector( [ 3, 0 ] ), 0 ) ]
Sample = tuple[ Vector, int ]

def learn( p: Perceptron, epochs: int, dataset: list[ Sample ] ) -> None:
    
    n = len( dataset )
    for i in range( epochs ):

        sample, exp = dataset[ i % n ]
        sample = sample.extended( 1 )

        actual = p.classify( sample )

        diff = actual - exp
        p.weights = p.weights - diff * sample


p = Perceptron( 2 )
p.set_weights( Vector( [0, 1, -1] ) )
learn( p, 6, train )
