import random
from vector import Vector

Sample = tuple[ Vector, int ]

class Perceptron:

    def __init__( self, dim: int ) -> None:
        self.dim = dim
        self.val = random.random( )
        self.weights = Vector( [ random.random( ) for _ in range( dim + 1 ) ] )

    def set_weights( self, v: Vector ) -> None:
        self.weights = v

    def get_weights( self ) -> Vector:
        return self.weights

    def classify( self, vector: Vector ) -> int:
        return 1 if self.weights * vector >= 0 else 0

    def get_dim( self ) -> int:
        return self.dim

    def get_val( self ) -> float:
        return self.val


def learn( p: Perceptron, epochs: int, dataset: list[ Sample ] ) -> None:
    
    n = len( dataset )
    for i in range( epochs ):

        sample, exp = dataset[ i % n ]
        sample = sample.extended( 1 )

        actual = p.classify( sample )

        diff = actual - exp
        old = p.get_weights( )
        p.set_weights( old - diff * sample )
