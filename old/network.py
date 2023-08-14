from perceptron import Perceptron
from vector import Vector
import activation

Neuron_type = Perceptron
Neuron_layer = list[ Neuron_type ]


class NeuralNetwork:

    def __init__( self,
                  neuron: Neuron_type,
                  hidden_layer_count: int,
                  hidden_layer_size: int,
                  input_size: int,
                  output_size: int ) -> None:
        
        self.activation = activation.sigmoid
        self.neuron = neuron

        self.hidden_layer_size = hidden_layer_size
        self.input_size = input_size
        self.output_size = output_size

        ## --- INPUT LAYER
        self.input_layer = self._create_layer( 0, self.input_size )  ## this layer is automatically initialized
                                                                     ## with values from input

        ## --- HIDDEN LAYERS
        self.hidden_layers: list[ Neuron_layer ] = [ self._create_layer( self.input_size, self.hidden_layer_size ) ]
        for _ in range( hidden_layer_count - 1 ):
            self.hidden_layers.append( self._create_layer( self.hidden_layer_size, self.hidden_layer_size ) )
        ## ---

        ## --- OUTPUT LAYER
        self.output_layer: Neuron_layer = self._create_layer( self.hidden_layer_size, self.output_size )
        ## ---

        self.layers = [ self.input_layer ] + self.hidden_layers + [ self.output_layer ]

    def _create_layer( self, input_dim: int, size: int ) -> Neuron_layer:
        return [ self.neuron( input_dim, self.activation ) for _ in range( size ) ]

    def get_layers( self ) -> list[ Neuron_layer ]:
        return self.layers

    def classify( self, v: Vector ) -> None:
        assert len( v ) == self.input_size, "Invalid input size"

        ## set input_layer
        for i in range( self.input_size ):
            self.input_layer[ i ].set_value( v[ i ] )

        processing = v
        for layer in self.layers[ 1: ]:

            to_process = []
            for neuron in layer:
                neuron.compute( processing )
                to_process.append( neuron.get_value() )

            processing = Vector( to_process )

    def get_output( self ) -> Vector:
        
        res = Vector( [ 0 for _ in range( self.output_size ) ] )
        for i, neuron in enumerate( self.output_layer ):
            res[ i ] = neuron.get_value()

        return res

