import network as net
from PIL import Image, ImageDraw, ImageFont
from vector import Vector

WIDTH = 800
HEIGHT = 600
BG = ( 0, 0, 0 )
FG = ( 255, 255, 255 )


def draw( network: net.NeuralNetwork ) -> Image:

    im = Image.new( "RGB", ( WIDTH, HEIGHT ), BG )
    dr = ImageDraw.Draw( im )

    layers = network.get_layers()
    n = len( layers )

    biggest_layer = max( layers, key=lambda x: len(x) )
    nr = HEIGHT // len( biggest_layer ) // 3

    shift = WIDTH // n
    left = 0
    right = shift

    prev_layer = []

    for layer in layers:

        cx = ( left + right ) // 2
        left, right = right, right + shift
        draw_layer( dr, layer, nr, cx, prev_layer )

    im.show()

def draw_layer( dr: ImageDraw.Draw,
                layer: net.Neuron_layer,
                nr: int,
                cx: int,
                prev_layer: list[ tuple[ int, int ] ] ) -> None:

    n = len( layer )

    high = 0
    low = HEIGHT // n
    shift = low
    curr_layer = []

    for neuron in layer:

        cy = ( high + low ) // 2

        curr_layer.append( (cx, cy) )

        for x, y in prev_layer:
            dr.line( [ (cx, cy), (x, y) ], fill=FG, width=1 )

        high, low = low, low + shift
        draw_neuron( dr, neuron, cx, cy, nr )

    prev_layer.clear()
    prev_layer.extend( curr_layer )


def draw_neuron( dr: ImageDraw.Draw,
                 neuron: net.Neuron_type,
                 cx: int,
                 cy: int,
                 nr: int ) -> None:

    dr.ellipse( [ (cx - nr, cy - nr), (cx + nr, cy + nr) ], fill=FG, outline=FG )
    # text = str( round( neuron.get_value(), 3 ) )
    # font_size = nr // 2
    # font = ImageFont.truetype( "arial.ttf", font_size )
    # dr.text( ( cx - font_size, cy - font_size // 2 ), text=text, font=font, align ="center")


net = net.NeuralNetwork( net.Perceptron, 2, 7, 3, 2 )
draw( net )
