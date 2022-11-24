# About matplotlib

# The Backend Layer: The backend layer is the part of matplotlib that talks to the GUI framework.
# It has three built-in abstract classes: FigureCanvas, Renderer, and Event.
# 1) The FigureCanvas is the part of the GUI that displays the figure.
# 2) The Renderer is the part of the GUI that draws the figure.
# 3) The Event is the part of the GUI that handles user input.

# The Artist Layer: The artist layer is the part of matplotlib that draws the figure.
# It is composed of one maing object: the Artist.
# The Artist is the base class for all objects that can be drawn on a canvas.
# There are two types of Artists: primitives and containers/composites.
# 1) Primitives are the basic objects that are drawn on the canvas, such as the Line2D, Rectangle, Text, AxesImage, etc.
# 2) Containers/composites are the objects that contain other Artists, such as the Axes, Axis, Tick, and Figure. Containers also have primitives as children.

# The Scripting Layer: Comprised mainly of pyplot, a scripting interface that works like the artist layer, but is lighter and  easier to use.
# It is also useful for interactive plotting, and for simple cases of programmatic plotting.


# --- Use the artist layer to generate a hisogram ---

# Import FigureCanvas. The "agg" in backenge_agg means that the figure will be rendered to a canvas that is an agg (antigrain) raster image.
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# Import Figure artist
from matplotlib.figure import Figure
import numpy as np
fig = Figure()
canvas = FigureCanvas(fig)

# create 10000 randome numbers using numpy
x = np.random.randn(10000)

# create an axes artist. 111 means 1 row, 1 column, 1st plot
ax = fig.add_subplot(111)

# generate a histgram of the 10000 random numbers, using 100 bins
ax.hist(x, 100)

# add a title to the figure and save it
ax.set_title('Normal distribution with $\mu=0, \sigma=1$')
fig.savefig('matplotlib_histogram_created_with_artist_layer.png')


# --- Use the scripting layer to generate the same histogram ---
import matplotlib.pyplot as plt

plt.hist(x, 100)
plt.title('Normal distribution with $\mu=0, \sigma=1$')
plt.savefig('matplotlib_histogram_created_with_scripting_layer.png')
plt.show()
