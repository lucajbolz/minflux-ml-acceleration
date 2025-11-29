import os
script_name = os.path.basename(__file__)

import matplotlib as mpl
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import matplotlib.pyplot as plt

from matplotlib.transforms import TransformedBbox, Bbox
from matplotlib.image import BboxImage
from matplotlib.legend_handler import HandlerBase

import seaborn as sns
import numpy as np


def figsize(scale, nplots = 1):
    fig_width_pt = 390.0                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = nplots*fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

pgf_with_latex = {                      # setup matplotlib to use latex for output
    #"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    #"text.usetex": True,                # use LaTeX to write all text
    #"font.family": "serif",
    #"font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
    #"font.sans-serif": [],
    #"font.monospace": [],
    "axes.labelsize": 7,               # LaTeX default is 10pt font.
    "font.size": 7,
    "legend.fontsize": 7,               # Make the legend/label fonts a little smaller
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.figsize": figsize(1.0),     # default fig size of 0.9 textwidth
    'legend.fancybox' : False
    ,'lines.linewidth' : 2.5
    #"pgf.preamble": [
    #    r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
    #    r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
    #    ]
    }

class Style:
    """
    Class for managing plot styles and settings.

    :param style: Style name, default is 'nature'.
    :type style: str
    :param color_scheme: Color scheme name, default is 'default'.
    :type color_scheme: str
    """

    def __init__(self,style='nature',color_scheme='default'):
        """
        Initialize the Style object with the specified style and color scheme.

        :param style: Style name.
        :type style: str
        :param color_scheme: Color scheme name.
        :type color_scheme: str
        """
        self.set_style(style,color_scheme)
        pass

    def set_style(self,style,color_scheme):
        """
        Set the plot style and color scheme.

        :param style: Style name.
        :type style: str
        :param color_scheme: Color scheme name.
        :type color_scheme: str
        """
        self._set_rcparams(style)
        self._define_color_scheme(color_scheme)
        pass

    def update(self):
        """
        Allow updating rcparams or colors.
        """
        pass

    def get_figsize(self,scale=1., cols = 1, rows= 1, ratio=(np.sqrt(5.0)-1.0)/2.0):
        """
        Calculate figure size based on scale, columns, rows, and ratio.

        :param scale: Scale factor for figure size, default is 1.
        :type scale: float
        :param cols: Number of columns, default is 1.
        :type cols: int
        :param rows: Number of rows, default is 1.
        :type rows: int
        :param ratio: Aesthetic ratio set to the golden mean, default is (np.sqrt(5.0) - 1.0) / 2.0.
        :type ratio: float

        :return: Figure size [width, height].
        :rtype: list
        """
        # Aesthetic ratio set to golden mean
        fig_width_pt = 512.15                          # Get this from LaTeX using \the\textwidth
        inches_per_pt = 1.0/72.27                       # Convert pt to inch
        fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
        plot_width = fig_width/cols                     # width of single plot
        fig_height = rows*plot_width*ratio        # height of figure in inches
        fig_size = [fig_width,fig_height]
        return fig_size

    def handle_ticks(self, axis,nx=3,ny=3):
        """
        Handle ticks for the given axis.

        :param axis: Axis object.
        :type axis: object
        :param nx: Number of ticks for x-axis, default is 3.
        :type nx: int
        :param ny: Number of ticks for y-axis, default is 3.
        :type ny: int
        """
        # Get the axis limits
        xmin, xmax = axis.get_xlim()
        ymin, ymax = axis.get_ylim()
        
        # Calculate the range for both x and y axes
        x_range = xmax - xmin
        y_range = ymax - ymin
        
        # Calculate the order of magnitude for both x and y ranges
        x_order = np.floor(np.log10(x_range))
        y_order = np.floor(np.log10(y_range))

        if np.abs(x_order)>0:
            # modify plot range to a nice range depending on order of magnitude
            xfac = 10**x_order
            xmin, xmax = np.floor(xmin/xfac)*xfac, np.ceil(xmax/xfac)*xfac
            # Re-Calculate the range for both x and y axes
            x_range = xmax - xmin
            # Re-Calculate the order of magnitude for both x and y ranges
            x_order = np.floor(np.log10(x_range))
        if np.abs(y_order)>0:
            yfac = 10**y_order
            ymin, ymax = np.floor(ymin/yfac)*yfac, np.ceil(ymax/yfac)*yfac
            y_range = ymax - ymin
            y_order = np.floor(np.log10(y_range))

        # determine best nx and ny for ticks, close to suggested nx and ny
        yspacing = y_range/ny
        while not np.isclose(round(yspacing, np.int(np.min([0,y_order]))), y_range / ny, atol=10**y_order):
            new_ny = ny
            idx = 0
            if yspacing < round(y_range / new_ny, .1):
                # Try increasing ny by one and check again
                new_ny = idx + 1
            elif new_ny > 2:
                # Try decreasing ny by one and check again
                new_ny -= 1
            else:
                # If ny is already 2, break the loop and accept the current value
                break
            idx+=1
        yspacing = y_range / ny
        
        x_ticks = np.linspace(xmin,xmax,nx,endpoint=True)
        y_ticks = np.linspace(ymin,ymax,ny,endpoint=True)
        
        # Update axis limits
        axis.set_xlim(xmin, xmax)
        axis.set_ylim(ymin, ymax)
        
        # Set the ticks on the axis
        axis.set_xticks(x_ticks)
        axis.set_yticks(y_ticks)
        pass
    
    def despine_axes(self,axes=[],end_ticks=False):
        """
        Remove spines from the specified axes and optionally disable ticks at the ends.

        :param axes: List of axes objects to despine, default is an empty list.
        :type axes: list
        :param end_ticks: Whether to disable ticks at the ends, default is False.
        :type end_ticks: bool
        """
        #TODO: add option to automatically add endcaps as with seaborn
        # Disable the frame
        for ax in axes:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.tick_params(top=False, labeltop=False, bottom=False, labelbottom=False,left=False,labelleft=False,right=False,labelright=False,direction='in')

    def drop_axes(self, axes=[], dirs=['left','right','bottom','top']):
        """
        Move spines outward by a specified distance for the specified axes.

        :param axes: List of axes objects to drop spines, default is an empty list.
        :type axes: list
        :param dirs: List of directions to drop spines, default is ['left', 'right', 'bottom', 'top'].
        :type dirs: list
        """
        for ax in axes:
            for dir in dirs:
                ax.spines[dir].set_position(('outward', 10))

    def hide_axes(self, axes=[], dirs=['left','right','bottom','top']):
        """
        Hide spines and ticks for the specified axes.

        :param axes: List of axes objects to hide spines and ticks, default is an empty list.
        :type axes: list
        :param dirs: List of directions to hide spines and ticks, default is ['left', 'right', 'bottom', 'top'].
        :type dirs: list
        """
        for ax in axes:
            ax.spines[dirs].set_visible(False)
            for dir in dirs:
                if dir=='top':
                    ax.tick_params(top=False, labeltop=False,which="both")
                if dir=='bottom':
                    ax.tick_params(bottom=False, labelbottom=False,which="both")
                if dir=='left':
                    ax.tick_params(left=False,labelleft=False,which="both")
                if dir=='right':
                    ax.tick_params(right=False,labelright=False,which="both")
    
    def _set_rcparams(self,style):
        """
        Set matplotlib rcParams based on the specified style.

        :param style: Style name.
        :type style: str
        """
        try:
            mpl.style.use(style)
        except:
            params = {}
            param_dict = dict(
                nature = {
                    "pgf.texsystem": "pdflatex"        # change this if using xetex or lautex
                    ,"text.usetex": False
                    ,"font.family": "sans-serif"
                    ,"axes.labelsize": 7               # LaTeX default is 10pt font.
                    ,"font.size": 7
                    ,"legend.title_fontsize":8
                    ,"legend.fontsize": 7               # Make the legend/label fonts a little smaller
                    ,"xtick.labelsize": 7
                    ,"ytick.labelsize": 7
                    ,"figure.figsize": self.get_figsize()     # default fig size of 0.9 textwidth
                    ,'figure.dpi':300
                    ,'legend.fancybox' : False
                    ,'legend.frameon' : False
                    ,'lines.linewidth' : 1.5
                    }
                ,latex = {
                    #"pgf.texsystem": "pdflatex"        # change this if using xetex or lautex
                    "text.usetex": True
                    ,"text.latex.preamble": r"\usepackage{amsmath}"
                    ,"font.family": "serif"
                    ,"font.serif": []                  # blank entries should cause plots to inherit fonts from the document
                    ,"font.sans-serif": []
                    ,"font.monospace": []
                    ,"axes.labelsize": 7               # LaTeX default is 10pt font.
                    ,"font.size": 7
                    ,"legend.title_fontsize":8
                    ,"legend.fontsize": 7               # Make the legend/label fonts a little smaller
                    ,"xtick.labelsize": 7
                    ,"ytick.labelsize": 7
                    ,"figure.figsize": self.get_figsize()     # default fig size of 0.9 textwidth
                    ,'figure.dpi':300
                    ,'legend.fancybox' : False
                    ,'legend.frameon' : False
                    ,'lines.linewidth' : 2.
                    }
            )
            try:
                params = param_dict[style]
                mpl.rcParams.update(params)
            except:
                raise Exception('Failed to find suitable rcParameters')
        pass

    def _define_color_scheme(self,color_scheme=None):
        """
        Define the color scheme for the plot.

        :param color_scheme: Name of the color scheme to use, default is None.
        :type color_scheme: str, optional
        """
        try:
            if color_scheme=='default':
                self.cmap_seq = mpl.cm.get_cmap('afmhot')
                self.cmap_div = mpl.cm.get_cmap('RdBu')
                colormap = self.cmap_div
                
                self.c10 = self.cmap_div(0.15)
                self.c11 = self.cmap_div(0.25)

                self.c20 = self.cmap_div(.85)
                self.c21 = self.cmap_div(.75)#self._generate_lighter_shades(self.c20, num_shades=1, brightness_factor=1.2)[0]

                self.white = "#FFFFFF" # white
                self.dark_white = '#FDF0D5' #dark white

                self.c30 = '#E6A817' # gold
                self.c31 = self._generate_lighter_shades(self.c30, num_shades=1, brightness_factor=1.2)[0]
            else:
                colormap = mpl.cm.get_cmap(color_scheme)
                self.c10 = colormap(0.1)
                self.c11 = self._generate_lighter_shades(self.c10, num_shades=1, brightness_factor=1.2)[0]

                self.c20 = colormap(0.5)
                self.c21 = self._generate_lighter_shades(self.c20, num_shades=1, brightness_factor=1.2)[0]

                self.c30 = colormap(0.9)
                self.c31 = self._generate_lighter_shades(self.c30, num_shades=1, brightness_factor=1.2)[0]

                self.white = "#FFFFFF" # white
                self.dark_white = '#FDF0D5' #dark white
                #self.c30 = '#E6A817' # gold

                self.cmap_seq = colormap
                self.cmap_div = colormap
                pass
            try:
                colors = colormap.colors
            except:
                try:
                    colors = colormap(np.arange(0,colormap.N))
                except:
                    colormap = mpl.cm.get_cmap('viridis')
                    colors = colormap.colors
                    print('Style: Forced switch to viridis colormap!')
            self.s_palette = sns.color_palette(colors)
            self.cmap_seq.set_bad(colors[0])
            self.cmap_div.set_bad(colors[0])
        except:
            pass
        pass

    def _generate_lighter_shades(self, color, num_shades=1, brightness_factor=1.2):
        """
        Generate lighter shades of a given color.

        :param color: The color for which lighter shades are generated.
        :type color: str or tuple
        :param num_shades: Number of lighter shades to generate, default is 1.
        :type num_shades: int, optional
        :param brightness_factor: Factor to adjust brightness, default is 1.2.
        :type brightness_factor: float, optional
        :return: List of lighter shades of the input color.
        :rtype: list
        """
        if '#' in color:#convert to rgb if hex is provided
            h = color.lstrip('#')
            color = tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
        hsv_color = rgb_to_hsv(color[:3])  # Convert RGB to HSV
        lighter_shades = []
        for i in range(num_shades):
            new_brightness = min(hsv_color[2] * brightness_factor, 1.0)
            new_color = hsv_to_rgb((hsv_color[0], hsv_color[1], new_brightness))
            lighter_shades.append(new_color)
            hsv_color = rgb_to_hsv(new_color[:3])
        return lighter_shades



class ImageHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):
        """
        Create artists for the legend.

        :param legend: The legend.
        :type legend: matplotlib.legend.Legend
        :param orig_handle: Original handle.
        :param xdescent: X descent.
        :param ydescent: Y descent.
        :param width: Width.
        :param height: Height.
        :param fontsize: Font size.
        :param trans: Transformation.
        :return: List of created artists.
        """

        # enlarge the image by these margins
        sx, sy = self.image_stretch 

        # create a bounding box to house the image
        bb = Bbox.from_bounds(xdescent - sx,
                              ydescent - sy,
                              width + sx,
                              height + sy)

        tbb = TransformedBbox(bb, trans)
        image = BboxImage(tbb)
        image.set_data(self.image_data)
        self.update_prop(image, orig_handle, legend)
        return [image]

    def set_image(self, *array, image_path=None, image_stretch=(0, 0)):
        """
        Set the image data or path for the handler.

        :param array: Image data.
        :param image_path: Path to the image file.
        :param image_stretch: Stretch margins for the image, default is (0, 0).
        """
        self.image_stretch = image_stretch
        if array:
            self.image_data = array
        elif image_path:
            try:
                self.image_data = plt.imread(image_path)
            except FileNotFoundError:
                raise Exception('File not found')
            except Exception as e:
                raise Exception('Error loading image:', e)