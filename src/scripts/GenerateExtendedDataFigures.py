"""
Generate all extended data figures for the paper at once.

copyright: @Thomas Hensel, 2023
"""

import os, sys
import shutil
script_name = os.path.basename(__file__)

import matplotlib.pyplot as plt

from lib.plotting.artefacts import Figures
from src.figures.StaticDataFigure import LineScanDataFigure


# List of scripts to run
base_dir = os.path.split(__file__)[0]
out = base_dir+'/tmp/'
if os.path.exists(out):
    shutil.rmtree(out)
os.makedirs(out,exist_ok=True)

kwargs = dict(plot_style='nature', color_scheme='default',show=False)
#kwargs = {'plot_style': 'dark_background', 'color_scheme': 'dark_background'}

fig = LineScanDataFigure.make_figure(**kwargs)
Figures().save_fig(fig, 'LineScan_ExtendedDataFigure',meta={'generating script': script_name},out_path=out)
plt.close()

print('All figures generated.')