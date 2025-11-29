"""
Generate all main text figures for the paper at once.

copyright: @Thomas Hensel, 2023
"""

import os
import shutil
script_name = os.path.basename(__file__)

import matplotlib.pyplot as plt

from lib.plotting.artefacts import Figures
from src.figures.PhenomenologicalFigure import DiffLimitFigure
from src.figures.TheoryFigure import CRBFigure
from src.figures.MeasurementPrinciplesFigure import MeasurementPrinciplesFigure
from src.figures.StaticDataFigure import StaticDataFigure
from src.figures.DynamicDataFigure import MinfluxDynamicFigure
from src.figures.SimulationFigure import MultifluxFigure

# List of scripts to run
base_dir = os.path.split(__file__)[0]
out = base_dir+'/tmp/'
if os.path.exists(out):
    shutil.rmtree(out)
os.makedirs(out,exist_ok=True)

kwargs = dict(plot_style='nature', color_scheme='default',show=False)

fig = DiffLimitFigure.make_figure(**kwargs)
Figures().save_fig(fig, 'DiffractionLimit',meta={'generating script': script_name},out_path=out)
plt.close()

fig = CRBFigure.make_figure(**kwargs)
Figures().save_fig(fig, 'TheoryCRB',meta={'generating script': script_name},out_path=out)
plt.close()

fig = MeasurementPrinciplesFigure.make_figure(**kwargs)
Figures().save_fig(fig, 'MeasurementPrinciple',meta={'generating script': script_name},out_path=out)
plt.close()

fig = StaticDataFigure.make_figure(**kwargs)
Figures().save_fig(fig, 'StaticData',meta={'generating script': script_name},out_path=out)
plt.close()

fig = MinfluxDynamicFigure.make_figure(**kwargs)
Figures().save_fig(fig, 'DynamicData',meta={'generating script': script_name},out_path=out)
plt.close()

fig = MultifluxFigure.make_figure(**kwargs)
Figures().save_fig(fig, 'MultifluxFigure',meta={'generating script': script_name},out_path=out)
plt.close()


print('All figures generated.')