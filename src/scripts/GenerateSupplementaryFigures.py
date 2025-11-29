"""
Generate all supplementary figures for the paper at once.

copyright: @Thomas Hensel, 2023
"""
import os
import shutil
script_name = os.path.basename(__file__)

import matplotlib.pyplot as plt

from lib.config import ROOT_DIR
from lib.plotting.artefacts import Figures
from src.figures.StaticDataFigure import MethodsCorrelationMatrix
from src.figures.StaticDataFigure import BatchHexbinCorrelations
from src.figures.StaticDataFigure import BatchHistogramsCumulativePMF
from src.figures.SimulationFigure import SupplementaryMultifluxFigure
from src.figures.LineScanPrincipleFigure import LineScanSI
from src.figures.ControlMethodFigure import ControlMethodFigure
from src.figures.StaticDataFigure import MinMaxComparison
from src.figures.TheoryFigure import SupplementaryCRBFigure

# List of scripts to run
out = os.path.join(ROOT_DIR,'output/')
os.makedirs(out,exist_ok=True)

kwargs = dict(plot_style='nature', color_scheme='default',show=False)

fig = MethodsCorrelationMatrix.make_figure(**kwargs)
Figures().save_fig(fig, 'MethodsCorrelationMatrix_SI',out_path=out)
plt.close()

figs, names = BatchHexbinCorrelations.make_figure(**kwargs)
for fig, name in zip(figs, names):
    Figures().save_fig(fig, name,out_path=out)
    plt.close()

figs, names = BatchHistogramsCumulativePMF.make_figure(**kwargs)
for fig, name in zip(figs, names):
    Figures().save_fig(fig, name,out_path=out)
    plt.close()

fig = LineScanSI.make_figure(**kwargs)
Figures().save_fig(fig, 'LineScanPrinciple_SI',out_path=out)
plt.close()

fig = ControlMethodFigure.make_figure(**kwargs)
Figures().save_fig(fig, 'ControlMethod_SI',out_path=out)
plt.close()

fig = SupplementaryMultifluxFigure.make_figure(**kwargs)
Figures().save_fig(fig, 'SupplementaryMultiflux_SI',out_path=out)
plt.close()

fig = MinMaxComparison.make_figure(**kwargs)
Figures().save_fig(fig, 'MinMaxLineScanComparison_SI',out_path=out)
plt.close()

fig = SupplementaryCRBFigure.make_figure(**kwargs)
Figures().save_fig(fig, 'CRB_N_SBR_SI',out_path=out)
plt.close()

print('All figures generated.')