#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 09:44:15 2021

@author: administrator
"""

import pyrqa
from pyrqa.analysis_type import Cross
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.opencl import OpenCL
import pandas as pd 

#%%
data_points_x = pd.Series(signal.resample_poly(ecg[0:500*60*10], 75, 500))
time_series_x = TimeSeries(data_points_x,
                           embedding_dimension=2,
                           time_delay=1)
data_points_y = pd.Series(ppg[0:75*60*10])
time_series_y = TimeSeries(data_points_y,
                           embedding_dimension=2,
                           time_delay=2)

time_series = (time_series_x,
               time_series_y)
settings = Settings(time_series,
                    analysis_type=Cross,
                    neighbourhood=FixedRadius(0.73),
                    similarity_measure=EuclideanMetric,
                    theiler_corrector=0)
computation = RQAComputation.create(settings,
                                    verbose=True)
result = computation.run()
result.min_diagonal_line_length = 2
result.min_vertical_line_length = 2
result.min_white_vertical_line_length = 2
print(result)
from pyrqa.computation import RPComputation
from pyrqa.image_generator import ImageGenerator
computation = RPComputation.create(settings)
result = computation.run()