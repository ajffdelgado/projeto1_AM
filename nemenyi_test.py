# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 23:54:01 2018

@author: Delgado
"""

import scikit_posthocs as sp
import pandas as pd
import numpy as np
x = np.array([[79.52,92.06,79.59],[43.38,54.54,46.82],[79.43,88.60,79.57]])
sol =  sp.posthoc_nemenyi_friedman(x)