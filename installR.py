# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 15:04:11 2019

@author: Jorge C. Lucero

Installs James Ramsay's R package for functional data analysis (fda)
"""

import rpy2.robjects.packages as rpackages

if rpackages.isinstalled('fda'):
    print('Package fda already installed')
else:
    print('Installing fda...')
    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1)  # select first mirror of R packages
    utils.install_packages('fda')
    print("Done!")
