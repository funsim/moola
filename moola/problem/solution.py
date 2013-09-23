class Solution(dict):
    ''' The Solution of an optimisation run. ''' 

    def __init__(self):
        self["Optimizer"] = None
        self["Number of iterations"] = None
        self["Number of functional evaluations"] = None
        self["Number of functional gradient evaluations"] = None
