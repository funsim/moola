class Solution(dict):
    ''' The Solution of an optimisation run. ''' 
    def __str__(self):
        s = ""
        for k, v in self.items():
            s += "%s:\t\t%s\n" % (k, v)
        return s

