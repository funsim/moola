class Solution(dict):
    ''' Stores the Solution of an optimisation run. '''

    def __str__(self):
        s = ""
        for k, v in list(self.items()):
            s += "%s:\t\t%s\n" % (k, v)
        return s

