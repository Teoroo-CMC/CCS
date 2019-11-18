import json
import numpy as np
import ccs.spline_functions as sf


class Constraints(object):
    ''' A class representing possible constraints on splines

    Attributes

    '''
    def __init__(self,c=None,RT=None,RF=None,Nswitch=None,ST=None,P=None,G=None,q=None,h=None):
        self.c=c
        self.RT=RT
        self.RF=RF
        self.Nswitch=Nswitch
        self.P = P
        self.q = q
        self.G = G
        self.h = h
    




class Twobody(Constraints):
    ''' Class representing two body 
     Attributes:
    Name -- str 
        The name of the atomic pair (eg: Zn,O,ZnO).
    Rcut -- float   
        The cutoff distance for splines.
    Rmin --  float
        The distance were splines begin.
    dx   -- float
        The gridsize for the splines.
    dismat -- numpy matrix
        The pairwise distances.
    V    -- Numpy Matrix
        The spline energy model matrix( refer to paper)
    c    -- numpy array
        The curvatures for the splines


    '''

    def __init__(self, name, Dismat,Nconfigs, Rcut=None, Rmin=None, Nknots=None, Cons=None):
        super().__init__(**Cons)
        self.name = name
        self.Rcut = Rcut
        self.Rmin = Rmin
        self.Nknots = Nknots
        self.Dismat = Dismat
        self.Nconfigs = Nconfigs
        self.dx = (self.Rcut - self.Rmin)/self.Nknots
        self.cols = self.Nknots + 1
        self.interval = np.linspace(self.Rmin,self.Rcut,self.cols,dtype=float)
        self.v = self.get_V()


    def get_V(self):
        return sf.spline_energy_model(self.Rcut,self.Rmin,self.Dismat,self.cols,self.dx,self.Nconfigs,self.interval)


class Objective():
    ''' Class for constructing the objective'''
    def __init__(self,l_twb,STO,ref_E):
        self.l_twb = l_twp
        self.STO = STO
        self.V=None
    def solver():
        for i in len(self.l_twp):
            self.V= None


#    def get_P():

#    def get_G():
#    def get_q():
#    def solver():




def get_data(filename):
    """ This file reads energy and distances file"""

    size = sum(1 for line in open(filename))
    df = pd.read_csv(filename, header=None, names=range(
        0, 10000), delim_whitespace=True)
    df = df.values
    E_dft = df[0:size, 0]  # No division by 4 required generally only for Si
    b = -1*E_dft  # NNOTE
    dis_mat = np.delete(df, [0], axis=1)
    return b, dis_mat



