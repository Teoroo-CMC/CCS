import json
import spline_functions as sf


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

    def __init__(self, name, Rcut=None, Rmin=None, Nknots=None, Dismat=None, Cons=None):
        super().__init__(**Cons)
        self.name = name
        self.Rcut = Rcut
        self.Rmin = Rmin
        self.Nknots = Nknots
        self.Dismat = Dismat
        try:
            self.config_size = sum(1 for line in open(self.Dismat))
        except FileNotFoundError:
            self.config_size= 1
        self.dx = (self.Rcut - self.Rmin)/self.Nknots
        self.cols = self.Nknots + 1
        self.v = self.get_V()

    def get_V(self):
        return sf.spline_energy_model(self.Rcut,self.Rmin,self.Dismat,self.cols,self.dx,self.config_size)


def objective(twobody,STO):




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


with open('input.json') as json_file:
    data = json.load(json_file)
    atom_pairs = []
    for key, values in data['Twobody'].items():
        # Append dismat to the dictionary
        print(values) 
        atom_pairs.append(Twobody(key, **values))
        print (atom_pairs[1].__dict__)



