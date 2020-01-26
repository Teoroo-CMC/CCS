import numpy as np
import logging

logger= logging.getLogger(__name__)
def spline_construction(rows,cols,dx):
       """ Form Spline coefficient matrices 
       
       Args:
           rows -- int
           The number of rows of the matrix
           cols -- int 
           The number of columns of the matrix 
           dx   -- int
           The size of the interval
   
       Returns:
           A   -- Matrix with a coefficients for spline
           B   -- Matrix with b coefficients for spline
           C   -- Matrix with c coefficients for spline
           D   -- Matrix with d coefficients for spline
       
       """


   
       C = np.zeros((rows, cols), dtype=float)
       np.fill_diagonal(C, 1, wrap=True)
       C = np.roll(C, 1, axis=1)
       
   
       D = np.zeros((rows, cols), dtype=float)
       i, j = np.indices(D.shape)
       D[i == j] = -1
       D[i == j - 1] = 1
       D = D / dx
   
       B = np.zeros((rows, cols), dtype=float)
       i, j = np.indices(B.shape)
       B[i == j] = -0.5
       B[i < j] = -1
       B[j == cols - 1] = -0.5
       B = np.delete(B, 0, 0)
       B = np.vstack((B, np.zeros(B.shape[1])))
       B = B * dx
   
   
       A = np.zeros((rows, cols), dtype=float)
       tmp = 1 / 3.0
       for row in range(rows - 1, -1, -1):
           A[row][cols - 1] = tmp
           tmp = tmp + 0.5
           for col in range(cols - 2, -1, -1):
               if row == col:
                   A[row][col] = 1 / 6.0
               if col > row:
                   A[row][col] = col - row
   
       A = np.delete(A, 0, 0)
       A = np.vstack((A, np.zeros(A.shape[1])))
       A = A * dx * dx
   
       return  C, D, B, A

def spline_eval012(a,b,c,d,r, Rcut, Rmin, dx, x):
    ''' This function returns cubic spline value given certain distances '''
    

    if r == Rmin:
        index = 1
    else:
        index = int(np.ceil((r-Rmin) / dx))

    if index >= 1:
        dr = r - x[index]
        f0 = a[index-1] + dr*( b[index-1] + dr*(0.5*c[index-1] + (d[index-1]*dr/3.0)))
        f1 = b[index-1] + dr*(c[index-1] + (0.5*d[index-1]*dr))
        f2 = c[index-1] + d[index-1]*dr
        print('value of f0'+str(f0))
        return f0, f1, f2
    else:
        raise ValueError(' r < Rmin')

def spline_energy_model(Rcut,Rmin,df,cols,dx,size,x):
    C,D,B,A = spline_construction(cols-1,cols,dx)
    logger.debug(" Number of configuration for v matrix: %s",size)
    v = np.zeros((size, cols)) 
    indices=[]
    for config in range(size):
        distances = [ i for i in df[config, :] if i <= Rcut and i >= Rmin ]
        u = 0
        for r in distances:
            index = int(np.ceil(np.around(((r-Rmin) / dx),decimals=5)))
            indices.append(index)
            delta = r - x[index]
            a = A[index - 1]
            b = B[index - 1] * delta
            d = D[index - 1] * np.power(delta, 3) / 6.0
            c_d = C[index - 1] * np.power(delta, 2) / 2.0
            u = u + a + b + c_d + d

        v[config, :] = u
    logger.debug("\n V matrix :%s",v)    
    return v

def write_splinerep(fname, expcoeffs, splcoeffs, rr, rcut):

    delta=0.01
    fp = open(fname,'w')
    fp.write('Spline\n')
    fp.write('{:d} {:4.3f}\n'.format(len(rr),rcut+delta))
    fp.write('{:15.8E} {:15.8E} {:15.8E}\n'.format(*expcoeffs))
   
    splcoeffs_format = ' '.join(['{:6.3f}'] * 2 + ['{:15.8E}'] * 4) + '\n'
    for ir in range(len(rr) - 1):
        rcur = rr[ir]
        rnext = rr[ir + 1]
        fp.write(splcoeffs_format.format(rcur, rnext, *splcoeffs[ir]))
    poly5coeffs_format = ' '.join(['{:6.3f}'] * 2 + ['{:15.8E}'] * 6) + '\n'
    fp.write(poly5coeffs_format.format(rr[-1],rr[-1]+delta, 0.0, 0.0, 0.0,0.0,0.0,0.0))
    fp.close()
    print_io_log(fname, 'Repulsive in Spline format')




def get_spline_coeffs(xx, yy, deriv0, deriv1):
    'Spline coefficients for a spline with given 1st derivatives at its ends.'

    nn = len(xx)
    kk = xx[1:] - xx[:-1]
    mu = kk[:-1] / (kk[:-1] + kk[1:])
    mun = 1.0
    alpha = 1.0 - mu
    alpha0 = 1.0
    mu = np.hstack((mu, [ mun ]))
    alpha = np.hstack(([ alpha0 ], alpha))
    dd = 6.0 / (kk[:-1] + kk[1:]) * ((yy[2:] - yy[1:-1]) / kk[1:] 
                                     - (yy[1:-1] - yy[0:-2]) / kk[:-1])
    d0 = 6.0 / kk[0] * ((yy[1] - yy[0]) / kk[0] - deriv0)
    dn = 6.0 / kk[-1] * (deriv1 - (yy[-1] - yy[-2]) / kk[-1])
    dd = np.hstack((d0, dd, dn))
    mtx = 2.0 * np.identity(nn, dtype=float)
    for ii in range(nn - 1):
        mtx[ii,ii+1] = alpha[ii]
        mtx[ii+1,ii] = mu[ii]
    mm = linalg.solve(mtx, dd)
    c0 = yy[:-1]
    c1 = (yy[1:] - yy[:-1]) / kk - (2.0 * mm[:-1] + mm[1:]) / 6.0 * kk
    c2 = mm[:-1] / 2.0
    c3 = (mm[1:] - mm[:-1]) / (6.0 * kk)
    mtx = np.array([ c0, c1, c2, c3 ])
    return np.transpose(mtx)



def append_spline (fin, fspl, fout):
    """Take electronic part from fin add fspl and write to fout."""
    with open(fspl, 'r') as f:
        spline = f.readlines()
    with open(fin, 'r') as f:
        skf = f.readlines()
    newskf = []
    for line in skf:
        if line == SPLINETAG:
            break
        else:
            newskf.append(line)
    with open(fout, 'w') as f:
        f.writelines(newskf)
        f.write('\n')         # do we need this line?
        f.writelines(spline)  # assuming fspl already has SPLINETAG

class Twobody(object):
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

    def __init__(self, name, Dismat,Nconfigs, Rcut=None, Rmin=None, Nknots=None,Nswitch=None):
        self.name = name
        self.Rcut = Rcut
        self.Rmin = Rmin
        self.Nknots = Nknots
        self.Nswitch = Nswitch
        self.Dismat = Dismat
        self.Nconfigs = Nconfigs
        self.dx = (self.Rcut - self.Rmin)/self.Nknots
        self.cols = self.Nknots + 1
        self.interval = np.linspace(self.Rmin,self.Rcut,self.cols,dtype=float)
        self.v = self.get_V()


    def get_V(self):
        return spline_energy_model(self.Rcut,self.Rmin,self.Dismat,self.cols,self.dx,self.Nconfigs,self.interval)




