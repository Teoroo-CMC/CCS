""" This module constructs and solves the spline objective"""
import logging
import itertools

import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from scipy.linalg import block_diag

import ccs.spline_functions as sf

logger = logging.getLogger(__name__)


class Objective():
    """  Objective function for ccs method """

    def __init__(self, l_twb, sto, ref_E, c='C', RT=None, RF=1e-6, switch=False, ST=None):
        """ Generates Objective class object
        
        Args:
            l_twb (list): list of Twobody class objects.
            sto (ndarray): An array containing number of atoms of each type.
            ref_E (ndarray): Reference energies.
            c (str, optional): Type of solver. Defaults to 'C'.
            RT ([type], optional): Regularization Type. Defaults to None.
            RF ([type], optional): Regularization factor. Defaults to 1e-6.
            switch (bool, optional): switch condition. Defaults to False.
            ST ([type], optional): switch search where there is data. Defaults to None.
        """
        self.l_twb = l_twb
        self.sto = sto
        self.ref_E = ref_E
        self.c = c
        self.RT = RT
        self.RF = RF
        self.switch = switch
        self.cols_sto = sto.shape[1]
        self.NP = len(l_twb)
        self.cparams = [self.l_twb[i].cols for i in range(self.NP)]
        self.ns = len(ref_E)
        logger.debug(" The reference energy : \n %s \n Number of pairs:%s", self.ref_E,self.NP)

    @staticmethod
    def solver(P, q, G, h, MAXITER=300, tol=(1e-10, 1e-10, 1e-10)):
        """ The solver for the objective
        
        Args:
            P (matrix): P matrix as per standard Quadratic Programming(QP) notation.
            q (matrix): q matrix as per standard QP notation.
            G (matrix): G matrix as per standard QP notation.
            h (matrix): h matrix as per standard QP notation
            MAXITER (int, optional): Maximum iteration steps. Defaults to 300.
            tol (tuple, optional): Tolerance value of the solution. Defaults to (1e-10, 1e-10, 1e-10).
        
        Returns:
            dictionary: The solution details are present in this dictionary
        """

        solvers.options['maxiters'] = MAXITER
        solvers.options['feastol'] = tol[0]
        solvers.options['abstol'] = tol[1]
        solvers.options['reltol'] = tol[2]
        sol = solvers.qp(P, q, G, h)
        return sol

    def eval_obj(self, x):
        """ mean square error function
        
        Args:
            x (ndarray): The solution for the objective.
        
        Returns:
            float: mean square error.
        """
        return np.format_float_scientific(np.sum((self.ref_E - (np.ravel(self.M.dot(x))))**2)/self.ns, precision=4)

    def plot(self, name, E_model, s_interval, Dismat,s_a, x):
        """ function to plot the results
        
        Args:
            E_model (ndarray): Predicted energies via spline.
            s_interval (list): Spline interval.
            s_a (ndarray): Spline a coeffcients.
            x (ndarrray): The solution array.
        """

        fig = plt.figure()

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(E_model, self.ref_E, 'bo')
        ax1.set_xlabel('Predicted energies')
        ax1.set_ylabel('Ref. energies')
        z = np.polyfit(E_model, self.ref_E, 1)
        p = np.poly1d(z)
        ax1.plot(E_model, p(E_model), 'r--')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.scatter(s_interval, s_a, c=[i < 0 for i in s_a])
        ax2.set_xlabel('Distance')
        ax2.set_ylabel('a coefficients')

        ax3 = fig.add_subplot(2, 2, 3)
        c = [i < 0 for i in x]
        ax3.scatter(s_interval[1:], x, c=c)
        ax3.set_ylabel('c coefficients')
        ax3.set_xlabel('Distance')

        ax4 = fig.add_subplot(2, 2, 4)
        n, bins, patches = plt.hist(x=np.ravel(Dismat), bins=s_interval, color='g', rwidth=0.85)
        ax4.set_ylabel('Frequency of a distance')
        ax4.set_xlabel('Spline interval')
        plt.tight_layout()
        plt.savefig(name+'-summary.png')
        plt.show()
        
    def get_coeffs(self,x,E_model):
        temp=0
        for i in range(self.NP):
            curvatures= x[temp : temp+self.cparams[i]]
            temp=self.cparams[i]
            s_a = np.dot(self.l_twb[i].A, curvatures)
            s_b = np.dot(self.l_twb[i].B, curvatures)
            s_c = np.dot(self.l_twb[i].C, curvatures)
            s_d = np.dot(self.l_twb[i].D, curvatures)

            
    #        splcoeffs = np.hstack((s_a, s_b, s_c, s_d))
            splderivs = sf.spline_eval012(s_a,s_b,s_c,s_d,self.l_twb[i].Rmin,self.l_twb[i].Rcut,self.l_twb[i].Rmin,self.l_twb[i].dx,self.l_twb[i].interval)
            expcoeffs= sf.get_expcoeffs(*splderivs,self.l_twb[i].Rmin)
            print (expcoeffs)
            print (type(expcoeffs))
            expbuf=0.5
            rexp = np.linspace( self.l_twb[i].Rmin -expbuf, self.l_twb[i].Rmin, int(expbuf/self.l_twb[i].dx)+1)
            expvals = sf.get_exp_values(expcoeffs,rexp)
            sf.write_as_nxy('headfit.dat', 'Exponentail head', (rexp, expvals), ('rr', 'exponential head'))
            s_a = np.insert(s_a,0,splderivs[i])
            splcoeffs = sf.get_spline_coeffs(self.l_twb[i].interval,s_a,splderivs[1],0)
            sf.write_splinerep(self.l_twb[i].name+"repulsive.dat", np.array(expcoeffs).tolist(), splcoeffs, self.l_twb[i].interval,self.l_twb[i].Rcut)
    #   #     print (type(splcoeffs))
        #    sf.write_splinecoeffs(self.l_twb[0], splcoeffs)
            self.plot(self.l_twb[i].name, E_model, self.l_twb[i].interval,self.l_twb[i].Dismat, s_a, s_c)
    
    def list_iterator(self):
        tmp = [range(i+1) for i in self.cparams]
       
        N_list = list(itertools.product(*tmp))
        return N_list
        

    def solution(self):
        """ Function to solve the objective with constraints
        """
        self.M = self.get_M()
        logger.debug("\n Shape of M matrix is : %s",self.M.shape)
        P = matrix(np.transpose(self.M).dot(self.M))
        eigvals = np.linalg.eigvals(P)
        logger.debug("Eigenvalues:%s",eigvals)
        logger.info('positive definite:%s',np.all((eigvals >0)))
        print(eigvals)
        print(np.all((eigvals >0)))
        q = -1*matrix(np.transpose(self.M).dot(self.ref_E))
        N_switch_id = 0
        Nswitch_list = self.list_iterator()
       # print(Nswitch_list)
        obj = np.zeros(([x+1 for x in self.cparams]))
        sol_list = []
        if self.l_twb[0].Nswitch is None:
            for N_switch_id in Nswitch_list:
                G = self.get_G(N_switch_id)
                print(N_switch_id)
                logger.debug(
                    "\n Nswitch_id : %s and G matrix:\n %s", N_switch_id, G)
                h = np.zeros(G.shape[1])
                sol = self.solver(P, q, matrix(G), matrix(h))
                obj[N_switch_id] = self.eval_obj(sol['x'])
                print(obj[N_switch_id])
                print(sol['x'])
                sol_list.append(sol)
                

            mse = np.min(obj)
            print (obj)
            opt_sol_index = np.ravel(np.argwhere(obj == mse))
            logger.info("\n The best switch is : %s and mse : %s", opt_sol_index,mse)
            
            G_opt = self.get_G(opt_sol_index)
            opt_sol = self.solver(P, q, matrix(G_opt), matrix(h))
            print(opt_sol['x'])
            print(self.eval_obj(opt_sol['x']))


        else:
            N_switch_id = [self.l_twb[i].Nswitch for i in range(self.NP)]
            G = self.get_G(N_switch_id)
            logger.debug(
                    "\n Nswitch_id : %d and G matrix:\n %s", N_switch_id, G)
            h = np.zeros(G.shape[1])
            opt_sol = self.solver(P, q, matrix(G), matrix(h))
            mse = float(self.eval_obj(opt_sol['x']))
            logger.info("\n Nknots: %s and mse : %s", N_switch_id,mse)
        x = np.array(opt_sol['x'])
        print(x)
        E_model=np.ravel(self.M.dot(x))
        curvatures = x[0:self.cparams[0]]
        epsilon = x[-self.cols_sto:]
        print(epsilon)
        logger.info("\n The optimal solution is : \n %s", x)
        logger.info("\n The optimal curvatures are:\n%s\nepsilon:%s",
                    curvatures, epsilon)
        self.get_coeffs(list(x),E_model)
#
        sf.write_error(E_model, self.ref_E, mse)
        return E_model



    def get_M(self):
        """ Returns the M matrix 
        
        Returns:
            ndarray: The M matrix
        """
        v = self.l_twb[0].v
        logger.debug("\n The first v matrix is:\n %s", v)
        logger.debug("\n Shape of the first v matrix is:\t%s", v.shape)
        logger.debug("\n The stochiometry matrix is:\n%s", self.sto)
        if self.NP == 1:
            m = np.hstack((v, self.sto))
            logger.debug("\n The m  matrix is:\n %s \n shape:%s", m, m.shape)
            return m
        else:
            for i in range(1, self.NP):
                logger.debug("\n The %d pair v matrix is :\n %s",
                             i+1, self.l_twb[i].v)
                v = np.hstack((v, self.l_twb[i].v))
                logger.debug(
                    "\n The v  matrix shape after stacking :\t %s", v.shape)
            m = np.hstack((v, self.sto))
            return m

    def get_G(self, n_switch):
        """ returns constraints matrix
        
        Args:
            n_switch (int): switching point to cahnge signs of curvatures.
        
        Returns:
            ndarray: returns G matrix
        """
        g = block_diag(-1*np.identity(n_switch[0]),
                       np.identity(self.l_twb[0].cols-n_switch[0]))
        logger.debug("\n g matrix:\n%s", g)
        if self.NP == 1:
            G = block_diag(g, np.zeros(self.cols_sto))
            return G
        else:
            for elem in range(1, self.NP):
                tmp_G = block_diag(g, -1*np.identity(n_switch[elem]),
                       np.identity(self.l_twb[elem].cols-n_switch[elem]))
                g = tmp_G
        G = block_diag(g, np.zeros(self.cols_sto))
        return G
