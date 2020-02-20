
import logging

import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from scipy.linalg import block_diag

import ccs.spline_functions as sf

logger = logging.getLogger(__name__)


class Objective():
    ''' Class for constructing the objective'''

    def __init__(self, l_twb, sto, ref_E, c='C', RT=None, RF=1e-6, switch=False, ST=None):
        self.l_twb = l_twb
        self.sto = sto
        self.ref_E = np.asarray(ref_E)
        self.c = c
        self.RT = RT
        self.RF = RF
        self.switch = switch
        self.cols_sto = sto.shape[1]
        self.NP = len(l_twb)
        self.cparams = [self.l_twb[i].cols for i in range(self.NP)]
        self.ns = len(ref_E)
        logger.debug(" The reference energy : \n %s", self.ref_E)

    @staticmethod
    def solver(P, q, G, h, MAXITER=300, tol=(1e-10, 1e-10, 1e-10)):

        solvers.options['maxiters'] = MAXITER
        solvers.options['feastol'] = tol[0]
        solvers.options['abstol'] = tol[1]
        solvers.options['reltol'] = tol[2]
        sol = solvers.qp(P, q, G, h)
        return sol

    def eval_obj(self, x):
        return np.format_float_scientific(np.sum((self.ref_E - (np.ravel(self.M.dot(x))))**2)/self.ns, precision=4)

    def plot(self, E_model, s_interval, s_a, x):

        fig = plt.figure()

        ax1 = fig.add_subplot(2, 2, 1)
        ax1.plot(E_model, self.ref_E, 'bo')
        ax1.set_xlabel('Predicted energies')
        ax1.set_ylabel('Ref. energies')
        z = np.polyfit(E_model, self.ref_E, 1)
        p = np.poly1d(z)
        ax1.plot(E_model, p(E_model), 'r--')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.scatter(s_interval[1:], s_a, c=[i < 0 for i in s_a])
        ax2.set_xlabel('Distance')
        ax2.set_ylabel('a coefficients')

        ax3 = fig.add_subplot(2, 2, 3)
        c = [i < 0 for i in x]
        ax3.scatter(s_interval[1:], x, c=c)
        ax3.set_ylabel('c coefficients')
        ax3.set_xlabel('Distance')

        ax4 = fig.add_subplot(2, 2, 4)
        n, bins, patches = plt.hist(x=np.ravel(
            self.l_twb[0].Dismat), bins=self.l_twb[0].interval, color='g', rwidth=0.85)
        ax4.set_ylabel('Frequency of a distance')
        ax4.set_xlabel('Spline interval')
        plt.tight_layout()
        plt.savefig('output.png')
#        plt.show()

    def solution(self):
        self.M = self.get_M()
        P = matrix(np.transpose(self.M).dot(self.M))
        q = -1*matrix(np.transpose(self.M).dot(self.ref_E))
        N_switch_id = 0
        obj = np.zeros(self.l_twb[0].cols)
        sol_list = []
        if self.l_twb[0].Nswitch == None:
            for count, N_switch_id in enumerate(range(self.l_twb[0].Nknots+1)):
                G = self.get_G(N_switch_id)
                logger.debug(
                    "\n Nswitch_id : %d and G matrix:\n %s", N_switch_id, G)
                h = np.zeros(G.shape[1])
                sol = self.solver(P, q, matrix(G), matrix(h))
                obj[count] = self.eval_obj(sol['x'])
                sol_list.append(sol)

            mse = np.min(obj)
            opt_sol_index = np.ravel(np.argwhere(obj == mse))
            logger.info("\n The best switch is : %d", opt_sol_index)
            opt_sol = sol_list[opt_sol_index[0]]

        else:
            N_switch_id = self.l_twb[0].Nswitch
            G = self.get_G(N_switch_id)
            logger.debug(
                    "\n Nswitch_id : %d and G matrix:\n %s", N_switch_id, G)
            h = np.zeros(G.shape[1])
            opt_sol = self.solver(P, q, matrix(G), matrix(h))
            mse = float(self.eval_obj(opt_sol['x']))
            logger.debug(
                    "\n mse: %s \n",mse)

        x = np.array(opt_sol['x'])
        E_model=np.ravel(self.M.dot(x))
        curvatures = x[0:self.cparams[0]]
        epsilon = x[-self.cols_sto:]
        logger.info("\n The optimal solution is : \n %s", x)
        logger.info("\n The optimal curvatures are:\n%s\nepsilon:%s",
                    curvatures, epsilon)
#
        s_a = np.dot(self.l_twb[0].A, curvatures)
        s_b = np.dot(self.l_twb[0].B, curvatures)
        s_c = np.dot(self.l_twb[0].C, curvatures)
        s_d = np.dot(self.l_twb[0].D, curvatures)
#
        sf.write_error(E_model, self.ref_E, mse)
        splcoeffs = np.hstack((s_a, s_b, s_c, s_d))
#   #     splderivs = sf.spline_eval012(s_a,s_b,s_c,s_d,self.l_twb[0].Rmin,self.l_twb[0].Rcut,self.l_twb[0].Rmin,self.l_twb[0].dx,self.l_twb[0].interval)
#   #     s_a = np.insert(s_a,0,splderivs[0])
#   #     splcoeffs = sf.get_spline_coeffs(self.l_twb[0].interval,s_a,splderivs[1],0)
#   #     print (type(splcoeffs))
        sf.write_splinecoeffs(self.l_twb[0], splcoeffs)
        self.plot(E_model, self.l_twb[0].interval, s_a, s_c)



    def get_M(self):
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
        g = block_diag(-1*np.identity(n_switch),
                       np.identity(self.l_twb[0].cols-n_switch))
        logger.debug("\n g matrix:\n%s", g)
        if self.NP == 1:
            G = block_diag(g, np.identity(self.cols_sto))
            return G
        else:
            for elem in range(1, self.NP):
                tmp_G = block_diag(g, -1*np.identity(self.l_twb[elem].cols))
                g = tmp_G
        G = block_diag(g, self.cols_sto)
        return G
