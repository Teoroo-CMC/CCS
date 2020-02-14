
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
        self.cparams = self.l_twb[0].cols
        self.ns = len(ref_E)
        logger.info(" The reference energy : \n %s", self.ref_E)
#        M = self.get_M()
#        self.P = np.transpose(self.M).dot(self.M)

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
        ax1 = plt.subplot(4, 1, 1)
        plt.plot(E_model, self.ref_E, 'b*')
#        plt.text(max(E_model)-0.01,max(-b)-0.02,str(min_err),fontsize=8)
        plt.xlabel('E_model')
        plt.ylabel('E_train')
        z = np.polyfit(E_model, self.ref_E, 1)
        p = np.poly1d(z)
        plt.plot(E_model, p(E_model), 'r--')

        ax2 = plt.subplot(4, 1, 2)
        plt.scatter(s_interval[1:], s_a, c=[i < 0 for i in s_a])
        plt.xlabel('r')
        plt.ylabel('a coefficients')
#        plt.text(max(s_interval)-2,max(s_a)-5,str(min_eps),fontsize=8)

        ax3 = plt.subplot(4, 1, 3)
        c = [i < 0 for i in x]
        plt.scatter(s_interval[1:], x, c=c)
        plt.xlabel('r')
        plt.ylabel('c')

        ax4 = plt.subplot(4, 1, 4)
        n, bins, patches = plt.hist(x=np.ravel(
            self.l_twb[0].Dismat), bins=self.l_twb[0].interval, color='g', rwidth=0.85)
        plt.show()

    def solution(self):
        self.M = self.get_M()
        P = matrix(np.transpose(self.M).dot(self.M))
        q = -1*matrix(np.transpose(self.M).dot(self.ref_E))
        N_switch_id = 0
        obj = np.zeros(self.l_twb[0].cols)
        sol_list = []
        if not self.switch:
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
            opt_sol = sol_list[opt_sol_index[0]]

        else:
            N_switch_id = self.l_twb[0].Nknots+1
            G = self.get_G(N_switch_id)
            h = np.zeros(G.shape[1])
            opt_sol = self.solver(P, q, G, h)

        x = np.array(opt_sol['x'])
        model_eng = np.ravel(self.M.dot(x))
        curvatures = x[0:self.cparams]
        epsilon = x[-self.cols_sto:]
        logger.info("\n The optimal solution is : \n %s", x)
        logger.info("\n The optimal curvatures are:\n%s\nepsilon:%s",
                    curvatures, epsilon)

        s_a = np.dot(self.l_twb[0].A, curvatures)
        s_b = np.dot(self.l_twb[0].B, curvatures)
        s_c = np.dot(self.l_twb[0].C, curvatures)
        s_d = np.dot(self.l_twb[0].D, curvatures)

        sf.write_error(model_eng, self.ref_E, mse)
        splcoeffs = np.hstack((s_a, s_b, s_c, s_d))
   #     splderivs = sf.spline_eval012(s_a,s_b,s_c,s_d,self.l_twb[0].Rmin,self.l_twb[0].Rcut,self.l_twb[0].Rmin,self.l_twb[0].dx,self.l_twb[0].interval)
   #     s_a = np.insert(s_a,0,splderivs[0])
   #     splcoeffs = sf.get_spline_coeffs(self.l_twb[0].interval,s_a,splderivs[1],0)
   #     print (type(splcoeffs))
        sf.write_splinecoeffs(self.l_twb[0], splcoeffs)
        self.plot(model_eng, self.l_twb[0].interval, s_a, s_c)

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
                tmp_G = block_diag(g, np.identity(self.l_twb[elem].cols))
                g = tmp_G
        G = block_diag(g, self.cols_sto)
