from cvxopt import matrix, solvers
import numpy as np
import ccs.spline_functions as sf


class Objective():
    ''' Class for constructing the objective'''

    def __init__(self, l_twb, sto, ref_E, c='C', RT=None, RF=1e-6, switch=False, ST=None):
        self.l_twb = l_twb
        self.NP = len(l_twb)
        self.sto = sto
        self.q = -1*ref_E
        self.c = c
        self.RT = RT
        self.RF = RF
        self.switch = switch
        self.M = self.get_M()
        self.P = np.transpose(self.M).dot(self.M)

    @staticmethod
    def solver(P, q, G, h, MAXITER=300, tol=(1e-10, 1e-10, 1e-10)):

        solvers.options['maxiters'] = MAXITER
        solvers.options['feastol'] = tol[1]
        solvers.options['abstol'] = tol[2]
        solvers.options['reltol'] = tol[3]
        sol = solvers.qp(P, q, G, h)
        return sol

    def solution(self):
        N_switch_id = 0
        if not self.switch:
            for count, N_switch_id in enumerate(range(self.l_twb[0].Nknots+1)):
                G = self.get_G(N_switch_id)
                s = self.solver(self.M, self.q, self.h)

    def get_M(self):
        v = self.l_twb[0].v
#        print (v.shape)
#        print (self.sto.shape)
        if self.NP == 1:
            m = np.hstack((v, self.sto))
            return m
        else:
            for i in range(1, self.NP):
                v = np.hstack((v, l_twb[i].v))
            m = np.hstack(v, sto)
            return m


#    def get_G():
#    def get_q():
#    def solver():
