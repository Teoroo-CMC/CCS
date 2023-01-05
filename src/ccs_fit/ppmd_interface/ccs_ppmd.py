# ------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2023  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
# ------------------------------------------------------------------------------#

import numpy as np
from ctypes import *
from ppmd import *
import json
from ppmd.coulomb.fmm import internal_to_ev
import math
from scipy.optimize import minimize as min
from time import time as time
from ase.calculators.singlepoint import SinglePointCalculator


# CLASSES AND FUNCTION TO EXPLOIT THE FAST ELECTROSTATIC SOLVERS OF THE PPMD CODE


# some alias for readability and easy modification if we ever
# wish to use CUDA.

PairLoop = pairloop.CellByCellOMP
ParticleLoop = loop.ParticleLoopOMP
State = state.State
PositionDat = data.PositionDat
ParticleDat = data.ParticleDat
ScalarArray = data.ScalarArray
Kernel = kernel.Kernel
GlobalArray = data.GlobalArray
Constant = kernel.Constant
IntegratorRange = method.IntegratorRange


def steepest_descent(A, constants):

    sd_kernel_code = """
    P.i[0] += alpha*F.i[0];
    P.i[1] += alpha*F.i[1];
    P.i[2] += alpha*F.i[2];
    """
    sd_kernel = Kernel("SD", sd_kernel_code, constants)
    SD = ParticleLoop(
        kernel=sd_kernel, dat_dict={"F": A.force(access.READ), "P": A.pos(access.INC)}
    )

    return SD


def newtonic(A, constants):

    N_kernel_code = """
    double dx = 4.0*abs((P.i[0]- P_old.i[0])/(F_old.i[0]-F.i[0]));
    double dy = 4.0*abs((P.i[1]- P_old.i[1])/(F_old.i[1]-F.i[1]));
    double dz = 4.0*abs((P.i[2]- P_old.i[2])/(F_old.i[2]-F.i[2]));

    P_old.i[0]=P.i[0];
    P_old.i[1]=P.i[1];
    P_old.i[2]=P.i[2];

    P.i[0] += (dx < alpha) ? F.i[0]*dx: alpha*F.i[0];
    P.i[1] += (dy < alpha) ? F.i[1]*dy: alpha*F.i[1];
    P.i[2] += (dz < alpha) ? F.i[2]*dz: alpha*F.i[2];

    F_old.i[0]=F.i[0];
    F_old.i[1]=F.i[1];
    F_old.i[2]=F.i[2];
    """
    N_kernel = Kernel("NK", N_kernel_code, constants)
    NK = ParticleLoop(
        kernel=N_kernel,
        dat_dict={
            "F": A.force(access.READ),
            "P": A.pos(access.INC),
            "F_old": A.force_old(access.INC),
            "P_old": A.pos_old(access.INC),
        },
    )

    return NK


class FIRE:
    def __init__(self, A, CCS=None, dt_start=0.1, a_start=0.1, dt_max=1.0):
        self.CCS = CCS
        self.dt_start = dt_start
        self.a_start = a_start
        self.constants = (Constant("dr_max", 0.2),)
        self.nmin = 5
        self.f_inc = 1.1
        self.f_dec = 0.5
        self.f_a = 0.99
        self.dt_max = dt_max
        A.F_sq = GlobalArray(ncomp=1, dtype=c_double)
        A.V_sq = GlobalArray(ncomp=1, dtype=c_double)
        A.a = GlobalArray(ncomp=1, dtype=c_double)
        A.dt = GlobalArray(ncomp=1, dtype=c_double)
        A.dr_norm = GlobalArray(ncomp=1, dtype=c_double)
        A.Fv = GlobalArray(ncomp=1, dtype=c_double)
        A.dt.set(dt_start)
        A.a.set(a_start)

        self.F1 = self.FIRE_LOOP_1(A)
        self.F2 = self.FIRE_LOOP_2(A)
        self.F3 = self.FIRE_LOOP_3(A)
        self.F4 = self.FIRE_LOOP_4(A)
        self.F5 = self.FIRE_LOOP_5(A)

    def FIRE_LOOP_1(self, A):

        kernel_code = """
        P_old.i[0]=P.i[0];
        P_old.i[1]=P.i[1];
        P_old.i[2]=P.i[2];
        V_old.i[0]=V.i[0];
        V_old.i[1]=V.i[1];
        V_old.i[2]=V.i[2];
        F_old.i[0]=F.i[0];
        F_old.i[1]=F.i[1];
        F_old.i[2]=F.i[2];
        F_sq[0] += (F.i[0]*F.i[0] + F.i[1]*F.i[1] + F.i[2]*F.i[2]);
        """
        kernel = Kernel("k", kernel_code, self.constants)
        pl = ParticleLoop(
            kernel=kernel,
            dat_dict={
                "P": A.pos(access.READ),
                "P_old": A.pos_old(access.INC),
                "V": A.vel(access.READ),
                "V_old": A.vel_old(access.INC),
                "F": A.force(access.READ),
                "F_old": A.force_old(access.INC),
                "dt": A.dt(access.READ),
                "F_sq": A.F_sq(access.INC_ZERO),
            },
        )

        return pl

    def FIRE_LOOP_2(self, A):

        kernel_code = """
        const double norm = sqrt(F_sq[0]);
        F_norm.i[0] = F.i[0]/norm; 
        F_norm.i[1] = F.i[1]/norm; 
        F_norm.i[2] = F.i[2]/norm; 
        V_sq[0] += (V.i[0]*V.i[0] + V.i[1]*V.i[1] + V.i[2]*V.i[2]);
        Fv[0] += (F.i[0]*V.i[0] + F.i[1]*V.i[1] + F.i[2]*V.i[2])*14.39964547842567;
        """
        kernel = Kernel("k", kernel_code, self.constants)
        pl = ParticleLoop(
            kernel=kernel,
            dat_dict={
                "V": A.vel(access.INC),
                "F_norm": A.force_norm(access.INC_ZERO),
                "F": A.force(access.READ),
                "M": A.mass(access.READ),
                "F_sq": A.F_sq(access.READ),
                "dt": A.dt(access.READ),
                "Fv": A.Fv(access.INC_ZERO),
                "V_sq": A.V_sq(access.INC_ZERO),
            },
        )

        return pl

    def FIRE_LOOP_3(self, A):

        kernel_code = """
        const double norm = sqrt(V_sq[0]);
        V.i[0] = V.i[0]*(1.0-a[0]) + a[0]*F_norm.i[0]*norm;
        V.i[1] = V.i[1]*(1.0-a[0]) + a[0]*F_norm.i[1]*norm;
        V.i[2] = V.i[2]*(1.0-a[0]) + a[0]*F_norm.i[2]*norm;
        """
        kernel = Kernel("k", kernel_code, self.constants)
        pl = ParticleLoop(
            kernel=kernel,
            dat_dict={
                "V": A.vel(access.INC),
                "P": A.pos(access.INC),
                "F_norm": A.force_norm(access.READ),
                "F": A.force(access.READ),
                "dt": A.dt(access.READ),
                "M": A.mass(access.READ),
                "a": A.a(access.READ),
                "V_sq": A.V_sq(access.READ),
            },
        )

        return pl

    def FIRE_LOOP_4(self, A):

        kernel_code = """
        V.i[0] += dt[0]*F.i[0]*14.39964547842567;
        V.i[1] += dt[0]*F.i[1]*14.39964547842567;
        V.i[2] += dt[0]*F.i[2]*14.39964547842567;
        dr.i[0] = dt[0]*V.i[0];
        dr.i[1] = dt[0]*V.i[1];
        dr.i[2] = dt[0]*V.i[2];
        dr_norm[0] += (dr.i[0]*dr.i[0]+dr.i[1]*dr.i[1]+dr.i[2]*dr.i[2]);
        """
        kernel = Kernel("k", kernel_code, self.constants)
        pl = ParticleLoop(
            kernel=kernel,
            dat_dict={
                "V": A.vel(access.INC),
                "P": A.pos(access.INC),
                "dr": A.dr(access.INC),
                "dr_norm": A.dr_norm(access.INC_ZERO),
                "F_norm": A.force_norm(access.READ),
                "F": A.force(access.READ),
                "dt": A.dt(access.READ),
                "M": A.mass(access.READ),
                "a": A.a(access.READ),
                "V_sq": A.V_sq(access.READ),
            },
        )

        return pl

    def FIRE_LOOP_5(self, A):

        kernel_code = """
        const double norm = sqrt(dr_norm[0]);
        if(norm < dr_max){
        P.i[0] += dr.i[0];
        P.i[1] += dr.i[1];
        P.i[2] += dr.i[2];
        }
        if(norm >= dr_max){
        P.i[0] += dr_max*dr.i[0]/norm;
        P.i[1] += dr_max*dr.i[1]/norm;
        P.i[2] += dr_max*dr.i[2]/norm;
        }

        """
        kernel = Kernel("k", kernel_code, self.constants)
        pl = ParticleLoop(
            kernel=kernel,
            dat_dict={
                "V": A.vel(access.INC),
                "P": A.pos(access.INC),
                "dr": A.dr(access.READ),
                "F_norm": A.force_norm(access.READ),
                "F": A.force(access.READ),
                "dt": A.dt(access.READ),
                "M": A.mass(access.READ),
                "a": A.a(access.READ),
                "dr_norm": A.dr_norm(access.READ),
                "V_sq": A.V_sq(access.READ),
            },
        )

        return pl

    def opt(self, A, steps=10000, fmax=0.05):
        mem = 0
        Eold, _ = self.CCS.eval(A)
        t1 = time()
        for it in range(steps):
            uphill = False
            Enew, _ = self.CCS.eval(A=A)
            if Enew > Eold:
                A.pos[:] = A.pos_old[:]
                A.vel[:] = A.vel_old[:]
                A.force[:] = A.force_old[:]
                Enew = Eold
                uphill = True
            Eold = Enew
            self.F1.execute()
            self.F2.execute()
            if (A.Fv[0] > 0 and not uphill) or (it == 0):
                self.F3.execute()
                if mem > self.nmin:
                    A.dt.set(min(A.dt[0] * self.f_inc, self.dt_max))
                    A.a.set(A.a[0] * self.f_a)
                mem += 1
            else:
                mem = 0
                with A.vel.modify_view() as m:
                    m[:, :] = 0.0
                A.a.set(self.a_start)
                A.dt.set(A.dt[0] * self.f_dec)
            if not uphill:
                self.F4.execute()
                self.F5.execute()
            MaxF = (
                np.max(np.linalg.norm(A.force[0 : A.npart], axis=1)) * internal_to_ev()
            )
            if A.domain.comm.rank == 0 and it % 1 == 0:
                print(
                    "{: 8d} | E: {: 12.8f} F: {: 12.8f}  a: {: 12.8f} dt: {: 12.8g}  Fv: {: 12.8f} ".format(
                        it, Eold, MaxF, A.a[0], A.dt[0], A.Fv[0]
                    )
                )
            if MaxF < fmax:
                t2 = time()
                print(
                    "------------------------------- OPTIMIZATION COMPLETE -------------------------------"
                )
                print(
                    "    Total time for opimization: ",
                    t2 - t1,
                    " s, Max force:",
                    np.max(np.linalg.norm(A.force.view * internal_to_ev(), axis=1)),
                    "ev/Å",
                )
                break


class CCS:
    def __init__(self, A, atoms, q=None, CCS_params="CCS_params.json"):

        A.Etot = GlobalArray(ncomp=1, dtype=c_double)

        with open(CCS_params, "r") as f:
            CCS_params = json.load(f)

        id = {}
        charges = np.zeros((len(atoms), 1))
        at_ids = np.zeros((len(atoms), 1))
        keys = list(q.keys())
        Ntypes = len(keys)

        for a in keys:
            id[a] = keys.index(a)

        one_body_E = 0.0
        i = -1
        for a in atoms.get_chemical_symbols():
            i += 1
            charges[i] = q[a] * CCS_params["Charge scaling factor"]
            at_ids[i] = id[a]
            try:
                one_body_E += CCS_params["One_body"][a] / 14.39964547842567
            except:
                pass

        N = len(atoms)
        Ex = atoms.cell[0, 0]
        Ey = atoms.cell[1, 1]
        Ez = atoms.cell[2, 2]

        # make a state object and set the global number of particles N

        A.npart = N
        A.Ex = Ex
        A.Ey = Ey
        A.Ez = Ez

        # give the state a domain and boundary condition
        A.domain = domain.BaseDomainHalo(extent=(Ex, Ey, Ez))
        A.domain.boundary_condition = domain.BoundaryTypePeriodic()

        # add a PositionDat to contain positions
        A.pos = PositionDat(ncomp=3)
        A.force = ParticleDat(ncomp=3)
        A.charge = ParticleDat(ncomp=1)
        A.at = ParticleDat(ncomp=1)  # ,dtype= c_int)
        A.mass = ParticleDat(ncomp=1)
        A.vel = ParticleDat(ncomp=3)

        # add spline table
        SP_interval = ScalarArray(ncomp=Ntypes**2, dtype=c_int)
        SP_Rmin = ScalarArray(ncomp=Ntypes**2)
        SP_Rcut = ScalarArray(ncomp=Ntypes**2)
        SP_Ntypes = ScalarArray(ncomp=1, dtype=c_int)
        SP_Ntypes[:] = Ntypes

        table_lenght = 0
        cnt = -1
        for Z_i in id:
            for Z_j in id:
                cnt += 1
                try:
                    try:
                        tmp_Rcut = CCS_params["Two_body"][Z_i + "-" + Z_j]["r_cut"]
                        tmp_Rmin = CCS_params["Two_body"][Z_i + "-" + Z_j]["r_min"]
                        tmp_A = CCS_params["Two_body"][Z_i + "-" + Z_j]["spl_a"]
                    except:
                        tmp_Rcut = CCS_params["Two_body"][Z_j + "-" + Z_i]["r_cut"]
                        tmp_Rmin = CCS_params["Two_body"][Z_j + "-" + Z_i]["r_min"]
                        tmp_A = CCS_params["Two_body"][Z_j + "-" + Z_i]["spl_a"]
                except:
                    tmp_Rcut = 0.0
                    tmp_Rmin = 0.0
                    tmp_A = [0.0]

                k = len(tmp_A)
                table_lenght += k
                SP_Rmin[cnt] = tmp_Rmin
                SP_Rcut[cnt] = tmp_Rcut
                SP_interval[cnt] = table_lenght

        SP_A = ScalarArray(ncomp=table_lenght)
        SP_B = ScalarArray(ncomp=table_lenght)
        SP_C = ScalarArray(ncomp=table_lenght)
        SP_D = ScalarArray(ncomp=table_lenght)

        table_lenght = 0  # THIS TIME FILL IN THE ACTUAL TABLES...
        cnt = -1
        for Z_i in id:
            for Z_j in id:
                cnt += 1
                try:
                    try:
                        tmp_A = CCS_params["Two_body"][Z_i + "-" + Z_j]["spl_a"]
                        tmp_B = CCS_params["Two_body"][Z_i + "-" + Z_j]["spl_b"]
                        tmp_C = CCS_params["Two_body"][Z_i + "-" + Z_j]["spl_c"]
                        tmp_D = CCS_params["Two_body"][Z_i + "-" + Z_j]["spl_d"]
                    except:
                        tmp_A = CCS_params["Two_body"][Z_j + "-" + Z_i]["spl_a"]
                        tmp_B = CCS_params["Two_body"][Z_j + "-" + Z_i]["spl_b"]
                        tmp_C = CCS_params["Two_body"][Z_j + "-" + Z_i]["spl_c"]
                        tmp_D = CCS_params["Two_body"][Z_j + "-" + Z_i]["spl_d"]
                except:
                    tmp_A = [0.0]
                    tmp_B = [0.0]
                    tmp_C = [0.0]
                    tmp_D = [0.0]

                k = len(tmp_A)
                table_lenght += k
                SP_A[table_lenght - k : table_lenght] = tmp_A
                SP_B[table_lenght - k : table_lenght] = tmp_B
                SP_C[table_lenght - k : table_lenght] = tmp_C
                SP_D[table_lenght - k : table_lenght] = tmp_D

        # Fill in A (atoms) and put origin is in the centre

        pos = (atoms.get_scaled_positions() - [0.5, 0.5, 0.5]) * [Ex, Ey, Ez]
        A.pos[:] = pos
        A.force[:] = 0.0
        A.at[:] = at_ids
        A.charge[:] = charges
        A.mass[:] = atoms.get_masses()[:][0]
        A.pos_old = ParticleDat(ncomp=3)
        A.force_old = ParticleDat(ncomp=3)
        A.pos_old[:] = A.pos[:]
        A.force_old[:] = A.force[:]
        A.vel_old = ParticleDat(ncomp=3)
        A.dr = ParticleDat(ncomp=3)
        A.force_norm = ParticleDat(ncomp=3)

        # broadcast the data accross MPI ranks
        A.scatter_data_from(0)

        # the pairloop guarantees that all particles such that |r_i - r_j| < r_n
        # are looped over. It may also propose pairs of particles such that
        # |r_i - r_j| >= r_n and it is the users responsibility to mask off these
        # cases
        kernel_src = """
        // Vector displacement from particle i to particle j.
        const double R0 = P.j[0] - P.i[0];
        const double R1 = P.j[1] - P.i[1];
        const double R2 = P.j[2] - P.i[2];

        // Distance, dr and spline interval ...
        const int pi = (int)at.i[0]*(int)SP_Ntypes[0] + (int)at.j[0];
        const double r2 = R0*R0 + R1*R1 + R2*R2;
        const double r  = sqrt(r2);


        if(r < SP_Rcut[pi]){
        const int k_end = SP_interval[pi]-1;
        int k_start = 0;
        if(pi > 0){
            k_start = SP_interval[pi-1];
        } 

        const double delta = (SP_Rcut[pi]-SP_Rmin[pi] ) / (k_end-k_start);
        const int r_int = floor( (r-SP_Rmin[pi])/delta);
        const double dr= r - SP_Rmin[pi] - r_int*delta;
        const double E = SP_A[r_int+k_start]+dr*(SP_B[r_int+k_start] +dr*(SP_C[r_int+k_start] + dr*SP_D[r_int+k_start]));
        const double f= (1/14.39964547842567)*(SP_B[r_int+k_start] +2*dr*(SP_C[r_int+k_start] + 1.5*dr*SP_D[r_int+k_start]));
        U[0] += 0.5*E / 14.39964547842567; // Complying with ppmd internal units
        F.i[0] += f*R0/r ;
        F.i[1] += f*R1/r ;
        F.i[2] += f*R2/r ;
        }

        """

        ccs_kernel = Kernel("CCS", kernel_src)

        # create a pairloop

        r_c = max(SP_Rcut[:])  # Global cutoff radius
        r_n = r_c + 0.1 * r_c

        ccs_pairloop = PairLoop(
            ccs_kernel,
            {
                "P": A.pos(access.READ),
                "F": A.force(access.INC_ZERO),
                "U": A.Etot(access.INC_ZERO),
                "SP_A": SP_A(access.READ),
                "SP_B": SP_B(access.READ),
                "SP_C": SP_C(access.READ),
                "SP_D": SP_D(access.READ),
                "SP_interval": SP_interval(access.READ),
                "SP_Rmin": SP_Rmin(access.READ),
                "SP_Rcut": SP_Rcut(access.READ),
                "SP_Ntypes": SP_Ntypes(access.READ),
                "at": A.at(access.READ),
            },
            shell_cutoff=r_n,
        )
        l = 12

        if np.linalg.norm(charges) > 0.0001:
            try:
                es = coulomb.fmm.PyFMM(
                    A.domain, free_space=False, r=max(3, math.log(N, 8)), l=l
                )
            except:
                es = coulomb.ewald.EwaldOrthoganal(A.domain, real_cutoff=10.0)
            self.es = es
        else:
            self.es = None
            print("SWITHING OF +Q.")

        self.twobody = ccs_pairloop
        self.onebody = one_body_E

    def eval(self, A, pos=None):

        # if pos is None:
        #     pos=copy( A.pos.view)

        try:
            # print("POSITIONS SET SUCCESFULLY")
            pos = pos.reshape((A.npart, 3))
            pos[:, 0] = np.mod(pos[:, 0] + A.Ex, A.Ex) - 0.5 * A.Ex
            pos[:, 1] = np.mod(pos[:, 1] + A.Ey, A.Ey) - 0.5 * A.Ey
            pos[:, 2] = np.mod(pos[:, 2] + A.Ez, A.Ez) - 0.5 * A.Ez
            with A.pos.modify_view() as m:
                m[:, :] = pos
        except:
            # print("FIALED TO SET POSITIONS!")
            pass

        self.twobody.execute()
        E = A.Etot[0]
        if self.es:
            Ee = self.es(A.pos, A.charge, A.force)
            E += Ee
        return (
            E + self.onebody
        ) * internal_to_ev(), -A.force.view.flatten() * internal_to_ev()


def optimize(A, atoms, CCS, Type="LBGS", Fmax=0.05):
    def _fun(x, CCS, A):
        f, g = CCS.eval(A, pos=x)
        return f, g

    with A.pos.modify_view() as m:
        m[:, :] = atoms.positions

    if Type == "LBGS":
        options = {
            "disp": 1,
            "iprint": 101,
            "maxiter": 1000,
            "gtol": Fmax,
            "ftol": 1e-12,
        }
        method = "L-BFGS-B"

    if Type == "CG":
        options = {
            "disp": 1,
            "iprint": 101,
            "maxiter": 1000,
            "gtol": Fmax,
            "norm": np.inf,
        }
        method = "cg"

    t1 = time()
    min(
        _fun,
        A.pos[0 : A.npart].flatten(),
        args=(CCS, A),
        method=method,
        jac=True,
        options=options,
    )
    t2 = time()
    print(
        "------------------------------- OPTIMIZATION COMPLETE -------------------------------"
    )
    print(
        "    Total time for opimization: ",
        t2 - t1,
        " s, Max force:",
        np.max(np.linalg.norm(A.force.view * internal_to_ev(), axis=1)),
        "ev/Å",
    )
    atoms.positions = A.pos[0 : A.npart]
    calculator = SinglePointCalculator(
        atoms, energy=A.Etot[0], free_energy=A.Etot[0], forces=A.force[0 : A.npart]
    )

    atoms.calc = calculator
    atoms.get_potential_energy()
    atoms.get_forces()
    return atoms
