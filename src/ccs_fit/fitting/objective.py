# ------------------------------------------------------------------------------#
#  CCS: Curvature Constrained Splines                                          #
#  Copyright (C) 2019 - 2023  CCS developers group                             #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
# ------------------------------------------------------------------------------#


"""This module constructs and solves the spline objective."""


import logging
from tqdm import tqdm
import itertools
import json
import bisect
from collections import OrderedDict
import numpy as np
from cvxopt import matrix, solvers
from scipy.linalg import block_diag
from math import isnan

logger = logging.getLogger(__name__)


class Objective:
    """Objective function for the ccs method."""

    def __init__(
        self,
        l_twb,
        l_one,
        l_chg,
        sto,
        energy_ref,
        force_ref,
        stress_ref,
        gen_params,
        energy_ewald=[],
        force_ewald=[],
        stress_ewald=[],
    ):
        """Generates Objective class object.

        Args:

            l_twb (list): list of Twobody class objects
            l_one (list): list of Onebody class objects
            sto (ndarray): array containing number of atoms of each type
            energy_ref (ndarray): reference energies
            ge_params (dict) : optionself.sto, s
            ewald (list, optional) : ewald energy values for CCS+Q

        """

        self.l_twb = l_twb
        self.l_one = l_one
        self.l_chg = l_chg
        self.sto = sto
        self.sto_full = self.sto


        self.energy_ref = energy_ref
        self.force_ref_x = [x[0] for x in force_ref]
        self.force_ref_y = [y[1] for y in force_ref]
        self.force_ref_z = [z[2] for z in force_ref]
        self.force_ref = np.array(
            [*self.force_ref_x, *self.force_ref_y, *self.force_ref_z]
        )
        self.stress_ref_xx = [x[0][0] for x in stress_ref]
        self.stress_ref_xy = [x[0][1] for x in stress_ref]
        self.stress_ref_xz = [x[0][2] for x in stress_ref]
        self.stress_ref_yx = [y[1][0] for y in stress_ref]
        self.stress_ref_yy = [y[1][1] for y in stress_ref]
        self.stress_ref_yz = [y[1][2] for y in stress_ref]
        self.stress_ref_zx = [z[2][0] for z in stress_ref]
        self.stress_ref_zy = [z[2][1] for z in stress_ref]
        self.stress_ref_zz = [z[2][2] for z in stress_ref]
        self.stress_ref = np.array(
            [*self.stress_ref_xx, *self.stress_ref_yy, *self.stress_ref_zz,*self.stress_ref_xy,*self.stress_ref_yz,*self.stress_ref_xz]
        )
        self.ref = np.hstack((self.energy_ref, self.force_ref, self.stress_ref))


        try:
            self.ewald_energy = np.array(energy_ewald).reshape(-1, 1).flatten()
            self.force_ewald_x = [x[0] for x in force_ewald]
            self.force_ewald_y = [y[1] for y in force_ewald]
            self.force_ewald_z = [z[2] for z in force_ewald]
            self.force_ewald = np.array(
                [*self.force_ewald_x, *self.force_ewald_y, *self.force_ewald_z]
                    )
            self.stress_ewald_xx = [x[0][0] for x in stress_ewald]
            self.stress_ewald_xy = [x[0][1] for x in stress_ewald]
            self.stress_ewald_xz = [x[0][2] for x in stress_ewald]
            self.stress_ewald_yx = [y[1][0] for y in stress_ewald]
            self.stress_ewald_yy = [y[1][1] for y in stress_ewald]
            self.stress_ewald_yz = [y[1][2] for y in stress_ewald]
            self.stress_ewald_zx = [z[2][0] for z in stress_ewald]
            self.stress_ewald_zy = [z[2][1] for z in stress_ewald]
            self.stress_ewald_zz = [z[2][2] for z in stress_ewald]
            self.stress_ewald = np.array(
                [*self.stress_ewald_xx, *self.stress_ewald_yy, *self.stress_ewald_zz,*self.stress_ewald_xy,*self.stress_ewald_yz,*self.stress_ewald_xz]
            )
            self.ewald = np.hstack((self.ewald_energy, self.force_ewald,self.stress_ewald)).flatten()
        except:
            pass

        self.charge_scaling = 0.0

        for kk, vv in gen_params.items():
            setattr(self, kk, vv)

        self.cols_sto = self.sto.shape[1]
        self.np = len(l_twb)
        self.no = len(l_one)
        self.cparams = [self.l_twb[i].N for i in range(self.np)]
        self.ns = len(energy_ref)


    def reduce_stoichiometry(self):
        """Function to remove linear dependencies in stochiometry matrix."""
        reduce = True
        n_redundant = 0
        while reduce:
            check = 0
            for ci in range(np.shape(self.sto)[1]):
                if np.linalg.matrix_rank(self.sto[:, 0 : ci + 1]) < (ci + 1):
                    print("    There is linear dependence in stochiometry matrix.")
                    print(f"    Removing onebody term: {self.l_one[ci + n_redundant].name} ")
                    logger.info("    There is linear dependence in stochiometry matrix.")
                    logger.info(f"    Removing onebody term: {self.l_one[ci + n_redundant].name} ")
                    self.sto = np.delete(self.sto, ci, 1)
                    self.l_one[ci + n_redundant].epsilon_supported = False
                    check = 1
                    n_redundant += 1
                    break
            if check == 0:
                reduce = False

        assert self.sto.shape[1] == np.linalg.matrix_rank(
            self.sto
        ), "Linear dependence in stochiometry matrix"
        self.cols_sto = self.sto.shape[1]





    def predict(self, xx):
        """Predict results.

        Args:

            xx (ndarrray): Solution array from training.
            Needs to be updated to handle merging and dissolving intervals

        """
        self.sto = self.sto_full
        self.mm = self.get_m()

        try:
            self.model_energies = np.ravel(
                self.mm[0 : self.l_twb[0].Nconfs, :].dot(xx)
            )
            error = self.model_energies - self.energy_ref
        except:
            self.model_energies = []
            error = []

        self.write_error(fname="error_test.out")
        return self.model_energies, error

    @staticmethod
    def solver(pp, qq, gg, hh, aa, bb, maxiter=300, tol=(1e-10, 1e-10, 1e-10)):
        """The solver for the objective.

        Args:

            pp (matrix): P matrix as per standard Quadratic Programming(QP)
                notation.
            qq (matrix): q matrix as per standard QP notation.
            gg (matrix): G matrix as per standard QP notation.
            hh (matrix): h matrix as per standard QP notation
            aa (matrix): A matrix as per standard QP notation.
            bb (matrix): b matrix as per standard QP notation
            maxiter (int, optional): maximum iteration steps (default: 300).
            tol (tuple, optional): tolerance value of the solution
                (default: (1e-10, 1e-10, 1e-10)).

        Returns:

            sol (dict): dictionary containing solution details

        """

        solvers.options["show_progress"] = False
        solvers.options["maxiters"] = maxiter
        solvers.options["feastol"] = tol[0]
        solvers.options["abstol"] = tol[1]
        solvers.options["reltol"] = tol[2]

        if aa:
            sol = solvers.qp(pp, qq, gg, hh, aa, bb)
        else:
            sol = solvers.qp(pp, qq, gg, hh)

        return sol

    def merge_intervals(self):
        # Merge intervals
        for i in range(self.np):
            self.l_twb[i].merge_intervals()
            self.cparams = [self.l_twb[i].N for i in range(self.np)]

    def unfold_intervals(self):
        for ii in range(self.np):
            self.l_twb[ii].dissolve_interval()

    def eval_obj(self, xx):
        """Mean squared error function.

        Args:

            xx (ndarray): the solution for the objective

        Returns:

            float: mean square error

        """

        return np.format_float_scientific(
            np.sum((self.ref - (np.ravel(self.mm.dot(xx)))) ** 2) / self.ns,
            precision=4,
        )

    def assign_parameter_values(self, xx):
        # One-body terms
        counter = -1
        if self.Interface == "CCS+Q":
            counter = 0
            self.charge_scaling = xx[-1] ** 0.5
            print(f"    Charge scaling computed from sqrt {xx[-1]}.")
            if xx[-1] < 0:
                print(f"Imaginary value for charge scaling: sqrt( {xx[-1]} ).")
        for k in range(self.no):
            i = self.no - k - 1
            if self.l_one[i].epsilon_supported:
                counter += 1
                self.l_one[i].epsilon = float(xx[-1 - counter])
        # Two-body terms
        ind = 0
        for ii in range(self.np):
            self.l_twb[ii].curvatures = np.asarray(
                xx[ind : ind + self.cparams[ii]]
            )
            ind = ind + self.cparams[ii]

    def compute_model(self,xx):
        self.model_energies = np.ravel(
            self.mm[0 : self.l_twb[0].Nconfs, :].dot(xx)
        )

        if self.l_twb[0].Nconfs_forces > 0:
            self.model_forces = np.ravel(
                    self.mm[self.l_twb[0].Nconfs: self.l_twb[0].Nconfs + 3* self.l_twb[0].Nconfs_forces, :].dot(xx)
            )

        if self.l_twb[0].Nconfs_stresses > 0:
            self.model_stress = np.ravel(
                    self.mm[self.l_twb[0].Nconfs + 3* self.l_twb[0].Nconfs_forces :self.l_twb[0].Nconfs + 3* self.l_twb[0].Nconfs_forces+6*self.l_twb[0].Nconfs_stresses, :].dot(xx)
            )



    def list_iterator(self):
        """Iterates over the self.np attribute."""

        tmp = []
        for elem in range(self.np):
            if self.l_twb[elem].swtype.lower() == "rep":
                tmp.append([self.l_twb[elem].N])
            if self.l_twb[elem].swtype.lower() == "att":
                tmp.append([-1])
            if self.l_twb[elem].swtype.lower() == "sw":
                if self.l_twb[elem].search_mode.lower() == "full":
                    tmp2=[]
                    tmp2.append(-1)
                    tmp2.extend(self.l_twb[elem].indices)
                    tmp.append(tmp2)
                elif self.l_twb[elem].search_mode.lower() == "range":
                    range_center = self.l_twb[elem].range_center
                    range_width = self.l_twb[elem].range_width
                    Rmin = self.l_twb[elem].Rmin
                    Rcut = self.l_twb[elem].Rcut
                    res = self.l_twb[elem].res
                    range_min = max(
                        0,
                        bisect.bisect_left(
                            self.l_twb[elem].rn,
                            (range_center - range_width / 2),
                        ),
                    )
                    range_max = min(
                        self.l_twb[elem].N,
                        bisect.bisect_left(
                            self.l_twb[elem].rn,
                            (range_center + range_width / 2),
                        ),
                    )
                    tmp.append(self.l_twb[elem].indices[range_min:range_max])
                    print(
                        "    Range search turned on for element pair {}; {} possible switch indices in range of {:.2f}-{:.2f} Å.".format(
                            self.l_twb[elem].name,
                            len(self.l_twb[elem].indices[range_min:range_max]),
                            max(
                                Rmin,
                                int((range_center - range_width - Rmin) / res)
                                * res
                                + Rmin,
                            ),
                            min(
                                Rcut,
                                int((range_center + range_width - Rmin) / res)
                                * res
                                + Rmin,
                            ),
                        )
                    )
                elif self.l_twb[elem].search_mode.lower() == "point":
                    search_indices = [
                        bisect.bisect_left(self.l_twb[elem].rn, search_point)
                        for search_point in self.l_twb[elem].search_points
                    ]
                    search_indices = np.unique(search_indices)
                    search_indices=search_indices[search_indices<len(self.l_twb[elem].rn)].tolist()
                    print(
                        "    Switch points located at {} to for element pair {} based on point search.".format(
                            "["
                            + ", ".join(
                                [
                                    "{:.2f}".format(
                                        self.l_twb[elem].rn[search_index]
                                    )
                                    for search_index in search_indices
                                ]
                            )
                            + "] Å",
                            self.l_twb[elem].name,
                        )
                    )
                    tmp.append(
                        [
                            self.l_twb[elem].indices[search_index]
                            for search_index in search_indices
                        ]
                    )
                elif self.l_twb[elem].search_mode.lower() == "sparse":
                    search_indices = [
                        bisect.bisect_left(self.l_twb[elem].rn, search_point)
                        for search_point in np.arange(self.l_twb[elem].Rmin, self.l_twb[elem].Rcut, self.l_twb[elem].search_resolution)
                    ]
                    search_indices = np.unique(search_indices).tolist()
                    print(
                        "    Switch points located at {} to for element pair {} based on point search.".format(
                            "["
                            + ", ".join(
                                [
                                    "{:.2f}".format(
                                        self.l_twb[elem].rn[search_index]
                                    )
                                    for search_index in search_indices
                                ]
                            )
                            + "] Å",
                            self.l_twb[elem].name,
                        )
                    )
                    tmp.append(
                        [
                            self.l_twb[elem].indices[search_index]
                            for search_index in search_indices
                        ]
                    )
                else:
                    raise SyntaxError(
                        'Error: search mode not recognized! Please use one of the following recognized options; ["full", "range", "point"]'
                    )

        n_list = list(itertools.product(*tmp))

        return n_list

    def get_m(self):
        """Constructs the M matrix.

        Returns:

            ndarray: The M matrix.

        """

        # Add energy data
        tmp = []
        for ii in range(self.np):
            tmp.append(self.l_twb[ii].vv)
        vv = np.hstack([*tmp])
        mm = np.hstack((vv, self.sto))

        # Add force data
        tmp = []
        for ii in range(self.np):
            tmp.append(self.l_twb[ii].fvv_x)
        fvv_x = np.hstack([*tmp])
        fvv_x = np.hstack(
            (
                fvv_x,
                np.zeros((self.l_twb[ii].Nconfs_forces, np.shape(self.sto)[1])),
            )
        )
        mm = np.vstack((mm, fvv_x))

        tmp = []
        for ii in range(self.np):
            tmp.append(self.l_twb[ii].fvv_y)
        fvv_y = np.hstack([*tmp])
        fvv_y = np.hstack(
            (
                fvv_y,
                np.zeros((self.l_twb[ii].Nconfs_forces, np.shape(self.sto)[1])),
            )
        )
        mm = np.vstack((mm, fvv_y))

        tmp = []
        for ii in range(self.np):
            tmp.append(self.l_twb[ii].fvv_z)
        fvv_z = np.hstack([*tmp])
        fvv_z = np.hstack(
            (
                fvv_z,
                np.zeros((self.l_twb[ii].Nconfs_forces, np.shape(self.sto)[1])),
            )
        )
        mm = np.vstack((mm, fvv_z))

        # Add stress data
        tmp = []
        for ii in range(self.np):
            tmp.append(self.l_twb[ii].svv_xx)
        svv_xx = np.hstack([*tmp])
        svv_xx = np.hstack(
            (
                svv_xx,
                np.zeros((self.l_twb[ii].Nconfs_stresses, np.shape(self.sto)[1])),
            )
        )
        mm = np.vstack((mm, svv_xx))
        tmp = []
        for ii in range(self.np):
            tmp.append(self.l_twb[ii].svv_yy)
        svv_yy = np.hstack([*tmp])
        svv_yy = np.hstack(
            (
                svv_yy,
                np.zeros((self.l_twb[ii].Nconfs_stresses, np.shape(self.sto)[1])),
            )
        )
        mm = np.vstack((mm, svv_yy))
        tmp = []
        for ii in range(self.np):
            tmp.append(self.l_twb[ii].svv_zz)
        svv_zz = np.hstack([*tmp])
        svv_zz = np.hstack(
            (
                svv_zz,
                np.zeros((self.l_twb[ii].Nconfs_stresses, np.shape(self.sto)[1])),
            )
        )
        mm = np.vstack((mm, svv_zz))
        tmp = []
        for ii in range(self.np):
            tmp.append(self.l_twb[ii].svv_xy)
        svv_xy = np.hstack([*tmp])
        svv_xy = np.hstack(
            (
                svv_xy,
                np.zeros((self.l_twb[ii].Nconfs_stresses, np.shape(self.sto)[1])),
            )
        )
        mm = np.vstack((mm, svv_xy))
        tmp = []
        for ii in range(self.np):
            tmp.append(self.l_twb[ii].svv_yz)
        svv_yz = np.hstack([*tmp])
        svv_yz = np.hstack(
            (
                svv_yz,
                np.zeros((self.l_twb[ii].Nconfs_stresses, np.shape(self.sto)[1])),
            )
        )
        mm = np.vstack((mm, svv_yz))
        tmp = []
        for ii in range(self.np):
            tmp.append(self.l_twb[ii].svv_xz)
        svv_xz = np.hstack([*tmp])
        svv_xz = np.hstack(
            (
                svv_xz,
                np.zeros((self.l_twb[ii].Nconfs_stresses, np.shape(self.sto)[1])),
            )
        )
        mm = np.vstack((mm, svv_xz))

        if self.Interface == "CCS+Q":
            # THIS IS A BIT AKWARD CAN IT BE FIXED?
            mm = np.hstack((mm, np.atleast_2d(self.ewald).T))

        return mm

    def get_g(self, n_switch):
        """Returns constraints matrix.

        Args:

            n_switch (int): switching point to change signs of curvatures.

        Returns:

            ndarray: returns G and A matrices

        """

        aa = np.zeros(0)
        tmp = []
        for elem in range(self.np):
            tmp.append(self.l_twb[elem].switch_const(n_switch[elem]))
        gg = block_diag(*tmp)

        gg = block_diag(gg, np.zeros_like(np.eye(self.cols_sto)))
        if self.Interface == "CCS+Q":
            gg = block_diag(gg, -1)

        return gg, aa

    def write_error(self, fname="CCS_error_energies.out"):
        """Write the errors in energies to file.

        Args:

            fname (str, optional): Output filename (default: 'CCS_error_energies.out').

        """
        header = "{:<15}{:<15}{:<15}{:<15}".format(
            "Reference", "Predicted", "Error", "#atoms"
        )
        error = abs(self.energy_ref - self.model_energies)
        maxerror = max(abs(error))
        mse = ((error) ** 2).mean()
        Natoms = self.l_one[0].stomat
        for i in range(1, self.no):
            Natoms = Natoms + self.l_one[i].stomat
        footer = "MSE = {:2.5E}\nMaxerror = {:2.5E}".format(mse, maxerror)
        np.savetxt(
            fname,
            np.transpose([self.energy_ref, self.model_energies, error, Natoms]),
            header=header,
            footer=footer,
            fmt="%-15.5f",
        )


    def write_error_forces(
        self, mdl_for, ref_for, fname="CCS_error_forces.out"
    ):
        """Write the errors in forces to file.

        Args:

            mdl_for (ndarray): Force prediction values from splines.
            ref_for (ndarray): Reference force values.
            fname (str, optional): Output filename (default: 'CCS_error_forces.out').

        """
        header = "{:<15}{:<15}{:<15}".format("Reference", "Predicted", "Error")
        error = abs(ref_for - mdl_for)
        maxerror = max(abs(error))
        mse = ((error) ** 2).mean()

        footer = "MSE = {:2.5E}\nMaxerror = {:2.5E}".format(mse, maxerror)
        np.savetxt(
            fname,
            np.transpose([ref_for, mdl_for, error]),
            header=header,
            footer=footer,
            fmt="%-15.5f",
        )

    def write_error_stresses(
        self, mdl_str, ref_str, fname="CCS_error_stresses.out"
    ):
        """Write the errors in stresses to file.

        Args:

            mdl_str (ndarray): Stress prediction values from splines.
            ref_str (ndarray): Reference stress values.
            fname (str, optional): Output filename (default: 'CCS_error_stresses.out').

        """
        header = "{:<15}{:<15}{:<15}".format("Reference", "Predicted", "Error")
        error = abs(ref_str - mdl_str)
        maxerror = max(abs(error))
        mse = ((error) ** 2).mean()

        footer = "MSE = {:2.5E}\nMaxerror = {:2.5E}".format(mse, maxerror)
        np.savetxt(
            fname,
            np.transpose([ref_str, mdl_str, error]),
            header=header,
            footer=footer,
            fmt="%-15.5f",
        )

    def write_CCS_params(self, fname="CCS_params.json"):
        """Write the CCS parameters to file.

        Args:

            fname (str, optional): Output filename (default: 'CCS_params.json').

        """
        CCS_params = OrderedDict()
        #CCS_params["Charge scaling factor"] = float(self.charge_scaling)
        if self.charge_scaling != 0.0:
            scaled_chg = {key: value * float( self.charge_scaling) for key, value in self.l_chg.items()}
            CCS_params["Charges"] =  scaled_chg

        eps_params = OrderedDict()
        for k in range(self.no):
            if self.l_one[k].epsilon_supported:
                eps_params[self.l_one[k].name] = self.l_one[k].epsilon
            else:
                eps_params[self.l_one[k].name]=0.0

        CCS_params["One_body"] = eps_params

        two_bodies_dict = OrderedDict()
        for k in range(self.np):
            two_body_dict = OrderedDict()
            two_body_dict["r_min"] = self.l_twb[k].rn[0]
            two_body_dict["r_cut"] = self.l_twb[k].Rcut
            two_body_dict["dr"] = self.l_twb[k].res
            r_values = list(np.array(self.l_twb[k].rn))
            two_body_dict["r"] = list(r_values)
            two_body_dict["exp_a"] = self.l_twb[k].expcoeffs[0]
            if not isnan(self.l_twb[k].expcoeffs[1]):
                two_body_dict["exp_b"] = self.l_twb[k].expcoeffs[1]
            else:
                # print("WARNING: THE EXPONENTIAL FOR PAIR {} IS POORLY RESOLVED,
                # PROCEED WITH CAUTION!".format(self.l_twb[k].name))
                two_body_dict["exp_b"] = 0
            if not isnan(self.l_twb[k].expcoeffs[2]):
                two_body_dict["exp_c"] = self.l_twb[k].expcoeffs[2]
            else:
                two_body_dict["exp_c"] = 0
            # if two_body_dict["exp_a"]<0:
            # print("STRONG WARNING: THE EXPONENTIAL WALL IS ACTUALLY ATTRACTIVE!!!")
            a_values = list(self.l_twb[k].splcoeffs[:, 0])
            a_values.append(0)
            two_body_dict["spl_a"] = a_values
            b_values = list(self.l_twb[k].splcoeffs[:, 1])
            # if b_values[0]>0:
            # print("WARNING: THE PAIR {} IS ONLY SAMPLED IN A REGION WHERE IT IS STILL
            # REPULSIVE. THIS INDICATES THAT THE INTERACTION IS MOST LIKELY POORLY
            # RESOLVED, PROCEED WITH CAUTION!".format(self.l_twb[k].name))
            b_values.append(0)
            two_body_dict["spl_b"] = b_values
            c_values = list(self.l_twb[k].splcoeffs[:, 2])
            c_values.append(0)
            two_body_dict["spl_c"] = c_values
            d_values = list(self.l_twb[k].splcoeffs[:, 3])
            d_values.append(0)
            two_body_dict["spl_d"] = d_values
            two_bodies_dict[self.l_twb[k].name] = two_body_dict

        CCS_params["Two_body"] = two_bodies_dict
        with open(fname, "w") as f:
            json.dump(CCS_params, f, indent=8)

    def unconstrained_fit(self):
        # Solving unconstrained problem
        xx = np.linalg.lstsq(self.mm, self.ref, rcond=None)
        xx = xx[0]
        print(
            "    MSE of unconstrained problem is: ",
            ((self.mm.dot(xx) - self.ref) ** 2).mean(),
        )
        logger.info(f"    MSE of unconstrained problem is: {((self.mm.dot(xx) - self.ref) ** 2).mean()}")
        xx = xx.reshape(len(xx), 1)
        self.assign_parameter_values(xx)
        self.compute_model(xx)
        self.write_error(fname="UNC_error_energies.out")
        if self.l_twb[0].Nconfs_forces > 0:
            self.write_error_forces(self.model_forces, self.force_ref,fname="UNC_error_forces.out")
        if self.l_twb[0].Nconfs_stresses > 0:
            self.write_error_stresses(self.model_stress, self.stress_ref,fname="UNC_error_stresses.out")

        try:
            if self.Merging == "True":
                self.unfold_intervals()
        except:
            pass

        for ii in range(self.np):
            self.l_twb[ii].get_spline_coeffs()
            self.l_twb[ii].get_expcoeffs()

        self.write_CCS_params(fname="UNC_params.json")

        try:
            if self.Merging == "True":
                self.merge_intervals()
        except:
            pass

    def ridge_regresssion(self):
        # Solving ridge regression problem

        from sklearn import linear_model

        for lmb in self.RidgeLambda:
            ridge = linear_model.Ridge(alpha=lmb, fit_intercept=False)
            ridge.fit(self.mm, self.ref)
            ridge_pred = ridge.predict(self.mm)
            print(
                "    MSE from ridge regression: ",
                ((ridge_pred - self.ref) ** 2).mean(),
                "Regularization (alpha): ",
                lmb,
            )
            xx = ridge.coef_
            self.assign_parameter_values(xx)
            self.compute_model(xx)
            self.write_error(fname=f"RIDGE_error_energies_{lmb}.out")
            if self.l_twb[0].Nconfs_forces > 0:
                self.write_error_forces(self.model_forces, self.force_ref,fname=f"RIDGE_error_forces_{lmb}.out")
            if self.l_twb[0].Nconfs_stresses > 0:
                self.write_error_stresses(self.model_stress, self.stress_ref,fname=f"RIDGE_error_stresses_{lmb}.out")

            try:
                if self.Merging == "True":
                    self.unfold_intervals()
            except:
                pass

            for ii in range(self.np):
                self.l_twb[ii].get_spline_coeffs()
                self.l_twb[ii].get_expcoeffs()

            self.write_CCS_params(fname=f"RIDGE_params_{lmb}.json")

            try:
                if self.Merging == "True":
                    self.merge_intervals()
            except:
                pass

    def solution(self):
        """Function to solve the objective with constraints."""

        # Merging intervals
        try:
            if self.Merging == "True":
                self.merge_intervals()
        except:
            pass

        # Reduce stoichiometry
        self.reduce_stoichiometry()
        self.mm = self.get_m()

        # Define P and Q matrices
        pp = matrix(np.transpose(self.mm).dot(self.mm))
        eigvals = np.linalg.eigvals(pp)
        qq = -1 * matrix(np.transpose(self.mm).dot(self.ref))
        nswitch_list = self.list_iterator()
        obj = []

        logger.info("positive definite:%s", np.all((eigvals > 0)))
        logger.info("Condition number (logarithm):%f", np.log( np.linalg.cond(pp)))

        # Perform regular linear regression
        if self.DoUnconstrainedFit == "True":
            self.unconstrained_fit()

        # Perform ridge linear regression
        if self.DoRidgeRegression == "True":
            self.ridge_regresssion()

        # Search for optimum switch points
        for n_switch_id in tqdm(
            nswitch_list, desc="    Finding optimum switch", colour="#800080"
        ):
            [gg, aa] = self.get_g(n_switch_id) # Constraint matrices
            hh = np.zeros(gg.shape[0])         # Constraint matrices 
            bb = np.zeros(aa.shape[0])         # Constraint matrices
            sol = self.solver(
                pp, qq, matrix(gg), matrix(hh), matrix(aa), matrix(bb)
            )
            obj.append(float(self.eval_obj(sol["x"])))

        obj = np.asarray(obj) # List of objective values (MSE)
        mse = np.min(obj)
        opt_sol_index = int(np.ravel(np.argwhere(obj == mse)[0]))

        best_switch_r = np.around(
            [
                nswitch_list[opt_sol_index][elem] * self.l_twb[elem].res
                + self.l_twb[elem].Rmin
                for elem in range(self.np)
            ],
            decimals=2,
        )
        elem_pairs = [self.l_twb[elem].name for elem in range(self.np)]

        print(
            f"    The best switch is {nswitch_list[opt_sol_index][:]} with rmse: {mse**0.5}, corresponding to distances of {best_switch_r} Å for element pairs {elem_pairs[:]}."
        )
        logger.info(
            f"    The best switch is {nswitch_list[opt_sol_index][:]} with rmse: {mse**0.5}, corresponding to distances of {best_switch_r} Å for element pairs {elem_pairs[:]}."
        )

        # Repeat fit using optimum switches (repeating fit rather than saving all results saves memory)
        [g_opt, aa] = self.get_g(nswitch_list[opt_sol_index])
        bb = np.zeros(aa.shape[0])

        opt_sol = self.solver(
            pp, qq, matrix(g_opt), matrix(hh), matrix(aa), matrix(bb)
        )

        xx = np.array(opt_sol["x"])
        self.assign_parameter_values(xx)
        self.compute_model(xx)
        self.write_error()
        if self.l_twb[0].Nconfs_forces > 0:
            self.write_error_forces(self.model_forces, self.force_ref)
        if self.l_twb[0].Nconfs_stresses > 0:
            self.write_error_stresses(self.model_stress, self.stress_ref)

        # Unfold the spline to an equidistant grid
        try:
            if self.Merging == "True":
                self.unfold_intervals()
        except:
            pass

        x_unfolded = []
        for ii in range(self.np):
            self.l_twb[ii].get_spline_coeffs()
            self.l_twb[ii].get_expcoeffs()
            x_unfolded = np.hstack(
                (x_unfolded, np.array(self.l_twb[ii].curvatures).flatten())
            )
        for onb in self.l_one:
            if onb.epsilon_supported:
                x_unfolded = np.hstack((x_unfolded, np.array(onb.epsilon)))
            else:
                x_unfolded = np.hstack((x_unfolded, 0.0))
        xx = x_unfolded
 
        # Write parameters to file
        self.write_CCS_params()

        return self.model_energies, mse, xx
