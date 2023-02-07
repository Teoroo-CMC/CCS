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


logger = logging.getLogger(__name__)


class Objective:
    """Objective function for the ccs method."""

    def __init__(
        self,
        l_twb,
        l_one,
        sto,
        energy_ref,
        force_ref,
        gen_params,
        energy_ewald=[],
        force_ewald=[],
    ):
        """Generates Objective class object.

        Args:

            l_twb (list): list of Twobody class objects
            l_one (list): list of Onebody class objects
            sto (ndarray): array containing number of atoms of each type
            energy_ref (ndarray): reference energies
            ge_params (dict) : options
            ewald (list, optional) : ewald energy values for CCS+Q

        """

        self.l_twb = l_twb
        self.l_one = l_one
        self.sto = sto
        self.sto_full = self.sto

        self.energy_ref = energy_ref
        self.force_ref_x = [x[0] for x in force_ref]
        self.force_ref_y = [y[1] for y in force_ref]
        self.force_ref_z = [z[2] for z in force_ref]
        self.force_ref = np.array(
            [*self.force_ref_x, *self.force_ref_y, *self.force_ref_z]
        )
        self.ref = np.hstack((self.energy_ref, self.force_ref))

        # WHY DO I NEED TO FLATTEN?
        self.ewald_energy = np.array(energy_ewald).reshape(-1, 1).flatten()
        self.force_ewald_x = [x[0] for x in force_ewald]
        self.force_ewald_y = [y[1] for y in force_ewald]
        self.force_ewald_z = [z[2] for z in force_ewald]
        self.force_ewald = np.array(
            [*self.force_ewald_x, *self.force_ewald_y, *self.force_ewald_z]
        )

        # WHY DO I NEED TO FLATTEN?
        self.ewald = np.hstack((self.ewald_energy, self.force_ewald)).flatten()

        self.charge_scaling = 0.0

        for kk, vv in gen_params.items():
            setattr(self, kk, vv)

        self.cols_sto = self.sto.shape[1]
        self.np = len(l_twb)
        self.no = len(l_one)
        self.cparams = [self.l_twb[i].N for i in range(self.np)]
        self.ns = len(energy_ref)

        logger.debug(
            "The reference energy : \n %s \n Number of pairs:%s",
            self.energy_ref,
            self.np,
        )

    def reduce_stoichiometry(self):
        reduce = True
        n_redundant = 0
        while reduce:
            check = 0
            for ci in range(np.shape(self.sto)[1]):
                if np.linalg.matrix_rank(self.sto[:, 0: ci + 1]) < (ci + 1):
                    print("    There is linear dependence in stochiometry matrix!")
                    print(
                        "    Removing onebody term: "
                        + self.l_one[ci + n_redundant].name
                    )
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

    def solution(self):
        """Function to solve the objective with constraints."""

        # COMMENTS MERGING THE INTERVALS
        try:
            if self.merging == "True":
                self.merge_intervals()
        except:
            pass

        # Reduce stoichiometry
        self.reduce_stoichiometry()

        self.mm = self.get_m()
        logger.debug("\n Shape of M matrix is : %s", self.mm.shape)

        pp = matrix(np.transpose(self.mm).dot(self.mm))
        eigvals = np.linalg.eigvals(pp)
        qq = -1 * matrix(np.transpose(self.mm).dot(self.ref))
        nswitch_list = self.list_iterator()
        obj = []
        sol_list = []

        logger.info("positive definite:%s", np.all((eigvals > 0)))
        logger.info("Condition number:%f", np.linalg.cond(pp))


        #Evaluting the fittnes
        mm_trimmed=self.mm
        mm_trimmed=np.delete(mm_trimmed,0,1)
        pp_trimmed=matrix(np.transpose(mm_trimmed).dot(mm_trimmed))
        eigvals_trimmed = np.linalg.eigvals(pp_trimmed)   
        # print(f"    Condition number is: {np.linalg.cond(pp_trimmed)} ( {len(eigvals_trimmed)} {np.abs(max(eigvals_trimmed))} {np.abs(min(eigvals_trimmed))})")

        if self.do_unconstrained_fit == "True":
            ##### START UNC ####
            #Solving unconstrained problem
            xx=np.linalg.lstsq(self.mm,self.ref,rcond=None)
            xx=xx[0]
            print("    MSE of unconstrained problem is: ", ((self.mm.dot(xx)   - self.ref)**2).mean()    )
            xx=xx.reshape(len(xx),1)
            self.assign_parameter_values(xx)

            self.model_energies = np.ravel(
                self.mm[0: self.l_twb[0].Nconfs, :].dot(xx))
            self.write_error(fname="UNC_error.out")

            if self.l_twb[0].Nconfs_forces > 0:
                model_forces = np.ravel(
                    self.mm[-3*self.l_twb[0].Nconfs_forces:, :].dot(xx)
                )
                self.write_error_forces(model_forces, self.force_ref,fname="UNC_error_forces.out")

            try:
                if self.merging == "True":
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

            self.write_CCS_params(fname="UNC_params.json")

            try:
                if self.merging == "True":
                    self.merge_intervals()
            except:
                pass

            ##### END UNC ####            

        for n_switch_id in tqdm(
            nswitch_list, desc="    Finding optimum switch", colour="#800080"
        ):
            [gg, aa] = self.get_g(n_switch_id)
            hh = np.zeros(gg.shape[0])
            bb = np.zeros(aa.shape[0])
            sol = self.solver(pp, qq, matrix(gg), matrix(hh),
                              matrix(aa), matrix(bb))
            obj.append(float(self.eval_obj(sol["x"])))

        obj = np.asarray(obj)
        mse = np.min(obj)
        opt_sol_index = int(np.ravel(np.argwhere(obj == mse)[0]))

        # logger.info(
        #     "\n The best switch is : %s with mse: %s", *
        #     nswitch_list[opt_sol_index], mse
        # )
        best_switch_r = np.around([nswitch_list[opt_sol_index][elem]*self.l_twb[elem].res+self.l_twb[elem].Rmin for elem in range(self.np)], decimals=2)
        elem_pairs = [self.l_twb[elem].name for elem in range(self.np)]
        print(
            f"    The best switch is {nswitch_list[opt_sol_index][:]} with mse: {mse}, corresponding to distances of {best_switch_r} Å for element pairs {elem_pairs[:]}.")

        # [{' '.join(['{:2f}'.format(best_switch_r[elem]) for elem in range(self.np)])}]

        [g_opt, aa] = self.get_g(nswitch_list[opt_sol_index])
        bb = np.zeros(aa.shape[0])

        opt_sol = self.solver(pp, qq, matrix(
            g_opt), matrix(hh), matrix(aa), matrix(bb))

        xx = np.array(opt_sol["x"])
        self.assign_parameter_values(xx)

        self.model_energies = np.ravel(
            self.mm[0: self.l_twb[0].Nconfs, :].dot(xx))

        if self.l_twb[0].Nconfs_forces > 0:
            model_forces = np.ravel(
                self.mm[-3*self.l_twb[0].Nconfs_forces:, :].dot(xx)
            )
            self.write_error_forces(model_forces, self.force_ref)

        self.write_error()

        # COMMENT: Unfold the spline to an equidistant grid
        try:
            if self.merging == "True":
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

        self.write_CCS_params()

        return self.model_energies, mse, xx

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
                self.mm[0: self.l_twb[0].Nconfs, :].dot(xx))
            error = self.model_energies - self.energy_ref
            mse = ((error) ** 2).mean()
        except:
            self.model_energies = []
            error = []
            mse = 0

        self.write_error(fname="error_test.out")
        return self.model_energies, error

    @ staticmethod
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
            np.sum((self.ref - (np.ravel(self.mm.dot(xx)))) ** 2) / self.ns, precision=4
        )

    def assign_parameter_values(self, xx):
        # Onebodies
        counter = -1
        if self.interface == "CCS+Q":
            counter = 0
            self.charge_scaling = xx[-1] ** 0.5
        for k in range(self.no):
            i = self.no - k - 1
            if self.l_one[i].epsilon_supported:
                counter += 1
                self.l_one[i].epsilon = float(xx[-1 - counter])
        # Two-bodies
        ind = 0
        for ii in range(self.np):
            self.l_twb[ii].curvatures = np.asarray(
                xx[ind: ind + self.cparams[ii]])
            ind = ind + self.cparams[ii]
            # Unfold the spline to an equdistant grid
            # self.l_twb[ii].dissolve_interval()
            # self.l_twb[ii].get_spline_coeffs()
            # self.l_twb[ii].get_expcoeffs()

    def list_iterator(self):
        """Iterates over the self.np attribute."""

        tmp = []
        for elem in range(self.np):
            if self.l_twb[elem].Swtype == "rep":
                tmp.append([self.l_twb[elem].N])
            if self.l_twb[elem].Swtype == "att":
                tmp.append([0])
            if self.l_twb[elem].Swtype == "sw":
                if self.l_twb[elem].search_mode.lower() == "full":
                    tmp.append(self.l_twb[elem].indices)
                elif self.l_twb[elem].search_mode.lower() == "range":
                    range_center = self.l_twb[elem].range_center
                    range_width = self.l_twb[elem].range_width
                    Rmin = self.l_twb[elem].Rmin
                    Rcut = self.l_twb[elem].Rcut
                    res = self.l_twb[elem].res
                    range_min = max(0, bisect.bisect_left(self.l_twb[elem].rn, (range_center - range_width/2)))
                    range_max = min(self.l_twb[elem].N, bisect.bisect_left(self.l_twb[elem].rn, (range_center + range_width/2)))
                    tmp.append(self.l_twb[elem].indices[range_min:range_max])
                    print("    Range search turned on for element pair {}; {} possible switch indices in range of {:.2f}-{:.2f} Å.".format(self.l_twb[elem].name, len(self.l_twb[elem].indices[range_min:range_max]), max(Rmin, int((range_center - range_width - Rmin)/res)*res + Rmin), min(Rcut, int((range_center + range_width - Rmin)/res)*res + Rmin)))
                elif self.l_twb[elem].search_mode.lower() == "point":
                    search_indices = [bisect.bisect_left(self.l_twb[elem].rn, search_point) for search_point in self.l_twb[elem].search_points]
                    search_indices = np.unique(search_indices).tolist()
                    print("    Switch points located at {} to for element pair {} based on point search.".format('[' + ', '.join(["{:.2f}".format(self.l_twb[elem].rn[search_index]) for search_index in search_indices]) + '] Å', self.l_twb[elem].name)) 
                    tmp.append([self.l_twb[elem].indices[search_index] for search_index in search_indices])
                else:
                    raise SyntaxError("Error: search mode not recognized! Please use one of the following recognized options; [\"full\", \"range\", \"point\"]")

        n_list = list(itertools.product(*tmp))

        return n_list

    def get_m(self):
        """Returns the M matrix.

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
            (fvv_x, np.zeros(
                (self.l_twb[ii].Nconfs_forces, np.shape(self.sto)[1])))
        )
        mm = np.vstack((mm, fvv_x))

        tmp = []
        for ii in range(self.np):
            tmp.append(self.l_twb[ii].fvv_y)
        fvv_y = np.hstack([*tmp])
        fvv_y = np.hstack(
            (fvv_y, np.zeros(
                (self.l_twb[ii].Nconfs_forces, np.shape(self.sto)[1])))
        )
        mm = np.vstack((mm, fvv_y))

        tmp = []
        for ii in range(self.np):
            tmp.append(self.l_twb[ii].fvv_z)
        fvv_z = np.hstack([*tmp])
        fvv_z = np.hstack(
            (fvv_z, np.zeros(
                (self.l_twb[ii].Nconfs_forces, np.shape(self.sto)[1])))
        )
        mm = np.vstack((mm, fvv_z))

        if self.interface == "CCS+Q":
            # THIS IS A BIT AKWARD CAN IT BE FIXED?
            mm = np.hstack((mm, np.atleast_2d(self.ewald).T))

        return mm

    def get_g(self, n_switch):
        """Returns constraints matrix.

        Args:

            n_switch (int): switching point to change signs of curvatures.

        Returns:

            ndarray: returns G and A matrix

        """

        aa = np.zeros(0)
        tmp = []
        for elem in range(self.np):
            tmp.append(self.l_twb[elem].switch_const(n_switch[elem]))
        gg = block_diag(*tmp)

        gg = block_diag(gg, np.zeros_like(np.eye(self.cols_sto)))
        if self.interface == "CCS+Q":
            gg = block_diag(gg, -1)

        return gg, aa

    def write_error(self, fname="CCS_error.out"):
        """Prints the errors in a file.

        Args:

            mdl_eng (ndarray): Energy prediction values from splines.
            ref_eng (ndarray): Reference energy values.
            mse (float): Mean square error.
            fname (str, optional): Output filename (default: 'error.out').

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
            np.transpose(
                [self.energy_ref, self.model_energies, error, Natoms]),
            header=header,
            footer=footer,
            fmt="%-15.5f",
        )

        print("    Final root mean square error in energy: ", (np.square(error/Natoms)).mean()
              ** 0.5, " (eV/atoms) [NOTE: Only elements specified in Onebody are considered in atom count!]")

    def write_error_forces(self, mdl_for, ref_for, fname="CCS_error_forces.out"):
        """Prints the errors in a file.

        Args:

            mdl_eng (ndarray): Energy prediction values from splines.
            ref_eng (ndarray): Reference energy values.
            mse (float): Mean square error.
            fname (str, optional): Output filename (default: 'error.out').

        """
        header = "{:<15}{:<15}{:<15}".format("Reference", "Predicted", "Error")
        error = abs(ref_for - mdl_for)
        maxerror = max(abs(error))
        mse = ((error) ** 2).mean()

        footer = "MSE = {:2.5E}\nMaxerror = {:2.5E}".format(mse, maxerror)
        np.savetxt(fname, np.transpose([ref_for, mdl_for, error]), header=header,
                   footer=footer, fmt='%-15.5f')

    def write_CCS_params(self,fname="CCS_params.json"):

        CCS_params = OrderedDict()
        CCS_params["Charge scaling factor"] = float(self.charge_scaling)

        eps_params = OrderedDict()
        for k in range(self.no):
            if self.l_one[k].epsilon_supported:
                eps_params[self.l_one[k].name] = self.l_one[k].epsilon
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
            two_body_dict["exp_b"] = self.l_twb[k].expcoeffs[1]
            two_body_dict["exp_c"] = self.l_twb[k].expcoeffs[2]
            a_values = list(self.l_twb[k].splcoeffs[:, 0])
            a_values.append(0)
            two_body_dict["spl_a"] = a_values
            b_values = list(self.l_twb[k].splcoeffs[:, 1])
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

    def gen_Buckingham(self):
        print("Getting to generate a Buckingham potential from the spline data!") 