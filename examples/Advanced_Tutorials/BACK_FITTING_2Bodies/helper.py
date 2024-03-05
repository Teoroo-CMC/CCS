from ase.io import read, write
from ase.build import bulk
import numpy as np
import ase.db as db
from ase.visualize import view
from ase.calculators.lj import LennardJones
import matplotlib.pyplot as plt
import json
import sympy
import matplotlib.pyplot as plt
from ase.units import kB
from ccs_fit.ase_calculator.ccs_ase_G2B import G2B
from ase.calculators.singlepoint import SinglePointCalculator


class fit_task:
    def __init__(
        self,
        pairs=None,
        crystal=None,
        Tm=0.0,
        r_cut=0.0,
        V_ij=None,
        charge_dict=None,
        damping=0.5,
        CCS_res=0.1,
    ):
        self.pairs = pairs
        self.crystal = crystal
        cell = self.crystal.get_cell()
        n_repeat = r_cut * np.linalg.norm(np.linalg.inv(cell), axis=0)
        n_repeat = np.ceil(n_repeat).astype(int)
        self.train_cell = self.crystal * n_repeat
        print("Required supercell size based on R_cut is: ", n_repeat)
        self.r_cut = r_cut
        self.Tm = Tm
        self.V_ij = V_ij
        self.V = {}
        self.charge_dict = charge_dict
        self.damping = damping
        self.CCS_res = CCS_res
        self.Espan = 3 * kB * Tm

    def assign_params(self, dict):
        for pair in dict:
            st = self.V_ij
            for key in dict[pair]:
                st = st.replace(key, str(dict[pair][key]))
            self.V[pair] = st
        # BUILD INPUT FOR GENERAL TWO BODY (G2B) AND CCS

        # CCS
        CCS_input = {
            "General": {
                "interface": "CCS+Q",
                "merging": "True",
                "do_unconstrained_fit": "True",
            },
            "Twobody": {},
        }

        for pair in self.pairs:
            CCS_input["Twobody"][pair] = {
                "Rcut": self.r_cut,
                "Resolution": self.CCS_res,
                "Swtype": "sw",
                "const_type": "Mono",
            }

        # SAVE TO FILE
        with open("CCS_input.json", "w") as f:
            json.dump(CCS_input, f, indent=8)

        # G2B
        self.G2B_input = input = {
            "Charge scaling factor": 1.0,
            "One_body": {},
            "Two_body": {},
        }

        for pair in self.pairs:
            self.G2B_input["Two_body"][pair] = {
                "r_min": 0.0,
                "r_cut": self.r_cut,
                "V_func": self.V[pair],
            }

        with open("G2B_input.json", "w") as f:
            json.dump(self.G2B_input, f, indent=8)

    def evaluateG2B(self, pair, r):
        Vofr = []
        V = sympy.sympify(self.V[pair])
        for rs in r:
            f_eval = V.subs({"r_ij": rs})
            Vofr.append(f_eval.evalf())
        return Vofr

    def init_training(self, min_scale=0.0, max_scale=1e12, vol_scan=None, scan=True):
        calc = G2B(
            G2B_params=self.G2B_input,
            charge=True,
            q=self.charge_dict,
            charge_scaling=True,
        )
        self.train_cell.calc = calc

        self.E0 = self.train_cell.get_potential_energy() / len(self.train_cell)
        orig_cell = self.train_cell.get_cell()
        orig_struc = self.train_cell.copy()
        print("Energy widow: ", self.E0, " - ", self.E0 + self.Espan, " eV/atom")

        vols = []
        Es = []

        if vol_scan is not None:
            self.vol_scan = vol_scan
        # FIRST DETERMINE VOLUME RANGE AND APPEND RELEVANT STRUCTURES TO TRAININGSET
        self.min_scale = min_scale
        self.max_scale = max_scale
        Emax = self.E0 + self.Espan
        if scan:
            counter = 0
            for scale in np.arange(0.8, 1.30, 0.01):
                new_struct = self.train_cell.copy()
                new_struct.set_cell(self.train_cell.cell * scale, scale_atoms=True)
                new_struct.calc = self.train_cell.calc
                nrg = new_struct.get_potential_energy() / len(new_struct)
                if nrg - Emax < 0 and self.min_scale == 0.0:
                    self.min_scale = scale
                if nrg - Emax < 0:
                    self.max_scale = scale
                if nrg - Emax < 0:
                    counter += 1
                    print(
                        "Found structure with scale : ",
                        scale,
                        " and energy",
                        nrg,
                    )
                vols.append(
                    (scale**3) * self.train_cell.get_volume() / len(self.train_cell)
                )
                Es.append(nrg)

            self.vol_scan = {"V": vols, "E": Es}

    def scramble(self, size=10, DB=None, damping=None):
        from Harmonic_sampler import simple_shake

        G2B_DB = db.connect(DB)
        self.DB = DB
        counter = 0
        if damping is not None:
            self.damping = damping
        # APPEND SCRAMBELED STRUCTURES TO THE TRAININGSET
        while counter < size:
            scale = (
                np.random.random() * (self.max_scale - self.min_scale) + self.min_scale
            )
            new_struct = self.train_cell.copy()
            new_struct.set_cell(self.train_cell.cell * scale, scale_atoms=True)
            new_struct = simple_shake(new_struct, self.Espan * self.damping * scale)
            new_struct.calc = self.train_cell.calc
            F = new_struct.get_forces()
            E = new_struct.get_potential_energy()
            nrg = new_struct.get_potential_energy() / len(new_struct)
            if nrg - self.E0 - self.Espan < 0:
                counter += 1
                calculator = SinglePointCalculator(
                    new_struct, energy=E, free_energy=E, forces=F
                )
                new_struct.calc = calculator
                new_struct.get_potential_energy()
                new_struct.get_forces()
                print(
                    "Added strucutre No: ",
                    counter,
                    " with E: ",
                    nrg,
                    "  Scale: ",
                    scale,
                )
                G2B_DB.write(new_struct)

    def compare_q(self, UNC=False):
        with open("CCS_params.json", "r") as f:
            CCS_params = json.load(f)
        if UNC == True:
            with open("UNC_params.json", "r") as f:
                CCS_params = json.load(f)

        sq = CCS_params["Charge scaling factor"]

        print("--- Comparison of charges ---")
        for sp in self.charge_dict:
            print(
                sp,
                " :",
                sq * self.charge_dict[sp],
                " vs target:",
                self.charge_dict[sp],
            )
        return np.abs(sq - 1.0) * 100.0

    def plot_summary(self):
        with open("CCS_params.json", "r") as f:
            CCS_params = json.load(f)
        with open("UNC_params.json", "r") as f:
            UNC_params = json.load(f)

        for pair in self.pairs:
            r = np.array(CCS_params["Two_body"][pair]["r"])
            e = CCS_params["Two_body"][pair]["spl_a"]
            e_UNC = UNC_params["Two_body"][pair]["spl_a"]
            e_ref = self.evaluateG2B(pair, r)
            # plt.xlim(2.5,4)
            # plt.ylim(-0.1,0.25)
            plt.xlabel("Distance (Å)")
            plt.ylabel("Potential (eV)")
            plt.plot(r, e_ref, color="black", label="True potential " + pair)
            plt.plot(
                r,
                np.array(e),
                "--",
                color="red",
                label="Fitted potential " + pair,
            )
            plt.plot(
                r,
                np.array(e_UNC),
                "--",
                color="blue",
                label="Fitted unconst. potential " + pair,
            )
            # plt.plot(r,np.array(e)-np.array(e_ref),color='black',label="Potential difference "+pair)
            plt.legend()
            plt.show()

        try:
            err_F = np.loadtxt("CCS_error_forces.out")
            plt.xlabel("Reference force (eV/Å)")
            plt.ylabel("Fitted force (eV/Å)")
            plt.plot(
                [min(err_F[:, 0]), max(err_F[:, 0])],
                [min(err_F[:, 0]), max(err_F[:, 0])],
                "--",
                color="black",
            )
            plt.scatter(
                err_F[:, 0],
                err_F[:, 1],
                facecolors="none",
                edgecolors="red",
                alpha=0.1,
            )
            plt.show()
        except:
            pass

        err_F = np.loadtxt("CCS_error.out")
        err_F[:, 0] = err_F[:, 0] / err_F[:, 3]
        err_F[:, 1] = err_F[:, 1] / err_F[:, 3]
        plt.xlabel("Reference Energy (eV/atom)")
        plt.ylabel("Fitted Energy (eV/atom)")
        plt.plot(
            [min(err_F[:, 0]), max(err_F[:, 0])],
            [min(err_F[:, 0]), max(err_F[:, 0])],
            "--",
            color="black",
        )
        plt.scatter(
            err_F[:, 0],
            err_F[:, 1],
            facecolors="none",
            edgecolors="red",
            alpha=0.8,
        )
        plt.show()

        fig, ax = plt.subplots(figsize=(6, 6))  # set the figure size to 6x6 inches
        ax.set_xlabel("Energy (eV/atom)")
        ax.set_ylabel("Error (meV/atom)")
        # ax.set_ylim(-2,2)
        ax.hexbin(
            err_F[:, 1],
            y=1000 * (err_F[:, 1] - err_F[:, 0]),
            gridsize=20,
            cmap="Blues",
        )
        plt.show()

    def check_sampling(self):
        fig, ax = plt.subplots(figsize=(6, 6))  # set the figure size to 6x6 inches

        ax.set_xlabel(r"Volume (Å$^3$/atom)")
        ax.set_ylabel("Energy (eV/atom)")
        ax.set_ylim(self.E0 - 0.1, self.E0 + self.Espan + 0.1)
        ax.set_xlim(
            self.min_scale**3 * self.train_cell.get_volume() / len(self.train_cell) - 1,
            self.max_scale**3 * self.train_cell.get_volume() / len(self.train_cell) + 1,
        )
        ax.plot(
            [
                self.min_scale**3 * self.train_cell.get_volume() / len(self.train_cell),
                self.max_scale**3 * self.train_cell.get_volume() / len(self.train_cell),
            ],
            [self.E0 + self.Espan, self.E0 + self.Espan],
            "--",
            color="black",
        )
        ax.plot(self.vol_scan["V"], np.array(self.vol_scan["E"]), color="black")
        DB = db.connect(self.DB)
        vols = []
        EvsV = []
        for row in DB.select():
            vols.append(row.volume / row.natoms)
            EvsV.append(row.energy / row.natoms)

        ax.hexbin(x=vols, y=np.array(EvsV), gridsize=20, cmap="Blues")
        plt.show()  # show the plot

    def calculate_overlap_mae(self, UNC=False):
        from scipy.integrate import trapz
        from ccs_fit.ase_calculator.ccs_ase_calculator import spline_table

        overlap = {}
        tot_overlap = 0
        with open("CCS_params.json", "r") as f:
            CCS_params = json.load(f)
        if UNC == True:
            with open("UNC_params.json", "r") as f:
                CCS_params = json.load(f)

        for pair in self.pairs:
            elem1, elem2 = pair.split("-")
            tb = spline_table(elem1, elem2, CCS_params)
            rs = np.arange(
                CCS_params["Two_body"][pair]["r_min"],
                CCS_params["Two_body"][pair]["r_cut"],
                0.001,
            )
            e_CCS = [tb.eval_energy(elem) for elem in rs]
            e_ref = self.evaluateG2B(pair, rs)
            abs_de = np.abs(np.array(e_CCS) - np.array(e_ref))
            I = trapz(abs_de, rs)
            overlap[pair] = I / (rs[-1] - rs[0])  # Integral divided by interval lenght
            tot_overlap += I / (rs[-1] - rs[0])
        overlap["Total"] = tot_overlap
        return overlap

    def calculate_overlap_rmse(self, UNC=False):

        from scipy.integrate import trapz
        from ccs_fit.ase_calculator.ccs_ase_calculator import spline_table

        overlap = {}
        tot_overlap = 0
        with open("CCS_params.json", "r") as f:
            CCS_params = json.load(f)
        if UNC == True:
            with open("UNC_params.json", "r") as f:
                CCS_params = json.load(f)

        for pair in self.pairs:
            elem1, elem2 = pair.split("-")
            tb = spline_table(elem1, elem2, CCS_params)
            rs = np.arange(
                CCS_params["Two_body"][pair]["r_min"],
                CCS_params["Two_body"][pair]["r_cut"],
                0.001,
            )
            e_CCS = [tb.eval_energy(elem) for elem in rs]
            e_ref = self.evaluateG2B(pair, rs)
            abs_de = (np.array(e_CCS) - np.array(e_ref)) ** 2
            I = trapz(abs_de, rs)
            overlap[pair] = (
                I / (rs[-1] - rs[0])
            ) ** 0.5  # Integral divided by interval lenght
            tot_overlap += (I / (rs[-1] - rs[0])) ** 0.5
        overlap["Total"] = tot_overlap
        return overlap

    # The analytical integrals cannot be solved I think...
    # Perhaps if we use the square of the function instead...
    # def calculate_overlap(self):
    #     overlap=0
    #     with open("CCS_params.json", "r") as f:
    #         CCS_params = json.load(f)

    #     for pair in self.pairs:
    #         dr=CCS_params["Two_body"][pair]["r"][1]-CCS_params["Two_body"][pair]["r"][0]
    #         rs=CCS_params["Two_body"][pair]["r"]
    #         for i,r in enumerate(rs):
    #             a=CCS_params["Two_body"][pair]["spl_a"][i]
    #             b=CCS_params["Two_body"][pair]["spl_b"][i]
    #             c=CCS_params["Two_body"][pair]["spl_c"][i]
    #             d=CCS_params["Two_body"][pair]["spl_d"][i]
    #             Vtmp=f"abs({self.V[pair]}-{str(a)} - {str(b)}*(R_ij-{str(r)}) - {str(c)}*(R_ij-{str(r)})**2 - {str(d)}*(R_ij-{str(r)})**3)"
    #             sympy.sympify(Vtmp)
    #             r_ij = sympy.Symbol('r_ij')
    #             I=sympy.integrate(Vtmp, (r_ij,r,r+dr))
    #             val=I.evalf()
    #             #overlap += float(val)
    #             print(float(val))

    #     return overlap

    def calculate_overlap_CCSs_rmse(CCS_params1, CCS_params2):
        from scipy.integrate import trapz
        from ccs_fit.ase_calculator.ccs_ase_calculator import spline_table

        tot_overlap = 0
        with open(CCS_params1, "r") as f:
            CCS_params1 = json.load(f)
        with open(CCS_params2, "r") as f:
            CCS_params2 = json.load(f)

        for pair in self.pairs:
            elem1, elem2 = pair.split("-")
            tb = spline_table(elem1, elem2, CCS_params)
            rs = np.arange(
                CCS_params["Two_body"][pair]["r_min"],
                CCS_params["Two_body"][pair]["r_cut"],
                0.001,
            )
            e_CCS = [tb.eval_energy(elem) for elem in rs]
            e_ref = self.evaluateG2B(pair, rs)
            abs_de = (np.array(e_CCS) - np.array(e_ref)) ** 2
            I = trapz(abs_de, rs)
            overlap[pair] = (
                I / (rs[-1] - rs[0])
            ) ** 0.5  # Integral divided by interval lenght
            tot_overlap += (I / (rs[-1] - rs[0])) ** 0.5
        overlap["Total"] = tot_overlap
        return overlap
