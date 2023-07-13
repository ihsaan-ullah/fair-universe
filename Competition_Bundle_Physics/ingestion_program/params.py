# ------------------------------------------------------
# Class Params
# initiates and computes parameters for data generation
# ------------------------------------------------------

import numpy as np


class Params:

    def __init__(self,
                 pi=0.1,
                 nu_1=10000,
                 mu_range=[0.9, 1.1],
                 systematics=[],
                 verbose=True):

        self.translation = None

        self.verbose = verbose
        self.systematics = systematics

        # 1. Draw $\mu$ uniformly between 0.9 and 1.1
        self.mu = np.round(np.random.uniform(mu_range[0], mu_range[1], 1)[0], 2)

        # 2. Set systematics
        self.set_systematics()

        # 3. Initialize $\nu_1$ = 10000
        self.nu_1 = nu_1

        # 4. Initilaize $\pi$ = 0.01
        self.pi = pi

        # 5. Compute $\nu$ = $\nu_1 (\mu \pi + (1-\pi))$
        self.nu = int(self.nu_1 * (self.mu * self.pi + (1-self.pi)))

        # 6. Compute $\gamma$ = $\nu \pi$
        self.gamma = self.nu_1 * self.pi

        # 7. Compute $\beta$ = $\nu1 (1-\pi)$
        self.beta = self.nu_1 * (1-self.pi)

        # 8. Compute $p_s$ = $\pi \mu /(\mu \pi + (1-\pi))$
        self.p_s = np.round(self.mu*self.pi / (self.mu*self.pi + (1-self.pi)), 2)

        # 9. Compute $p_b$  = $(1 - \pi)/(\mu \pi + (1-\pi))$
        self.p_b = np.round((1-self.pi) / (self.mu*self.pi + (1-self.pi)), 2)

        # 10. Draw $N \sim Poisson(\nu)$
        self.N = np.random.poisson(self.nu)

        if self.verbose:
            print("------------------")
            print("Toy 2D Parameters")
            print("------------------")
            print(f"pi = {self.pi}\nmu = {self.mu}\nnu = {self.nu}\nnu1 = {self.nu_1}\nbeta = {self.beta}\ngamma = {self.gamma}\nps = {self.p_s}\npb = {self.p_b}\nN = {self.N}\n")

    def set_systematics(self):

        for systematic in self.systematics:

            if systematic["name"] == "Translation":

                z_range = systematic["z_range"]
                z_angles = systematic["z_angles"]

                # Draw $z_magnitude$ uniformly between range
                z_magnitude = np.random.uniform(z_range[0], z_range[1], 1)[0]

                # Draw $alpha$ randomly between range
                alpha = np.random.choice(z_angles)

                self.translation = {
                    "name": "Translation",
                    "z_magnitude": z_magnitude,
                    "alpha": alpha
                }

    def get_translation(self):
        return self.translation

    def get_p_s(self):
        return self.p_s

    def get_p_b(self):
        return self.p_b

    def get_mu(self):
        return self.mu

    def get_pi(self):
        return self.pi

    def get_nu(self):
        return self.nu

    def get_nu_1(self):
        return self.nu_1

    def get_N(self):
        return self.N
