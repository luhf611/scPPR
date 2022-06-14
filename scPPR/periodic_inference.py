import numpy as np
import pandas as pd
from scipy.stats import norm

np.random.seed()

class PeriodicInference(object):
    def __init__(self, N, N_iterations_MCMC = int(1e5), N_iterations_MCMC_optimisation = int(1e4), N_passes_optimisation = 3):
        # N - number of phase
        if N % 2 == 1:
            N += 1
        self.N = N
        self.N_iterations_MCMC = N_iterations_MCMC
        self.N_iterations_MCMC_optimisation = N_iterations_MCMC_optimisation
        self.N_passes_optimisation = N_passes_optimisation

        self.stage_tau = 2 * np.pi / N
        self.all_phases = np.arange(N) * self.stage_tau

    @staticmethod
    def generate_data(n_samples, n_biomarkers):
        biomarker_phases = np.random.rand(n_biomarkers) * 2 * np.pi
        sample_phases = np.random.rand(n_samples, 1) * 2 * np.pi
        mu = np.tile(sample_phases, (1, n_biomarkers))
        mu = np.cos(mu - biomarker_phases)
        data = np.random.normal(mu, np.ones(mu.shape))
        return biomarker_phases, sample_phases, data

    def estimate_uncertainty(self, data, phase_init=None):
        """Estimate uncertainty

        Parameters
        ----------
        data: array-like of shape (n_biomarkers, n_samples)
            log2(TPM + 1)
        phase_init: array-like of shape (n_biomarkers, )
            init phases of biomarkers
        
        Returns
        -------
        ml_phases: array, shape (n_biomarkers, 1)
            the most probable phases of biomarkers found across MCMC samples
        ml_likelihood: array, shape (1, )
            the likelihood of the most probable model found across MCMC samples
        samples_phase: array, shape (n_biomarkers, N_iterations_MCMC)
            samples of the probable phases obtained from MCMC sampling
        samples_likeilhood:array, shape (N_iterations_MCMC, )
            samples of the likelihood of each model sampled by the MCMC sampling
        """

        n_biomarkers = data.shape[1]
        if not phase_init:
            phase_init = np.random.choice(self.all_phases, n_biomarkers)
            phase_init[0] = 0
        # phase_init = np.random.rand(n_biomarkers) * 2 * np.pi
        print("phase_init: ", phase_init)
        # Perform a few initial passes where the perturbation sizes of the MCMC uncertainty estimation are tuned
        
        print("Optimise mcmc settings")
        phase_sigma_opt = self.optimise_mcmc_settings(data, phase_init)
        print("phase_sigma_opt: ", phase_sigma_opt)

        # Run the full MCMC algorithm to estimate the uncertainty
        print("Run mcmc")
        return self.perform_mcmc(data, phase_init, self.N_iterations_MCMC, phase_sigma_opt)

    def optimise_mcmc_settings(self, data, phase_init):
        phase_sigma_currentpass = 1.0

        for i in range(self.N_passes_optimisation):
            _, _, samples_phase_currentpass, _ = self.perform_mcmc(data, phase_init, self.N_iterations_MCMC_optimisation, phase_sigma_currentpass)

            phase_sigma_currentpass = np.std(samples_phase_currentpass, axis=1, ddof=1)  # np.std is different to Matlab std, which normalises to N-1 by default
            phase_sigma_currentpass[phase_sigma_currentpass < 0.01] = 0.01  # magic number

        return phase_sigma_currentpass

    def perform_mcmc(self, data, phase_init, n_iterations, phase_sigma):
        # Take MCMC samples of the uncertainty in the model parameters
        n_biomarkers = data.shape[1]

        samples_phase = np.zeros((n_biomarkers, n_iterations))
        samples_likelihood = np.zeros((n_iterations, 1))
        samples_phase[:, 0] = phase_init

        accept = 0
        for i in range(n_iterations):
            # if i % (n_iterations / 10) == 0:
            #     print('Iteration', i, 'of', n_iterations, ',', int(float(i) / float(n_iterations) * 100.), '% complete')
            if i > 0:
                current_phase = samples_phase[:, i-1]

                if isinstance(phase_sigma, float):
                    this_phase_sigma = np.ones(current_phase.shape) * phase_sigma
                else:
                    this_phase_sigma = phase_sigma

                new_phase = current_phase.copy()
                index = np.random.randint(1, n_biomarkers)
                # confine the phase of the first biomarker to [0, pi] to avoid sampling oscillation
                if index == 1:
                    choice_phase = self.all_phases[:self.N//2+1]
                else:
                    choice_phase = self.all_phases
                weight = self.calc_coeff(this_phase_sigma[index]) * self.calc_exp(choice_phase, current_phase[index], this_phase_sigma[index])
                weight /= np.sum(weight)
                new_phase[index] = np.random.choice(choice_phase, 1, p=weight)
            

                samples_phase[:, i] = new_phase

            likelihood_sample, _, _ = self.calculate_likelihood(data, samples_phase[:, i])
            samples_likelihood[i] = likelihood_sample

            if i > 0:
                ratio = np.exp(samples_likelihood[i] - samples_likelihood[i-1])
                if ratio < np.random.rand():
                    samples_likelihood[i] = samples_likelihood[i-1]
                    samples_phase[:, i] = samples_phase[:, i-1]
                else:
                    accept += 1
            
            if i % (n_iterations / 10) == 0:
                print('Iteration', i, 'of', n_iterations, ',', int(float(i) / float(n_iterations) * 100.), '% complete')

        print('Accept Ratio:', accept / n_iterations)
        ml_likelihood = max(samples_likelihood)
        perm_index = np.where(samples_likelihood == ml_likelihood)
        perm_index = perm_index[0]
        ml_phases = samples_phase[:, perm_index]

        return ml_phases, ml_likelihood, samples_phase, samples_likelihood

    def calculate_likelihood(self, data, biomarker_phases):
        """Compute likelihood

        Parameters
        ----------
        data: array-like of shape (n_biomarkers, n_samples)
            log2(TPM + 1)
        biomarker_phases: array-like of shape (n_biomarkers, )
            estimated phases of biomarkers
        
        Returns
        -------
        loglike: the log-likelihood of the current model
        total_prob_subj: the total probability of the current model for each subject
        p_perm_k: the probability of each subjects data at each stage of each subtype in the current model
        """

        N = self.N
        n_samples, n_biomarkers = data.shape
        tau_val = np.linspace(0, np.pi*2, N+1)

        point_value = np.tile(tau_val, (n_biomarkers, 1))
        point_value = np.cos(point_value - np.reshape(biomarker_phases, (n_biomarkers, 1)))

        # shape: n_biomarkers * N
        stage_initial_value = point_value[:, :-1]
        stage_final_value = point_value[:, 1:]

        stage_initial_tau = tau_val[:-1]
        stage_final_tau = tau_val[1:]

        # slope of stage
        stage_a = (stage_final_value - stage_initial_value) / self.stage_tau
        # intercept of stage
        stage_b = stage_initial_value - stage_a * stage_initial_tau
        std_biomarker = np.ones(n_biomarkers)

        # shape: n_samples * N
        iterative_mean = (np.tile(data[:, 0], (N, 1)).T - stage_b[0, :]) / stage_a[0, :]
        # shape: N
        iterative_std = std_biomarker[0] / stage_a[0, :]
        iterative_kappa = np.ones((n_samples, N))

        for b in range(1, n_biomarkers):
            mu1 = iterative_mean
            mu2 = (np.tile(data[:, b], (N, 1)).T - stage_b[b, :]) / stage_a[b, :]
            std1 = iterative_std
            std2 = std_biomarker[b] / stage_a[b, :]
            cov1 = np.power(std1, 2)
            cov2 = np.power(std2, 2)

            munew = (cov2 * mu1 + cov1 * mu2) / (cov1 + cov2)
            covnew = cov1 * cov2 / (cov1 + cov2)
            kappaval = norm.pdf(mu1, mu2, np.sqrt(cov1 + cov2))
            iterative_mean = munew
            iterative_std = np.sqrt(covnew)
            iterative_kappa = iterative_kappa * kappaval

        iterative_const = 1 / np.prod(np.abs(stage_a), axis=0)
        cdf_diff_val = norm.cdf(stage_final_tau, iterative_mean, iterative_std) - norm.cdf(stage_initial_tau, iterative_mean, iterative_std)
        p_perm_k = iterative_const * iterative_kappa * cdf_diff_val

        total_prob_subj = np.sum(p_perm_k, 1)
        loglike = sum(np.log(total_prob_subj + 1e-250))

        return loglike, total_prob_subj, p_perm_k

    def predict_phases(self, data, biomarker_phases):
        """Estimate phase of each sample

        Parameters
        ----------
        data: array-like of shape (n_biomarkers, n_samples)
            log2(TPM + 1)
        biomarker_phases: array-like of shape (n_biomarkers, )
            estimated phases of biomarkers
        """
        _, _, p_perm_k = self.calculate_likelihood(data, biomarker_phases)
        phases_index = np.argmax(p_perm_k, 1)
        return self.all_phases[phases_index]

    @staticmethod
    def calc_coeff(sig):
        return 1. / np.sqrt(np.pi * 2.0) * sig

    @staticmethod
    def calc_exp(x, mu, sig):
        x = (x - mu) / sig
        return np.exp(-.5 * x * x)
