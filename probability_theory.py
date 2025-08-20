from scipy.stats import chi2, norm, kstest, bootstrap, ttest_ind, ks_2samp, mannwhitneyu, cramervonmises, pearsonr, spearmanr
from scipy.special import gamma, erf
from scipy.optimize import fsolve, root_scalar
from scipy.integrate import quad
import numpy as np
import pandas as pd
import warnings

class ProbClass:
    '''
    The ProbClass class contains all probability theory distributions and related functions to perform statistical analysis.
    This version includes the following probability distributions
        GUM: Gumbel
        GEV: Generalized Extreme Value
        LOG: Log-Normal
        GPD: Generalized Perato
        EXP: Exponential
        NOR: Normal
    It has the capability to perform bootstrapping on probability functions or other regular functions with single input.
    '''
    def __init__(self,):
        warnings.simplefilter("ignore", RuntimeWarning)
        self.start_years_list:list = ['1 year', '2 years', '3 years', '4 years']
        self.start_year:int = 2
        self.start_status:list = ['Start-up', 'Regular Operation']
        self.static_indices:list = ['mean', 'median', 'std', 'min', 'max', '5%', '95%']
        self.static_funcs:list = [np.mean, np.median, np.std, np.min, np.max, np.percentile, np.percentile]
        self.static_table_columns: list = ['Yearly volumes of gas flared per capacity (bcm/bcm [%])', \
                                           'Number of flares detected per year (#)', \
                                           'Number of consecutive days with flaring (days)']
        self.static_table_title:list = ['Yearly Volume of Gas Flared per Capacity', 'Yearly Number of Flaring Days', 'Number of Consecutive Days with Flaring']
        self.static_table_units: list = ['(bcm/bcm [%])', '(#)', '(days)']

    def check_class(self, class_obj, name:str, nameClass:str):
        if not hasattr(class_obj, "__init__"): raise ValueError(f'{name} is not the {nameClass} object')
        return

    def normal_by_lmoments(self, sample:np.ndarray):
        return self.lmoment_normal_fit(self.samlmom2(sample))
    
    def lmoment_normal_fit(self, lmom):
        """
        Estimate Normal distribution parameters (mu, sigma) from L-moments.

        Parameters:
            l1 : First L-moment (mean)
            l2 : Second L-moment

        Returns:
            mu : mean of the normal distribution
            sigma : standard deviation
        """
        return lmom[0], np.pi*lmom[1]
    
    def check_correlation(self, vnf, out1=None, out2=None, out3=None):
        """
        check_correlation function determines the pearson and spearman correlation between volume of gas flared and number of flaring days and capacity
        it returns the dataframe to save from main script.
        """
        if out1 is None: out1, out2, out3 = pd.DataFrame({}), pd.DataFrame({}), pd.DataFrame({})
        vol = vnf.gas_filter * vnf.capacity_filter
        df = pd.DataFrame({'VNF':vnf.vnf_filter.values.flatten(), 'GAS':vol.values.flatten(), 'Capacity':vnf.capacity_filter.values.flatten()})
        df = df.where(df>0, np.nan).dropna().sort_values(by=['GAS'])

        pr = pearsonr(df['VNF'].values, df['GAS'].values)
        sp = spearmanr(df['VNF'].values, df['GAS'].values)
        out1.at[vnf.sname, 'pearson-static'] = np.round(pr.statistic, 3)
        out1.at[vnf.sname, 'pearson-pvalue'] = np.round(pr.pvalue, 3)
        out1.at[vnf.sname, 'spearman-static'] = np.round(sp.statistic, 3)
        out1.at[vnf.sname, 'spearman-pvalue'] = np.round(sp.pvalue, 3)
        
        pr = pearsonr(df['VNF'].values, df['Capacity'].values)
        sp = spearmanr(df['VNF'].values, df['Capacity'].values)
        out2.at[vnf.sname, 'pearson-static'] = np.round(pr.statistic, 3)
        out2.at[vnf.sname, 'pearson-pvalue'] = np.round(pr.pvalue, 3)
        out2.at[vnf.sname, 'spearman-static'] = np.round(sp.statistic, 3)
        out2.at[vnf.sname, 'spearman-pvalue'] = np.round(sp.pvalue, 3)
        
        pr = pearsonr(df['Capacity'].values, df['GAS'].values)
        sp = spearmanr(df['Capacity'].values, df['GAS'].values)
        out3.at[vnf.sname, 'pearson-static'] = np.round(pr.statistic, 3)
        out3.at[vnf.sname, 'pearson-pvalue'] = np.round(pr.pvalue, 3)
        out3.at[vnf.sname, 'spearman-static'] = np.round(sp.statistic, 3)
        out3.at[vnf.sname, 'spearman-pvalue'] = np.round(sp.pvalue, 3)
        
        return out1, out2, out3

    def normalfun(self, ftype, x, parhat):
        """
        Evaluate PDF, CDF, or Quantile for a Normal distribution.

        Parameters:
            ftype : str, one of ['dnorm', 'pnorm', 'qnorm']
            x : input array (values for 'dnorm' and 'pnorm', probabilities for 'qnorm')
            parhat : [mu, sigma] parameters of the normal distribution

        Returns:
            array of evaluated results
        """
        ftype = ftype.lower()
        if ftype not in ['dnorm', 'pnorm', 'qnorm']: raise ValueError("Invalid function type. Use 'dnorm', 'pnorm', or 'qnorm'.")

        mu, sigma = parhat
        if sigma <= 0: raise ValueError("Standard deviation must be positive.")
        x = np.asarray(x, dtype=float)

        if ftype == 'dnorm': f = norm.pdf(x, loc=mu, scale=sigma)
        elif ftype == 'pnorm':f = norm.cdf(x, loc=mu, scale=sigma)
        elif ftype == 'qnorm':
            if np.any((x < 0) | (x > 1)): raise ValueError("Probabilities for 'qnorm' must be in [0, 1].")
            f = norm.ppf(x, loc=mu, scale=sigma)

        return f

    def boot_normal(self, x, ftype, query, nboot=1000, fx=1.0, conf=0.95):
        """
        BOOT.NORMAL evaluate EXPONENTIAL functions with uncertainty estimates by bootstrapping.
        x : array-like, data used for estimating the empirical distribution function
        ftype : 'pnorm', 'qnorm' function type to be evaluated
        query : array-like, data points at which the function is evaluated
        nboot : int, number of bootstrap samples for uncertainty estimation
        fx : float, fraction of data to be sampled
        """

        ftype = ftype.lower()
        if ftype not in ['pnorm', 'qnorm']: raise ValueError("Invalid function type. Use 'pnorm', 'qnorm'.")

        if ftype == 'qnorm' and (np.any(query <= 0) or np.any(query >= 1)): raise ValueError("boot.normal: invalid query points.")

        x = np.asarray(x, dtype=float)
        n = len(x)
        np.random.default_rng()
        x_sample = np.random.choice(x, size=(nboot, int(fx * n)), replace=True)

        query = np.asarray(query, dtype=float)
        n_query = len(query)
        y = np.empty((nboot, n_query))

        for i in range(nboot):
            parhat = self.normal_by_lmoments(x_sample[i, :])
            y[i, :] = self.normalfun(ftype, query, parhat)

        parhat_full = self.normal_by_lmoments(x)        
        y[0, :] = self.normalfun(ftype, query, parhat_full)

        ci_low, ci_high = self.bca_confidence_intervals(y, conf)
        return y[0, :], ci_low, ci_high

    def gpd_by_lmoments(self, sample:np.ndarray):
        return self.lmoment_gpd_fit(self.samlmom2(sample))

    def lmoment_gpd_fit(self, lmom):
        """
        Estimate Generalized Pareto distribution parameters (k, alpha, xi) from L-moments.

        Parameters:
            l1 : First L-moment
            l2 : Second L-moment
            t3 : L-skewness (tau_3)

        Returns:
            k_hat : shape parameter
            alpha_hat : scale parameter
            xi_hat : location parameter
        """
        l1, l2, t3 = [lmom[0], lmom[1], lmom[2] / lmom[1]]
        # Estimate shape parameter k
        k_hat = (1 - 3 * t3) / (1 + t3)

        # Estimate scale alpha
        alpha_hat = l2 * (1 + k_hat) * (2 + k_hat)

        # Estimate location xi
        xi_hat = l1 - alpha_hat / (1 + k_hat)

        return -k_hat, alpha_hat, xi_hat
    
    def gpdfun(self, ftype, x, parhat):
        """
        Evaluate PDF, CDF, or quantile function of the Generalized Pareto distribution (Wikipedia form).

        Inputs:
            ftype : str, one of ['dgp', 'pgp', 'qgp']
            x : array-like input (values for 'dgp' and 'pgp', probabilities for 'qgp')
            parhat : [k, alpha, xi] where:
                k     : shape parameter
                alpha : scale parameter (must be > 0)
                xi    : location parameter
        Returns:
            Evaluated array of PDF, CDF, or quantile values.
        """
        ftype = ftype.lower()
        if ftype not in ['dgp', 'pgp', 'qgp']: raise ValueError("Invalid function type. Use 'dgp', 'pgp', or 'qgp'.")

        k, alpha, xi = parhat
        if alpha <= 0: raise ValueError("Scale parameter alpha must be positive.")

        x = np.asarray(x, dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            if ftype == 'dgp':
                z = (x - xi) / alpha
                if k != 0:
                    f = np.where(
                        (z >= 0) & (1 + k * z > 0),
                        (1 / alpha) * np.power(1 + k * z, -1 / k - 1),
                        0.0)
                else:
                    f = np.where(z >= 0, (1 / alpha) * np.exp(-z), 0.0)

            elif ftype == 'pgp':
                z = (x - xi) / alpha
                if k != 0:
                    f = np.where(
                        (z >= 0) & (1 + k * z > 0),
                        1 - np.power(1 + k * z, -1 / k),
                        0.0)
                else:
                    f = np.where(z >= 0, 1 - np.exp(-z), 0.0)

                # Clip for numerical stability
                f = np.clip(f, 1e-12, 1 - 1e-12)

            elif ftype == 'qgp':
                if np.any((x < 0) | (x > 1)): raise ValueError("Probabilities must be in [0, 1].")

                if k != 0: f = xi + (alpha / k) * (np.power(1 - x, -k) - 1)
                else: f = xi - alpha * np.log(1 - x)

            return f
        raise ValueError("GPD fun failed!")
    
    def boot_gpd(self, x, ftype, query, nboot=1000, fx=1.0, conf=0.95):
        """
        BOOT.GPD evaluate EXPONENTIAL functions with uncertainty estimates by bootstrapping.
        x : array-like, data used for estimating the empirical distribution function
        ftype : 'pexp', 'qexp' function type to be evaluated
        query : array-like, data points at which the function is evaluated
        nboot : int, number of bootstrap samples for uncertainty estimation
        fx : float, fraction of data to be sampled
        """

        ftype = ftype.lower()
        if ftype not in ['pgp', 'qgp']: raise ValueError("Invalid function type. Use 'pgp', 'qgp'.")

        if ftype == 'qgp' and (np.any(query <= 0) or np.any(query >= 1)): raise ValueError("boot.gpd: invalid query points.")

        x = np.asarray(x)
        n = len(x)
        np.random.default_rng()
        x_sample = np.random.choice(x, size=(nboot, int(fx * n)), replace=True)

        query = np.asarray(query)
        n_query = len(query)
        y = np.empty((nboot, n_query))

        for i in range(nboot):
            parhat = self.gpd_by_lmoments(x_sample[i, :])
            y[i, :] = self.gpdfun(ftype, query, parhat)

        parhat_full = self.gpd_by_lmoments(x)        
        y[0, :] = self.gpdfun(ftype, query, parhat_full)

        ci_low, ci_high = self.bca_confidence_intervals(y, conf)
        return y[0, :], ci_low, ci_high

    def exponential_by_lmoments(self, sample:np.ndarray):
        return self.lmoment_exponential_fit(self.samlmom2(sample), sample)
    
    def lmoment_exponential_fit(self, lmom, sample):
        """
        Estimate (mu, sigma) fpr the 2-parameter Exponential distribution using L-moments
        Based on Hosking (1999).
        
        Parameters:
            l1: First L-moment
            l2: Second L-moment

        Returns:
            alpha : scale parameter of the exponential distribution
            xi : location parameter (minimum value of the distribution)
        """
        l1, l2 = [lmom[0], lmom[1]]
        alpha = 2 * l2
        # if alpha <= 0:
        #     l1, l2 = np.median(sample), np.std(sample)/np.sqrt(np.pi)
        #     alpha = 2 * l2
        xi = l1 - alpha

        return alpha, xi

    def exponentialfun(self, ftype, x, parhat):
        """
        Computes PDF, CDF, or quantile function for a 2-parameter exponential distribution.

        Parameters:
            ftype : 'dexp', 'pexp', or 'qexp'
            x : array-like, evaluation input (value or probability)
            parhat : [alpha, xi], exponential parameters

        Returns:
            f : evaluated output (PDF, CDF, or quantile)
        """
        ftype = ftype.lower()
        if ftype not in ['dexp', 'pexp', 'qexp']: raise ValueError("Invalid function type. Use 'dexp', 'pexp', or 'qexp'.")

        alpha, xi = parhat
        if alpha <= 0: raise ValueError("Invalid scale parameter alpha.")

        x = np.asarray(x).astype(float)

        if ftype == 'dexp': f = np.where(x >= xi, (1 / alpha) * np.exp(-(x - xi) / alpha), 0.0)
        elif ftype == 'pexp': f = np.where(x >= xi, 1 - np.exp(-(x - xi) / alpha), 0.0)
        elif ftype == 'qexp':
            if np.any((x < 0) | (x > 1)): raise ValueError("For qexp, input probabilities must be in [0, 1].")
            f = xi - alpha * np.log(1 - x)

        return f
    
    def boot_exponential(self, x, ftype, query, nboot=1000, fx=1.0, conf=0.95):
        """
        BOOT.EXPONENFUN evaluate EXPONENTIAL functions with uncertainty estimates by bootstrapping.
        x : array-like, data used for estimating the empirical distribution function
        ftype : 'pexp', 'qexp' function type to be evaluated
        query : array-like, data points at which the function is evaluated
        nboot : int, number of bootstrap samples for uncertainty estimation
        fx : float, fraction of data to be sampled
        """

        ftype = ftype.lower()
        if ftype not in ['pexp', 'qexp']: raise ValueError("Invalid function type. Use 'pexp', 'qexp'.")

        if ftype == 'qexp' and (np.any(query <= 0) or np.any(query >= 1)): raise ValueError("boot.exponenfun: invalid query points.")

        x = np.asarray(x)
        n = len(x)
        np.random.default_rng()
        x_sample = np.random.choice(x, size=(nboot, int(fx * n)), replace=True)

        query = np.asarray(query)
        n_query = len(query)
        y = np.empty((nboot, n_query))

        for i in range(nboot):
            parhat = self.exponential_by_lmoments(x_sample[i, :])
            y[i, :] = self.exponentialfun(ftype, query, parhat)

        parhat_full = self.exponential_by_lmoments(x)        
        y[0, :] = self.exponentialfun(ftype, query, parhat_full)

        ci_low, ci_high = self.bca_confidence_intervals(y, conf)
        return y[0, :], ci_low, ci_high

    def lognormal_by_lmoments(self, sample:np.ndarray):
        return self.lmoment_lognormal_fit_inverse(self.samlmom2(sample))

    def lmoment_lognormal_fit_inverse(self, lmom):
        """
        Estimate (mu, sigma, xi) for the 3-parameter log-normal distribution using L-moments.
        Based on Hosking (1990), using fsolve and lmom[3] as initial guess for sigma.

        Parameters:
            lmom : list or array-like
                L-moments: [l1, l2, l3, initial_sigma_guess]

        Returns:
            mu_hat : Mean of underlying normal distribution
            sigma_hat : Std dev of underlying normal distribution
            xi_hat : Location parameter of the original log-normal distribution
        """
        l1, l2, t3 = lmom[0], lmom[1], lmom[2] / lmom[1]
        sigma_init = 0.5 # np.pi * l2 if lmom[3] < 0 else lmom[3]

        def tau3_of_sigma(sigma):
            if sigma <= 0:
                return 1e6  # Prevent invalid domain

            def integrand(x):
                return erf(x / np.sqrt(3)) * np.exp(-x**2)

            integral_result, _ = quad(integrand, 0, sigma / 2)
            tau3 = (6 / np.sqrt(np.pi)) * (1 / erf(sigma / 2)) * integral_result
            return tau3

        def objective(sigma):
            return tau3_of_sigma(sigma[0]) - t3

        # Use fsolve to estimate sigma
        sigma_solution = fsolve(objective, x0=[sigma_init], xtol=1e-9)
        sigma_hat = sigma_solution[0]
        if sigma_hat <= 0:
            raise RuntimeError(f"fsolve failed: estimated sigma = {sigma_hat}")

        # Compute mu
        erf_half_sigma = erf(sigma_hat / 2)
        A = l2 / erf_half_sigma
        mu_hat = np.log(A) - 0.5 * sigma_hat**2

        # Compute xi
        xi_hat = l1 - np.exp(mu_hat + 0.5 * sigma_hat**2)

        return mu_hat, sigma_hat, xi_hat

    def lognormalfun(self, ftype, x, parhat):
        """
        LOGNORMALFUN computes pdf, cdf, or quantile of a shifted Log-normal distribution.

        Parameters:
            ftype : str
                'dlogn' = PDF, 'plogn' = CDF, 'qlogn' = quantile function
            x : array-like
                Input values (non-negative for 'dlogn' and 'plogn', probabilities [0,1] for 'qlogn')
            parhat : list or array-like
                [mu, sigma, xi] where:
                    mu    : mean of the underlying normal distribution
                    sigma : std dev of the underlying normal distribution
                    xi    : location shift (original variable = xi + lognormal(mu, sigma))

        Returns:
            f : array
                Output values corresponding to selected function type
        """
        ftype = ftype.lower()
        if ftype not in ['dlogn', 'plogn', 'qlogn']: raise ValueError("Invalid function type. Use 'dlogn', 'plogn', or 'qlogn'.")

        mu, sigma, xi = parhat
        x = np.asarray(x).astype(float)

        if sigma <= 0: raise ValueError("lognormalfun: invalid sigma parameter.")

        if ftype == 'dlogn':
            f = np.where(
                x > xi,
                (1 / ((x - xi) * sigma * np.sqrt(2 * np.pi))) *
                np.exp(-((np.log(x - xi) - mu) ** 2) / (2 * sigma ** 2)),
                0.0)

        elif ftype == 'plogn':
            f = np.zeros_like(x)
            mask = x > xi
            f[mask] = norm.cdf((np.log(x[mask] - xi) - mu) / sigma)
            # f = np.where(
            #     x > xi,
            #     norm.cdf((np.log(x - xi) - mu) / sigma),
            #     0.0)

        elif ftype == 'qlogn':
            if np.any((x < 0) | (x > 1)): raise ValueError("lognormalfun: for quantile function, x must be in [0, 1].")
            f = xi + np.exp(mu + sigma * norm.ppf(x))

        return f

    def boot_lognormal(self, x, ftype, query, nboot=1000, fx=1.0, conf=0.95):
        """
        BOOT.LOGNFUN evaluate LOGNORMAL functions with uncertainty estimates by bootstrapping.
        x : array-like, data used for estimating the empirical distribution function
        ftype : 'plogn', 'qlogn' function type to be evaluated
        query : array-like, data points at which the function is evaluated
        nboot : int, number of bootstrap samples for uncertainty estimation
        fx : float, fraction of data to be sampled
        """

        ftype = ftype.lower()
        if ftype not in ['plogn', 'qlogn']: raise ValueError("Invalid function type. Use 'plogn', 'qlogn'.")

        if ftype == 'qlogn' and (np.any(query <= 0) or np.any(query >= 1)): raise ValueError("boot.lognfun: invalid query points.")

        x = np.asarray(x)
        n = len(x)
        np.random.default_rng()
        x_sample = np.random.choice(x, size=(nboot, int(fx * n)), replace=True)

        query = np.asarray(query)
        n_query = len(query)
        y = np.empty((nboot, n_query))

        for i in range(nboot):
            parhat = self.lognormal_by_lmoments(x_sample[i, :])
            y[i, :] = self.lognormalfun(ftype, query, parhat)

        parhat_full = self.lognormal_by_lmoments(x)        
        y[0, :] = self.lognormalfun(ftype, query, parhat_full)

        ci_low, ci_high = self.bca_confidence_intervals(y, conf)
        return y[0, :], ci_low, ci_high

    def gumbel_by_lmoments(self, sample:np.array):
        lmom = self.samlmom2(sample)
        scale   = lmom[1] / np.log(2)
        loc     = lmom[0] - np.euler_gamma * scale
        return [loc, scale, 0]

    def hybrid_gev_params(self, sample:np.array):
        """
        hybrid_pargev_fsolve estimates the parameters of the Generalized Extreme Value 
        distribution given the L-moments of samples followed by feasibility adjustment
        inspired by D.J. Dupuis & M. Tsao (1998)
        """

        parhat = self.pargev_fsolve(self.samlmom2(sample))
        upper_bound = parhat[0] - parhat[1] / parhat[2]

        if parhat[2] < 0:
            max_sample = np.max(sample)
            if upper_bound < max_sample: parhat = [parhat[0], parhat[1], parhat[1]/(max_sample - parhat[0])]
        
        return parhat

    def samlmom2(self, sample:np.array):
        """
        samlmom2 returns the first three L-moments of samples
        sample is the 1-d array
        n is the total number of the samples, j is the j_th sample
        """
        n = len(sample)
        sample = np.sort(sample.reshape(n))[::-1]
        b0 = np.mean(sample)
        b1 = np.array([(n - j - 1) * sample[j] / n / (n - 1)
                    for j in range(n)]).sum()
        b2 = np.array([(n - j - 1) * (n - j - 2) * sample[j] / n / (n - 1) / (n - 2)
                    for j in range(n - 1)]).sum()
        c = ((2*b1 - b0) / (3*b2 - b0)) - (np.log(2) / np.log(3))
        initial_guess = 7.8590 * c + 2.9554 * (c ** 2)
        lmom1 = b0
        lmom2 = 2 * b1 - b0
        lmom3 = 6 * (b2 - b1) + b0
        return lmom1, lmom2, lmom3, initial_guess

    def pargev_fsolve(self, lmom):
        """
        pargev_fsolve estimates the parameters of the Generalized Extreme Value 
        distribution given the L-moments of samples
        """
        lmom_ratios = [lmom[0], lmom[1], lmom[2] / lmom[1]]
        f = lambda x, t: 2 * (1 - 3**(-x)) / (1 - 2**(-x)) - 3 - t
        G = fsolve(f, x0=lmom[3], args=(lmom_ratios[2]))[0]
        para3 = G
        GAM = gamma(1 + G)
        para2 = lmom_ratios[1] * G / (GAM * (1 - (2**-G)))
        para1 = lmom_ratios[0] - para2 * (1 - GAM) / G
        return para1, para2, -para3

    def gevfun(self, ftype, x, parhat):
        """
        GEVFUN pdf, cdf, and quantile of a GEV distribution.
        ftype : 'dgev', 'pgev', or 'qgev' for pdf, cdf, and quantile functions
        x : array-like, data array for function evaluation
        parhat : list, vector of GEV parameter estimates
        """

        ftype = ftype.lower()
        if ftype not in ['dgev', 'pgev', 'qgev']: raise ValueError("Invalid function type. Use 'dgev', 'pgev', or 'qgev'.")

        if x is None or len(x) == 0: raise ValueError("gevfun: x must be a non-empty data array.")
        x = np.array(x).astype(float)
        mu, sigma, ksi = parhat
        if sigma <= 0: raise ValueError("gevfun: invalid scale parameter.")

        if ftype == 'dgev':
            w = ((x - mu) / sigma).astype(float)
            z = w * ksi
            z = np.maximum(z, -1)

            if np.abs(ksi) < np.finfo(float).eps: f = -np.log(sigma) - w - np.exp(-w)
            else: f = -np.log(sigma) - (1 / ksi + 1) * np.log1p(z) - np.exp((-1 / ksi) * np.log1p(z))

            f[np.isinf(f)] = -np.inf
            f = np.exp(f)

        elif ftype == 'pgev':
            w = (x - mu) / sigma
            z = w * ksi
            z = np.maximum(z, -1)

            if np.abs(ksi) < np.finfo(float).eps: f = np.exp(-np.exp(-w))
            else: f = np.exp(-np.exp((-1 / ksi) * np.log1p(z)))

        elif ftype == 'qgev':
            if np.any(x > 1) or np.any(x < 0): raise ValueError("gevfun: for quantile function, x must be values in [0, 1].")

            if np.abs(ksi) < np.finfo(float).eps: f = mu - sigma * np.log(-np.log(x))
            else: f = mu + (sigma / ksi) * (np.exp(-ksi * np.log(-np.log(x))) - 1)

        return f

    def boot_gevfun(self, x, ftype, query, nboot=1000, fx=1.0, do_gumbel=False, conf=0.95):
        """
        BOOT.GEVFUN evaluate GEV functions with uncertainty estimates by bootstrapping.
        x : array-like, data used for estimating the empirical distribution function
        ftype : 'pgev' or 'qgev', function type to be evaluated
        query : array-like, data points at which the function is evaluated
        nboot : int, number of bootstrap samples for uncertainty estimation
        fx : float, fraction of data to be sampled
        """

        ftype = ftype.lower()
        if ftype not in ['pgev', 'qgev']: raise ValueError("Invalid function type. Use 'pgev' or 'qgev'.")

        if ftype == 'qgev' and (np.any(query <= 0) or np.any(query >= 1)): raise ValueError("boot.gevfun: invalid query points.")

        x = np.asarray(x)
        n = len(x)
        np.random.default_rng()
        x_sample = np.random.choice(x, size=(nboot, int(fx * n)), replace=True)

        query = np.asarray(query)
        n_query = len(query)
        y = np.empty((nboot, n_query))

        for i in range(nboot):
            if not do_gumbel: parhat = self.hybrid_gev_params(x_sample[i, :])
            else: parhat = self.gumbel_by_lmoments(x_sample[i, :])

            y[i, :] = self.gevfun(ftype, query, parhat)

        if not do_gumbel: parhat_full = self.hybrid_gev_params(x)
        else: parhat_full = self.gumbel_by_lmoments(x)
        
        y[0, :] = self.gevfun(ftype, query, parhat_full)

        ci_low, ci_high = self.bca_confidence_intervals(y, conf)
        return y[0, :], ci_low, ci_high

        # alpha = 1 - conf
        # return y[0, :], np.quantile(y, alpha/2, axis=0), np.quantile(y, 1-alpha/2, axis=0)

    def boot_emperical_function(self, x, func, nboot=9999, fx=1.0, conf=0.95, *args, **kwargs):
        """
        BOOT.FUN evaluate input functions with uncertainty estimates by bootstrapping.
            x : array-like, data used for estimating the empirical distribution function
            nboot : int, number of bootstrap samples for uncertainty estimation
            fx : float, fraction of data to be sampled
            conf: default is 95% confidence interval
            *args, **kwargs : additional arguments passed to func (e.g., q=95 for np.percentile)
        Returns
            the output of function on input data, lower confidence, upper confidence
        """
        if not callable(func): raise ValueError("Function input apply bootstrap is invalid!")

        x = np.asarray(x, dtype=float)
        check_na = pd.isna(x)
        if check_na.any(): x = x[~check_na]
        n = len(x)
        np.random.default_rng()
        x_sample = np.random.choice(x, size=(nboot, int(fx * n)), replace=True)

        y = np.empty((nboot, 1))

        for i in range(nboot):
            y[i, :] = func(x_sample[i, :], *args, **kwargs)

        y[0, :] = func(x, *args, **kwargs)

        ci_low, ci_high = self.bca_confidence_intervals(y, conf)
        return y[0, 0], ci_low[0], ci_high[0]

    def bca_confidence_intervals(self, bootstrap_samples, conf):
        """
        Calculate the BCa confidence intervals for the bootstrap samples.
        bootstrap_samples: array-like, bootstrap estimates
        conf: float, confidence level
        Returns:
        ci_lower: array, lower bounds of the confidence intervals
        ci_upper: array, upper bounds of the confidence intervals
        """
        nboot, n_query = bootstrap_samples.shape
        alpha = 1 - conf
        
        ci_lower = np.zeros(n_query)
        ci_upper = np.zeros(n_query)
        
        
        for i in range(n_query):
            # Calculate bias-correction (z0)
            z0 = norm.ppf((bootstrap_samples[:, i] < bootstrap_samples[0, i]).mean())
            
            if np.isinf(z0): z0 = np.sign(z0) * 10

            # Jackknife acceleration
            jackknife_estimates = np.array([np.mean(np.delete(bootstrap_samples[:, i], j)) for j in range(nboot)])
            jackknife_mean = jackknife_estimates.mean()

            a_num = np.sum((jackknife_mean - jackknife_estimates) ** 3)
            a_den = np.sum((jackknife_mean - jackknife_estimates) ** 2) ** 1.5
            a = a_num / (6.0 * a_den) if a_den != 0 else 0.0

            # Adjusted alpha percentiles and Ensure quantiles are valid
            alpha1 = np.clip(norm.cdf(z0 + (z0 + norm.ppf(alpha / 2)) / (1 - a * (z0 + norm.ppf(alpha / 2)))), 0, 1)
            alpha2 = np.clip(norm.cdf(z0 + (z0 + norm.ppf(1 - alpha / 2)) / (1 - a * (z0 + norm.ppf(1 - alpha / 2)))), 0, 1)

            ci_lower[i] = np.quantile(bootstrap_samples[:, i], alpha1 )
            ci_upper[i] = np.quantile(bootstrap_samples[:, i], alpha2 )
        
        
        return ci_lower, ci_upper

    def MLT_test(self, data:np.ndarray, nullFun:str, alterFun:str):
        """
        Perform Maximum Likelihood test to compare Null hypothesis function and Alternative hypotethis function
        Input:
            Flattened data: 1D numpy array. the np.nan values should be removed.
            nullFun (str): name of Null hypothesis function. Available options
                GUM
                EXP
                NOR
            alterFun (str): name of Alternative hypotethis function. Available options
                GEV
                GPD
                LOG
        Output:
            Likelihood ratio statistic and p-value: (np.float64, np.float64)
        """
        if not nullFun in ['GUM', 'EXP', 'NOR'] and not alterFun in ['GEV', 'GPD', 'LOG']:
            raise ValueError("Wrong null and alternative distribution for MLT_test is selected") 
        # Estimate parameters under the (null hypothesis) and (alternative hypothesis)
        log_like_null, log_like_alte = self.get_log_like_fun(nullFun, data), self.get_log_like_fun(alterFun, data)
        # Calculate the likelihood ratio test statistic
        lr_statistic = -2 * (log_like_null - log_like_alte)
        # Degrees of freedom is the difference in the number of parameters
        df = abs(self.get_num_param_fun(alterFun) - self.get_num_param_fun(nullFun))
        return lr_statistic, chi2.sf(lr_statistic, df)
    
    def get_log_like_fun(self, fun_name:str, data:np.ndarray) -> np.float64:
        """
        This function calculate the log_likelihood for the following prob_fun
            GUM: Gumbel
            GEV: Generalized Extreme Value
            LOG: Log-Normal
            GPD: Generalized Perato
            EXP: Exponential
            NOR: Normal
        Input:
            function_name: str
            flatten_data: 1D numpy array. the np.nan values should be removed.
        return
            log_likelihood: np.float64
        """
        # Estimate parameters under the Gumbel distribution
        if fun_name == 'GUM': return np.log(self.gevfun('dgev', data, self.gumbel_by_lmoments(data))).sum()
        # Estimate parameters under the GEV distribution
        if fun_name == 'GEV': return np.log(self.gevfun('dgev', data, self.hybrid_gev_params(data))).sum()
        # Estimate parameters under the LogNormal distribution
        if fun_name == 'LOG': return np.log(self.lognormalfun('dlogn', data, self.lognormal_by_lmoments(data))).sum()
        # Estimate parameters under the Exponential distribution
        if fun_name == 'EXP': return np.log(self.exponentialfun('dexp', data, self.exponential_by_lmoments(data))).sum()
        # Estimate parameters under the Generalized Pareto distribution
        if fun_name == 'GPD': return np.log(self.gpdfun('dgp', data, self.gpd_by_lmoments(data))).sum()
        # Estimate parameters under the Normal distribution
        if fun_name == 'NOR': return np.log(self.normalfun('dnorm', data, self.normal_by_lmoments(data))).sum()
        raise ValueError("Wrong input for get_log_like_fun")
        
    def get_num_param_fun(self, fun_name:str) -> np.int16:
        """
        This function gets the name of function
            GUM: Gumbel
            GEV: Generalized Extreme Value
            LOG: Log-Normal
            GPD: Generalized Perato
            EXP: Exponential
            NOR: Normal
        Returns the number of paramter of the fitted function
            Intiger: np.int16
        """
        if fun_name == 'GUM': return 2
        if fun_name == 'GEV': return 3
        if fun_name == 'LOG': return 3
        if fun_name == 'EXP': return 2
        if fun_name == 'GPD': return 3
        if fun_name == 'NOR': return 2
        raise ValueError("Wrong input for get_num_param_fun")

    def AIC_BIC_test(self, data, prob_funs:list=['GUM', 'GEV',]):
        """
        Perform AIC and BIC test. Since the sample size might be small, this function also calculate the AICc which is the AIC corrected.
        This function calculate the statistic for the following prob_funs
            GUM: Gumbel
            GEV: Generalized Extreme Value
            LOG: Log-Normal
            GPD: Generalized Perato
            EXP: Exponential
            NOR: Normal
        Input:
            Flattend_data: 1D numpy array. the np.nan values should be removed.
        Output:
            3 numpy arrays. First one contains the AICc. Second array contains AIC, The third array contains BIC
        """
        n = len(data)
        log_likes = [self.get_log_like_fun(i, data) for i in prob_funs]
        n_params  = [self.get_num_param_fun(i) for i in prob_funs]
        AIC_all   = np.array([2*n_fun - 2*log_like for n_fun, log_like in zip(n_params, log_likes)])
        AICc_all  = np.array([AIC + (2 * (n_fun**2) + n_fun*2)/(n- n_fun - 1) for AIC, n_fun in zip(AIC_all, n_params)])
        BIC_all   = np.array([n_fun * np.log(n) - 2 * log_like for n_fun, log_like in zip(n_params, log_likes)])
        return AICc_all, AIC_all, BIC_all

    def kstest_distribution(self, flattened_data, j):
        """
        Perform the one sample Kolmogorov-Smirnov test for goodness of fit
        Available prob_fun:
            GUM: Gumbel
            GEV: Generalized Extreme Value
            LOG: Log-Normal
            GPD: Generalized Perato
            EXP: Exponential
            NOR: Normal
        """

        if j == 'GUM': cdf_func = lambda x: self.gevfun('pgev', x, self.gumbel_by_lmoments(flattened_data))
        elif j == 'GEV': cdf_func = lambda x: self.gevfun('pgev', x, self.hybrid_gev_params(flattened_data))
        elif j == 'LOG': cdf_func = lambda x: self.lognormalfun('plogn', x, self.lognormal_by_lmoments(flattened_data))
        elif j == 'EXP': cdf_func = lambda x: self.exponentialfun('pexp', x, self.exponential_by_lmoments(flattened_data))
        elif j == 'GPD': cdf_func = lambda x: self.gpdfun('pgp', x, self.gpd_by_lmoments(flattened_data))
        elif j == 'NOR': cdf_func = lambda x: self.normalfun('pnorm', x, self.normal_by_lmoments(flattened_data))
        else: return 'Wrong Input'

        return kstest(flattened_data, cdf_func).pvalue

    def cvmtest_distribution(self, flattened_data, j):
        """
        Perform the one sample Kolmogorov-Smirnov test for goodness of fit
        Available prob_fun:
            GUM: Gumbel
            GEV: Generalized Extreme Value
            LOG: Log-Normal
            GPD: Generalized Perato
            EXP: Exponential
            NOR: Normal
        """

        if j == 'GUM': cdf_func = lambda x: self.gevfun('pgev', x, self.gumbel_by_lmoments(flattened_data))
        elif j == 'GEV': cdf_func = lambda x: self.gevfun('pgev', x, self.hybrid_gev_params(flattened_data))
        elif j == 'LOG': cdf_func = lambda x: self.lognormalfun('plogn', x, self.lognormal_by_lmoments(flattened_data))
        elif j == 'EXP': cdf_func = lambda x: self.exponentialfun('pexp', x, self.exponential_by_lmoments(flattened_data))
        elif j == 'GPD': cdf_func = lambda x: self.gpdfun('pgp', x, self.gpd_by_lmoments(flattened_data))
        elif j == 'NOR': cdf_func = lambda x: self.normalfun('pnorm', x, self.normal_by_lmoments(flattened_data))
        else: return 'Wrong Input'

        return cramervonmises(flattened_data, cdf_func).pvalue
    
    def distinguish_prob_funs(self, list_funs):
        '''
        This function gets the list of probability distribution names and returns two lists: null distributions and alternative distributions
        null distributions have 2 parameter and alternative distributions have 3 parameter
        Input:
            list of prob_funs including
            GUM: Gumbel
            GEV: Generalized Extreme Value
            LOG: Log-Normal
            GPD: Generalized Perato
            EXP: Exponential
            NOR: Normal
        Output:
            NULL_FUNCTIONS: list(str), ALTERNATIVE_FUNCTIONS: list(str) 
        '''
        return [i for i in list_funs if i in ['GUM', 'NOR', 'EXP']], [i for i in list_funs if i in ['GEV', 'GPD', 'LOG']]

    def significant_test(self, a, b, test='ttest') -> np.float64:
        """
        significant_test function performs one of the 'ttest', 'ks_2samp', 'mannwhitneyu' test on two
        samples and returns pvalue
        """
        if not test in ['ttest', 'ks_2samp', 'mannwhitneyu']: raise ValueError("available significant tests are 'ttest', 'ks_2samp', 'mannwhitneyu'") 
        if test == 'ttest': return ttest_ind(a, b, equal_var=False).pvalue
        if test == 'ks_2samp': return ks_2samp(a, b).pvalue
        return mannwhitneyu(a, b).pvalue
    
    def classify_pvalue(self, inp_:np.float64) -> str:
        """
        Classify the pvalue into ***, **, *, ^ for pvalue less than 0.0005, 0.005, 0.05, and above 0.05, respectively.
        """
        if inp_ < 0.005: return '***'
        if inp_ < 0.05: return "**"
        if inp_ < 0.1: return "*"
        return "^"

    def create_static_table(self, vnf:list, test:str='ttest'):
        """
        Create Distribution table of non-startup period for volume of gas flared per capacity, annual number of flaring days, and length of flaring episodes
        The statics include confidence intervals.
        Input:
            vnf: vnf object = VNF filtered for non-start-up scenario
            test: str = one of the 'ttest', 'ks_2samp', 'mannwhitneyu'
        """
        self.check_class(vnf, 'vnf', 'VNF')
        if not test in ['ttest', 'ks_2samp', 'mannwhitneyu']: raise ValueError("available significant tests are 'ttest', 'ks_2samp', 'mannwhitneyu'") 

        df:pd.DataFrame = self.merge_categories([self.rm_nan_fl(vnf.gas_filter_start.values), \
                                                 self.rm_nan_fl(vnf.vnf_filter_start.values), \
                                                 self.rm_nan_fl(vnf.flareEpisodes_filter_start['length'])])

        output = pd.DataFrame(columns=self.static_table_columns, index=self.static_indices)
        self.extract_static_table_single(df, ['Category'], output).to_csv(f'./Results/Distributions/Distribution_{vnf.sname}.csv')

    def merge_categories(self, vals:list, vnf_filter_name:str='') -> pd.DataFrame:
        if len(vals) != 3: raise ValueError("input list for merge_categories function should has length of 3 with order of gas, vnf, length")

        categs:list = [self.static_table_columns[0]] * len(vals[0]) + \
                      [self.static_table_columns[1]] * len(vals[1]) + \
                      [self.static_table_columns[2]] * len(vals[2])
        
        return pd.DataFrame({'values':np.concatenate(vals), 'Category':categs, 'vnf_filter_name':vnf_filter_name})
    
    def create_static_table_startup(self, vnfs:list, test:str='ttest'):
        """
        Create Distribution table of startup period for volume of gas flared per capacity, annual number of flaring days, and length of flaring episodes
        The statics include confidence intervals.
        Input:
            vnfs: list = [vnf1, vnf2, vnf3, vnf4]
            vnf1: vnf object = VNF filtered for 1 year  start-up
            vnf2: vnf object = VNF filtered for 2 years start-up
            vnf3: vnf object = VNF filtered for 3 years start-up
            vnf4: vnf object = VNF filtered for 4 years start-up
            test: str = one of the 'ttest', 'ks_2samp', 'mannwhitneyu'
        """
        for i_, vnf in enumerate(vnfs):
            self.check_class(vnf, f'vnf {i_}', 'VNF')
        if not test in ['ttest', 'ks_2samp', 'mannwhitneyu']: raise ValueError("available significant tests are 'ttest', 'ks_2samp', 'mannwhitneyu'") 
        
        sname = vnfs[0].sname
        df_gas, static_ttest = self.prep_df_start(dfs=[vnf.gas_filter_start for vnf in vnfs], from_pivot=False, test=test)
        output = pd.DataFrame(index=self.static_indices, columns=pd.MultiIndex.from_product([self.start_years_list, self.start_status]))
        self.extract_static_table(df_gas, ['Year', 'OperatingTime'], output, type_gas=True).to_csv(f'./Results/Distributions/Startup_gas_{sname}.csv')

        df_vnf, static_ttest = self.prep_df_start(dfs=[vnf.vnf_filter_start for vnf in vnfs], from_pivot=False, test=test)
        output = pd.DataFrame(index=self.static_indices, columns=pd.MultiIndex.from_product([self.start_years_list, self.start_status]))
        self.extract_static_table(df_vnf, ['Year', 'OperatingTime'], output, type_gas=False).to_csv(f'./Results/Distributions/Startup_vnf_{sname}.csv')


        df_len, static_ttest = self.prep_df_start(dfs=[vnf.flareEpisodes_filter_start for vnf in vnfs], from_pivot=True, test=test)
        output = pd.DataFrame(index=self.static_indices, columns=pd.MultiIndex.from_product([self.start_years_list, self.start_status]))
        self.extract_static_table(df_len, ['Year', 'OperatingTime'], output, type_gas=False).to_csv(f'./Results/Distributions/Startup_length_{sname}.csv')
        return
    
    def extract_static_table(self, dfIn:pd.DataFrame, groupColumns:list, output:pd.DataFrame, type_gas:bool=False):
        """
        Extract the statistics and the confidence interval of data based on groupColumns on the dfIn and store the results in output.
        """
        grouped = dfIn.groupby(groupColumns)
        for col in output.columns:
            for ind_, func in zip(self.static_indices, self.static_funcs):
                output = self.find_percision(cond=type_gas, dfQ=output, index_in=ind_, col_in=col, \
                                                values_in=self.set_boot_empirical_func(self.rm_nan_fl(grouped.get_group(col)['values'].values), ind_, func))
        return output
    
    def extract_static_table_single(self, dfIn:pd.DataFrame, groupColumns:list, output:pd.DataFrame):
        """
        Extract the statistics and the confidence interval of data based on groupColumns on the dfIn and store the results in output.
        """
        grouped = dfIn.groupby(groupColumns)
        for in_, col in enumerate(output.columns):
            for ind_, func in zip(self.static_indices, self.static_funcs):
                output = self.find_percision(cond=in_ == 0, dfQ=output, index_in=ind_, col_in=col, \
                                                values_in=self.set_boot_empirical_func(self.rm_nan_fl(grouped.get_group((col,))['values'].values), ind_, func))                
        return output

    def find_percision(self, cond:bool, dfQ:pd.DataFrame, index_in, col_in, values_in):
        """
        this fucntion is used in extract_static_table and extract_static_table_single functions
        Inputs:
            cond1 = whether it is volume of gas flared per capacity column or not
            dfQ = the input dataframe
            index_in = index of dfQ to update
            col_in = column of dfQ to update
            values_in = values to update in dfQ
        Output:
            dfQ: pd.DataFrame will be returned
        """
        if cond:
            dfQ.at[index_in, col_in] = f'{values_in[0]:.2g} ({values_in[1]:.2g}, {values_in[2]:.2g})'
            return dfQ
        dfQ.at[index_in, col_in] = f'{values_in[0]:.2g} ({values_in[1]:.2g}, {values_in[2]:.2g})'
        return dfQ

    def set_boot_empirical_func(self, values:np.ndarray, static_type:str, func):
        """
        Appropriately set the boot_emperical_function for the desired static function
        5% stands for np.percentile(data, q=5)
        95% stands for np.percentile(data, q=95)
        """
        if static_type == '5%': return self.boot_emperical_function(values, func, q=5, nboot=1000)
        if static_type == '95%': return self.boot_emperical_function(values, func, q=95, nboot=1000)
        return self.boot_emperical_function(values, func, nboot=1000)

    def prep_df_start(self, dfs:list, from_pivot:bool=True, test='ttest') -> pd.DataFrame:
        """
        The prep_df_start takes four VNF filtered obj, representing 1 year, 2 years, 3 years, 4 years startup periods.
        This function prepares the data to be used in create_df_start function
        Input:
            dfs: list = [df1, df2, df3, df4]
            df1: pd.DataFrame = VNF filtered for 1 year  start-up
            df2: pd.DataFrame = VNF filtered for 2 years start-up
            df3: pd.DataFrame = VNF filtered for 3 years start-up
            df4: pd.DataFrame = VNF filtered for 4 years start-up
        Output:
            output: pd.DataFrame = to be used inside plot_start_up_priod_box_plot function
            static_ttest: dict = which classify the statistic significance difference
        """
        if not type(dfs) is list and len(dfs) != 4: raise ValueError("prep_df_start function requires the list of four years startup DataFrames as the Input.")
        
        if not from_pivot:
            static_ttest = {i: self.classify_pvalue(self.significant_test(self.rm_nan_fl(df.iloc[:,:j].values), \
                                                          self.rm_nan_fl(df.iloc[:,j:].values), test=test)) \
                                                                for i, j, df in zip(self.start_years_list, [1, 2, 3, 4], dfs)}

            output = self.create_df_start({i:{'Start-up':self.rm_nan_fl(df.iloc[:,:j].values), \
                                            'Regular Operation':self.rm_nan_fl(df.iloc[:,j:].values)} \
                                                            for i, j, df in zip(self.start_years_list, [1, 2, 3, 4], dfs)})
        else:
            static_ttest = {i: self.classify_pvalue(self.significant_test(self.rm_nan_fl(df.loc[df.index.get_level_values(1) <= j, 'length'].values), \
                                                          self.rm_nan_fl(df.loc[df.index.get_level_values(1) > j, 'length'].values), test=test)) \
                                                                for i, j, df in zip(self.start_years_list, [1, 2, 3, 4], dfs)}

            output =  self.create_df_start({i:{'Start-up':self.rm_nan_fl(df.loc[df.index.get_level_values(1) <= j, 'length'].values), \
                                            'Regular Operation':self.rm_nan_fl(df.loc[df.index.get_level_values(1) > j, 'length'].values)} \
                                                            for i, j, df in zip(self.start_years_list, [1, 2, 3, 4], dfs)})
        return output, static_ttest

    def rm_nan_fl(self, data:np.ndarray, dtype=np.float64) -> np.ndarray:
        """
        rm_nan_fl function create a numpy array as type flaot (default), flatten the data and return non-nan values
        """
        temp = np.asarray(data, dtype=dtype).flatten(); temp = temp[temp != 0]
        return temp[~pd.isna(temp)]

    def create_df_start(self, dict_dfs:dict) -> pd.DataFrame:
        """
        The create_dataframe_sns gets a dictionary of data to create a DataFrame to be used with sns (i.e., seaborn) library
        This function cleans and flatten data and provides an appropriate format. This function works with startup period
        
        Input:
            dict_dfs: {'1 year':dict_data_1_year, '2 years':dict_data_2_years, ...}
            dict_data:dict = {'Start-up':data_start_up, 'Regular Operation':data_regular}
        Output:
            pd.DataFrame
        """
        return pd.concat([pd.DataFrame({'Year': y, 'OperatingTime': s, 'values': dict_dfs[y][s]}) \
                                            for y in self.start_years_list \
                                                for s in self.start_status], ignore_index=True)
    
    def merge_GOF(self, snames:list, labels=None):
        if labels is None: labels = self.start_status
        if len(labels) != len(snames): raise ValueError("labels and vnfs must have the same length.")
        
        out = pd.DataFrame(index=pd.Index(self.static_table_title, name='paramter'), columns=pd.MultiIndex.from_product([['cmvtest', 'kstest'], labels]))
        for sname, label in zip(snames, labels):
            for col in ['cmvtest', 'kstest']:
                out[(col, label)] = sname[col]
        out.to_csv(f'./Results/Statics/GOF_merged.csv')
        return
    
    def create_scenario_table(self, vnfs:None, labels=None, probabilities_values:np.ndarray=np.arange(0.9, 0, -0.1), prob_fun="LOG", UseStartUpForRegular=False):
        """
        Create the csv table for secnario design form startup and onlyReularOperation VNF_filters.
        Input:
            vnfs: list = [vnf1, vnf2]
            vnf1: vnf object = VNF filtered for 2 years start-up
            vnf2: vnf object = VNF filtered for only regular operation
        Output:
            Scenario DataFrame saved in Results/Statics
        """
        for i_, vnf in enumerate(vnfs):
            self.check_class(vnf, f'vnf {i_}', 'VNF')
        if labels is None: labels = self.start_status
        if len(labels) != len(vnfs): raise ValueError("labels and vnfs must have the same length.")
        
        prob_index = (probabilities_values*100).astype(int)
        return_periods = np.asarray([1/i for i in probabilities_values])
        return_periods_prob = 1 - 1 / return_periods
        out = pd.DataFrame( index=pd.MultiIndex.from_product([labels, prob_index], names=['Operation Type', 'Probability [%]']), \
                            columns=pd.MultiIndex.from_product([self.static_table_columns, ['Empirical', prob_fun]]) )
        
        for in_, (vnf, label) in enumerate(zip(vnfs, labels)):
            for im_, (df, col) in enumerate(zip([vnf.gas_filter_start, vnf.vnf_filter_start, vnf.flareEpisodes_filter_start], self.static_table_columns)):
                if im_ in [0, 1,]:
                    if in_ in [0,]: dfi = df.iloc[:, :2].values
                    else:
                        if UseStartUpForRegular: dfi = df.iloc[:, 2:].values
                        else: dfi = df.values
                else:
                    if in_ in [0,]: dfi = df.reset_index().loc[df.reset_index()['year_int'] < 3, 'length'].values
                    else:
                        if UseStartUpForRegular: dfi = df.reset_index().loc[df.reset_index()['year_int'] > 2, 'length'].values
                        else: dfi = df['length'].values

                flattened_data = self.rm_nan_fl(dfi)
                y1, y_low1, y_high1, _ = self.boot_correct_probability(flattened_data, return_periods_prob, return_periods_prob, prob_fun=prob_fun)
                all_ = [self.boot_emperical_function(flattened_data, np.quantile, nboot=1000, q=c_) for c_ in return_periods_prob]
                y2, y_low2, y_high2 = [c_[0] for c_ in all_], [c_[1] for c_ in all_], [c_[2] for c_ in all_]
                
                if im_ == 0:
                    out.loc[(label, prob_index), (col, prob_fun)] = [f'{i:.2g} ({j:.2g}, {k:.2g})' for i, j, k in zip(y1, y_low1, y_high1)]
                    out.loc[(label, prob_index), (col, 'Empirical')] = [f'{i:.2g} ({j:.2g}, {k:.2g})' for i, j, k in zip(y2, y_low2, y_high2)]
                else:
                    out.loc[(label, prob_index), (col, prob_fun)] = [f'{i:.2g} ({j:.2g}, {k:.2g})' for i, j, k in zip(y1, y_low1, y_high1)]
                    out.loc[(label, prob_index), (col, 'Empirical')] = [f'{i:.2g} ({j:.2g}, {k:.2g})' for i, j, k in zip(y2, y_low2, y_high2)]
        out.to_csv('./Results/Statics/Scenarios.csv')

    def boot_correct_probability(self, flattened_data:np.ndarray, return_periods_prob:np.ndarray, plotting_positions:np.ndarray, prob_fun:str='LOG'):
        """
        Find the correct functions to boot the prob_fun
        Available prob_fun:
            GUM: Gumbel
            GEV: Generalized Extreme Value
            LOG: Log-Normal
            GPD: Generalized Perato
            EXP: Exponential
            NOR: Normal
        """
        if prob_fun == 'GUM':
            parhat = self.gumbel_by_lmoments(flattened_data)
            return_periods_gumbel = self.gevfun('qgev', return_periods_prob, parhat)
            gumbel_quantiles = self.gevfun('qgev', plotting_positions, parhat)
            yy1, y_low, y_high = self.boot_gevfun(flattened_data, 'qgev', self.gevfun('pgev', gumbel_quantiles, parhat), do_gumbel=True)
        elif prob_fun == 'GEV':
            parhat = self.hybrid_gev_params(flattened_data)
            return_periods_gumbel = self.gevfun('qgev', return_periods_prob, parhat)
            gumbel_quantiles = self.gevfun('qgev', plotting_positions, parhat)
            yy1, y_low, y_high = self.boot_gevfun(flattened_data, 'qgev', self.gevfun('pgev', gumbel_quantiles, parhat), do_gumbel=False)
        elif prob_fun == 'LOG':
            parhat = self.lognormal_by_lmoments(flattened_data)
            return_periods_gumbel = self.lognormalfun('qlogn', return_periods_prob, parhat)
            gumbel_quantiles = self.lognormalfun('qlogn', plotting_positions, parhat)
            yy1, y_low, y_high = self.boot_lognormal(flattened_data, 'qlogn', self.lognormalfun('plogn', gumbel_quantiles, parhat))
        elif prob_fun == 'GPD':
            parhat = self.gpd_by_lmoments(flattened_data)
            return_periods_gumbel = self.gpdfun('qgp', return_periods_prob, parhat)
            gumbel_quantiles = self.gpdfun('qgp', plotting_positions, parhat)
            yy1, y_low, y_high = self.boot_gpd(flattened_data, 'qgp', self.gpdfun('pgp', gumbel_quantiles, parhat))
        elif prob_fun == 'EXP':
            parhat = self.exponential_by_lmoments(flattened_data)
            return_periods_gumbel = self.exponentialfun('qexp', return_periods_prob, parhat)
            gumbel_quantiles = self.exponentialfun('qexp', plotting_positions, parhat)
            yy1, y_low, y_high = self.boot_exponential(flattened_data, 'qexp', self.exponentialfun('pexp', gumbel_quantiles, parhat))
        elif prob_fun == 'NOR':
            parhat = self.normal_by_lmoments(flattened_data)
            return_periods_gumbel = self.normalfun('qnorm', return_periods_prob, parhat)
            gumbel_quantiles = self.normalfun('qnorm', plotting_positions, parhat)
            yy1, y_low, y_high = self.boot_normal(flattened_data, 'qnorm', self.normalfun('pnorm', gumbel_quantiles, parhat))
        else:
            raise ValueError('Wrong prob_fun used in boot_correct_probability function')
        return yy1, y_low, y_high, {'Q':gumbel_quantiles, 'T':return_periods_gumbel}

if __name__ == '__main__':
    pbt = ProbClass()
    print(pbt.boot_emperical_function(np.array([1, 2, 3, 2, 3, 1, 4, 3, 4, 2]), np.mean))
    print(pbt.boot_emperical_function(np.array([1, 2, 3, 2, 3, 1, 4, 3, 4, 2]), np.quantile, q=0.05))
    print(pbt.boot_emperical_function(np.array([1, 2, 3, 2, 3, 1, 4, 3, 4, 2]), np.quantile, q=0.95))
