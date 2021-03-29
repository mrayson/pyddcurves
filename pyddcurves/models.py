"""
Parametric functions
"""

import numpy as np
import pymc3 as pm

from theano import shared
from theano import tensor as tt

# parametric density functions
def double_tanh(beta, z): 
    return beta[0] - beta[1]*(np.tanh((z+beta[2])/beta[3])
        + np.tanh((z+beta[4])/beta[5]))

def single_tanh(beta, z): 
    return beta[0] - beta[1]*np.tanh((z+beta[2])/beta[3])

def double_tanh_pm(beta, timeidx, z):
    """
    Double-tanh function that accepts PyMC3 objects instead of numpy arrays
    """
    fac1 = (z+beta[2][timeidx])/beta[3][timeidx]
    fac2 = (z+beta[4][timeidx])/beta[5][timeidx]
    return beta[0][timeidx] - beta[1][timeidx]*( pm.math.tanh(fac1)
                + pm.math.tanh(fac2))

def single_tanh_pm(beta, timeidx, z):
    """
    Single-tanh function that accepts PyMC3 objects instead of numpy arrays
    """
    fac1 = (z+beta[2][timeidx])/beta[3][timeidx]
    return beta[0][timeidx] - beta[1][timeidx]*( pm.math.tanh(fac1))


def harmonic_beta(aa, Aa, Ba, omega, tdays):
    nomega = len(omega)
    amp = tt.ones_like(tdays) * aa
    for ii in range(nomega):
        amp += Aa[ii]*pm.math.cos(omega[ii]*tdays) + Ba[ii]*pm.math.sin(omega[ii]*tdays)
    
    return amp

def harmonic_beta_np(aa, Aa, Ba, omega, tdays):
    nomega = len(omega)
    amp = np.ones_like(tdays)[...,None] * aa[None,...]
    for ii in range(nomega):
        amp += Aa[...,ii]*np.cos(omega[ii]*tdays[...,None]) + Ba[...,ii]*np.sin(omega[ii]*tdays[...,None])
    
    return amp
    
# Main Inference function
def density_bhm_harmonic_dht(data, omega, use_mcmc=True, 
            nchains=2,ncores=2):
    # Full model
    tdays = shared(data['tdays'])
    nparams=6
    nt = data['n_times']
    nomega = len(omega)

    with pm.Model() as rho_model:
        ###
        # Create priors for each of our means
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
        aa = pm.Normal('aa', mu=0, sd=2, shape=4)
        # Order the mid-points
        aa_mid = pm.Normal('aa_mid', mu=np.array([1,2]), sd=np.array([0.25,0.25]), shape=2,
                transform=pm.distributions.transforms.ordered,
                testval=np.array([0.5,1.2]))

        Aa = pm.Normal('Aa', mu=0, sd=1,shape=(nomega,nparams))
        Ba = pm.Normal('Ba', mu=0, sd=1,shape=(nomega,nparams))
        
        
        mu_beta_0 = pm.Deterministic('mu_beta_0',harmonic_beta(aa[0], Aa[:,0], Ba[:,0], omega, tdays))
        mu_beta_1 = pm.Deterministic('mu_beta_1',harmonic_beta(aa[1], Aa[:,1], Ba[:,1], omega, tdays))
        mu_beta_2 = pm.Deterministic('mu_beta_2',harmonic_beta(aa_mid[0], Aa[:,2], Ba[:,2], omega, tdays))
        mu_beta_3 = pm.Deterministic('mu_beta_3',harmonic_beta(aa[2], Aa[:,3], Ba[:,3], omega, tdays))
        mu_beta_4 = pm.Deterministic('mu_beta_4',harmonic_beta(aa_mid[1], Aa[:,4], Ba[:,4], omega, tdays))
        mu_beta_5 = pm.Deterministic('mu_beta_5',harmonic_beta(aa[3], Aa[:,5], Ba[:,5], omega, tdays))
        

        sigma_beta = pm.HalfNormal('sigma_beta', sd=1.0, shape=(nparams,))
        sigma_curve = pm.HalfNormal('sigma_curve', sd=2.0 )

        beta_0 = pm.Normal('beta_0', mu=mu_beta_0, sd=sigma_beta[0], shape=nt)
        beta_1 = BoundedNormal('beta_1', mu=mu_beta_1, sd=sigma_beta[1], shape=nt)
        beta_3 = BoundedNormal('beta_3', mu=mu_beta_3, sd=sigma_beta[3], shape=nt)    
        beta_5 = BoundedNormal('beta_5', mu=mu_beta_5, sd=sigma_beta[5], shape=nt)
        
        # This is a trick for ordering along the last axis of a multivariate distribution
        # (it seems to work...)
        beta_mid = BoundedNormal('beta_mid', mu=tt.stack([mu_beta_2,mu_beta_4]).T, 
                                 sd=tt.stack([sigma_beta[2],sigma_beta[4]]).T, shape=(nt,2),
                                 transform=pm.distributions.transforms.ordered)
        
        beta_s = [beta_0, beta_1, beta_mid[...,0], beta_3, beta_mid[...,1], beta_5, ]

        ###
        # Generate the likelihood function using the deterministic variable as the mean
        mu_x = double_tanh_pm(beta_s, data['timeidx'], data['z'])


        # shape parameter not requires as shape is specified in the priors...
        rho_out = pm.Normal('rho', mu=mu_x, sd=sigma_curve, observed=data['rho'])
        
        ###
        # Inference step
        #trace = pm.sample(500)
        if use_mcmc:
            trace = pm.sample(500, tune=1500, step=pm.NUTS(), cores=ncores, chains=nchains)
        else:
            # Use variational inference
            inference = pm.ADVI()
            approx = pm.fit(n=20000, method=inference)
            trace = approx.sample(draws=500)
            
    
    return trace, rho_model, tdays

# Main Inference function
def density_bhm_dht(data, use_bhm=True, use_mcmc=True, 
            nchains=2,ncores=2):
    # Full model
    with pm.Model() as rho_model:
        ###
        # Create priors for each of our means
        BoundedNormal = pm.Bound(pm.Normal, lower=0.0)
        nparams=6
        nt = data['n_times']

        # Priors on the means (assume everything is scaled to ~O(1) )
        if use_bhm:
            mu_beta_0 = pm.Normal('mu_beta_0', mu=1.0, sd=1.0,)
            mu_beta_1 = BoundedNormal('mu_beta_1', mu=1., sd=1.,)
            mu_beta_3 = BoundedNormal('mu_beta_3', mu=1, sd=1.0,)
            mu_beta_5 = BoundedNormal('mu_beta_5', mu=0, sd=1.0,)
            mu_beta_mid = BoundedNormal('mu_beta_mid', mu=1., sd=1.0, shape=2,
                        transform=pm.distributions.transforms.ordered,
                          testval=np.array([0.1,0.2]))

            sigma_beta = pm.HalfNormal('sigma_beta', sd=1.0, shape=(nparams,))

        else:
            mu_beta_0 = 1
            mu_beta_1 = 1
            mu_beta_3 = 1
            mu_beta_5 = 1.
            mu_beta_6 = 0.
            mu_beta_mid = [0,1]
            sigma_beta=[1,1,1,1,1,1,1]

        sigma_curve = pm.HalfNormal('sigma_curve', sd=2.0 )

        beta_0 = pm.Normal('beta_0', mu=mu_beta_0, sd=sigma_beta[0], shape=nt)
        beta_1 = BoundedNormal('beta_1', mu=mu_beta_1, sd=sigma_beta[1], shape=nt)
        beta_3 = BoundedNormal('beta_3', mu=mu_beta_3, sd=sigma_beta[3], shape=nt)    
        beta_2 = BoundedNormal('beta_2', mu=mu_beta_mid[0], sd=sigma_beta[2], shape=nt)
        beta_4 = BoundedNormal('beta_4', mu=mu_beta_mid[1], sd=sigma_beta[4], shape=nt)
        beta_5 = BoundedNormal('beta_5', mu=mu_beta_5, sd=sigma_beta[5], shape=nt)



        beta_s = [beta_0, beta_1, beta_2, beta_3, beta_4, beta_5, ]


        ###
        # Generate the likelihood function using the deterministic variable as the mean
        mu_x = double_tanh_pm(beta_s, data['timeidx'], data['z'])

        # shape parameter not requires as shape is specified in the priors...
        rho_out = pm.Normal('rho', mu=mu_x, sd=sigma_curve, observed=data['rho'])
        
        ###
        # Inference step
        #trace = pm.sample(500)
        if use_mcmc:
            trace = pm.sample(2000, step=pm.NUTS(), cores=ncores, chains=nchains)

        else:
            # Use variational inference
            inference = pm.ADVI()
            approx = pm.fit(n=30000, method=inference)
            trace = approx.sample(draws=2000)
            
    
    return trace, rho_model

              