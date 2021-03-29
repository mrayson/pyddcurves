import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import h5py
from scipy import stats

from . import models 


# Output function
# Convert the trace and observed data to a hdf5 object

def bhm_6_to_h5(outfile, trace, obsdata, use_bhm=True, nparams=6):
    nsamples = len(trace)
    nt = obsdata['n_times']
    
    # Convert the trace to a numpy array
    beta_samples = np.zeros((nparams,nt,nsamples))
    mu_beta_samples = np.zeros((nparams, nsamples))

    params = ['beta_0','beta_1','beta_2','beta_3','beta_4','beta_5']

    for ii in range(nparams):
        beta_samples[ii,...] = trace[params[ii]][0:nsamples,:].T
        #mu_beta_samples[ii,...] = trace['mu_beta_%d'%ii][0:nsamples].T
    
    if use_bhm:
        # Convert the mean samples
        mu_beta_samples[0,...] = trace['mu_beta_0'][0:nsamples].T
        mu_beta_samples[1,...] = trace['mu_beta_1'][0:nsamples].T
        mu_beta_samples[3,...] = trace['mu_beta_3'][0:nsamples].T
        mu_beta_samples[5,...] = trace['mu_beta_5'][0:nsamples].T
        mu_beta_samples[2,...] = trace['mu_beta_mid'][0:nsamples,0].T
        mu_beta_samples[4,...] = trace['mu_beta_mid'][0:nsamples,1].T
    
    ###
    # Save to hdf5
    f = h5py.File(outfile,'w')
    f['beta_samples'] = beta_samples
    if use_bhm:
        f['mu_beta_samples'] = mu_beta_samples

    # Save all of the observed data into its own group
    g = f.create_group('data')

    for kk in obsdata.keys():
        g[kk] = obsdata[kk]

    print('Saved to %s with contents:'%outfile)
    print(f.name)
    for name in f:
        print('\t',name)

    print(g.name)
    for name in g:
        print('\t',name)

    f.close()

def bhm_harmonic_to_h5(outfile, trace, obsdata, omega):
    nparams=6
    nsamples = len(trace)
    nt = obsdata['n_times']
    
    # Convert the trace to a numpy array
    beta_samples = np.zeros((nparams,nt,nsamples))
    aa_samples = np.zeros((nparams, nsamples))


    # Convert the  samples
    beta_samples[0,...] = trace['beta_0'][0:nsamples].T
    beta_samples[1,...] = trace['beta_1'][0:nsamples].T
    beta_samples[3,...] = trace['beta_3'][0:nsamples].T
    beta_samples[5,...] = trace['beta_5'][0:nsamples].T
    beta_samples[2,...] = trace['beta_mid'][0:nsamples,:,0].T
    beta_samples[4,...] = trace['beta_mid'][0:nsamples,:,1].T
    
    # Order the mean of the harmonics
    aa_samples[0,...] = trace['aa'][0:nsamples,0].T
    aa_samples[1,...] = trace['aa'][0:nsamples,1].T
    aa_samples[2,...] = trace['aa_mid'][0:nsamples,0].T
    aa_samples[3,...] = trace['aa'][0:nsamples,2].T
    aa_samples[4,...] = trace['aa_mid'][0:nsamples,1].T
    aa_samples[5,...] = trace['aa'][0:nsamples,3].T
    
    ###
    # Save to hdf5
    f = h5py.File(outfile,'w')
    f['beta_samples'] = beta_samples
    
    # Save the other variables
    f['omega'] = np.array(omega)
    f['aa'] = aa_samples
    f['Aa'] = trace['Aa'][0:nsamples,...]
    f['Ba'] = trace['Ba'][0:nsamples,...]
    
    f['sigma_beta'] = trace['sigma_beta']
    f['sigma_curve'] = trace['sigma_curve']


    # Save all of the observed data into its own group
    g = f.create_group('data')

    for kk in obsdata.keys():
        g[kk] = obsdata[kk]

    print('Saved to %s with contents:'%outfile)
    print(f.name)
    for name in f:
        print('\t',name)

    print(g.name)
    for name in g:
        print('\t',name)

    f.close()

#############
# Beta prediction utilities
#############
def harmonic_beta_np(aa, Aa, Ba, omega, tdays):
    nomega = len(omega)
    amp = np.ones_like(tdays)[...,None] * aa[None,...]
    for ii in range(nomega):
        amp += Aa[...,ii]*np.cos(omega[ii]*tdays[...,None]) + Ba[...,ii]*np.sin(omega[ii]*tdays[...,None])
    
    return amp

    
def truncnorm(lower, upper, mu, sigma):
    return stats.truncnorm(
    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)


def beta_prediction(infile, time, outfile=None, scaled=False):
    """
    Create a prediction of beta's from harmonic MCMC hdf5 file
    
    Set outfile to save as hdf5
    
    Returns: beta_samples
    """
    # Read in the harmonic h5 file
    f = h5py.File(infile,'r')

    # Load the harmonic variables and time data variables
    omega = f['omega'][:] 
    aa = f['aa'][:]
    Aa = f['Aa'][:]
    Ba = f['Ba'][:]
    sigma_beta = f['sigma_beta'][:]

    # Save all of the observed data into its own group
    data = f['/data']
    z_std = np.array(data['z_std'])
    rho_std = np.array(data['rho_std'])
    rho_mu = np.array(data['rho_mu'])
    print(z_std,np.array(z_std))
    f.close()

    # Convert the time to days since 2000-1-1
    dt = time- np.datetime64('2000-01-01')
    tdays = dt.view('int')*1e-9/86400

    # print(aa.shape, Ba.shape)

    nsamples = aa.shape[-1]
    nt = tdays.shape[0]
    # Do a prediction on each beta parameter
    mean_samples = np.zeros((6,nt,nsamples))
    beta_samples = np.zeros((6,nt,nsamples))

    #noise_betas = np.random.normal(scale=sigma_beta[0:nsamples,:], size=(nt,nsamples,6))#.swapaxes(0,1)
    
    for ii in range(6):
        mean_samples[ii,...] = harmonic_beta_np(aa[ii,0:nsamples], Aa[0:nsamples,:,ii], Ba[0:nsamples,:,ii], omega, tdays)
        for nn in range(nsamples):
            
            if ii in [0,1]:
                lower = -1e5
            else:
                lower = 0
                
            beta_samples[ii,:,nn] = truncnorm(lower,1e5, mean_samples[ii,:,nn], sigma_beta[nn,ii]*np.ones((nt,)) ).rvs(nt)
    
    # Scale the results
    if scaled:
        beta_samples[0,...] *= rho_std
        beta_samples[0,...] += rho_mu
        beta_samples[1,...] *= rho_std
        beta_samples[2::,...] *= z_std
        
        z_std = 1.
        rho_std = 1.
        rho_mu = 0.
        
    
    if outfile is not None:
        # Save the results
        fout = h5py.File(outfile,'w')
        fout['beta_samples'] = beta_samples

        # Save all of the observed data into its own group
        g = fout.create_group('data')

        g['z_std'] = z_std
        g['rho_std'] = rho_std
        g['rho_mu'] = rho_mu
        g['time'] = time.view(int)

        fout.close()
        print(outfile)
    
    return beta_samples
    
# Input conversion function
def density_to_obsdict(rho, z2d, time64, ntimeavg, z_std, rho_mu, rho_std):
    """
    Convert the density/depth/time data to a dictionary
    """

    nt,nz = rho.shape
    
    nanidx = ~np.isnan(rho)
    
    # Create an array for the time ubdex
    timeidx = np.arange(0,nt, )[:,np.newaxis] * np.ones_like(rho)

    # Transform the time so that multiple obs. profiles share the same time idx
    timeidx /= ntimeavg
    timevec = np.floor(timeidx[nanidx]).astype(int) #+ 1
    ntidx = timevec.max() +1

    data = {
        'N':nanidx.sum(),
        'n_times':ntidx,
        'rho':(rho[nanidx]-rho_mu)/rho_std,
        'z':z2d[nanidx]/z_std,
        'timeidx':timevec,
        'rho_std':rho_std,
        'rho_mu':rho_mu,
        'z_std':z_std,
        'time':time64[::ntimeavg].view('<i8') # datetime64 to integer
    }
    # Compute the time in days as well
    dt = data['time'].view('<M8[ns]') - np.datetime64('2000-01-01')
    data.update({'tdays':dt.view('int')*1e-9/86400})
    
    return data

def merge_obs_dicts(d1,d2):
    """
    Merge two observed data dictionaries
    """
    N = d1['N']+d2['N']
    nt1 = d1['n_times']
    n_times = d1['n_times'] + d2['n_times']
    
    # Copy the first data set over
    dm = {}
    for kk in d1.keys():
        dm.update({kk:d1[kk]})
        
    # Update a few keys
    dm['N'] = N
    dm['n_times'] = n_times
    
    dm['rho'] = np.hstack([dm['rho'],d2['rho']])
    dm['z'] = np.hstack([dm['z'],d2['z']])
    dm['time'] = np.hstack([dm['time'],d2['time']])
    dm['tdays'] = np.hstack([dm['tdays'],d2['tdays']])

    # Important!! time index starts from the previous data set
    dm['timeidx'] = np.hstack([dm['timeidx'],d2['timeidx']+nt1])
    
    return dm


def plot_density_h5_step(h5file, tstep, samples = None, zmin=None):
    """
    Plot a single time step from a hdf5 file
    """
    
    with h5py.File(h5file,'r') as f:
        
        data = f['/data']
        z_std = data['z_std']
        rho_std = data['rho_std']
        rho_mu = data['rho_mu']
        
        if zmin is None:
            zmin = data['z'][:].min()*z_std
        
        zout = np.linspace(zmin,0,100)
        
        rhomean = np.zeros_like(zout).astype(float)
        
        nparams, nt, nsamples = f['beta_samples'].shape
        
        if samples is None:
            samples = nsamples
        plt.figure(figsize=(5,8))
        beta = f['beta_samples'][:]
        
        for rand_loc in np.random.randint(0, nsamples, samples):
            rhotmp = models.double_tanh([beta[ii,tstep,rand_loc] for ii in range(6)], zout/z_std)

            plt.plot(rhotmp*rho_std+rho_mu, zout, '0.5', lw=0.2, alpha=0.5)

            rhomean+=rhotmp*rho_std+rho_mu

        idx = data['timeidx'][:]==tstep
        plt.plot(data['rho'][idx]*rho_std+rho_mu, data['z'][idx]*z_std,'b.', alpha=0.1)


        rhomean /= samples
        #plt.plot(rhomean, zout, 'k--',) # Mean fit
        #plt.xlim(1020,1027)
        plt.ylim(zmin,0 )
        plt.ylabel('Depth [m]')
        plt.xlabel(r'$\rho$ [kg m$^{-3}$]')
        plt.title(data['time'][tstep].astype('<M8[ns]'))
    

# CSV parsing functions
def convert_time(tt):
    try:
        dt= datetime.strptime(tt, '%Y-%m-%dT%H:%M:%S')
    except:
        dt= datetime.strptime(tt, '%Y-%m-%d %H:%M')
    return dt

def read_density_csv(csvfile):
    # Reads into a dataframe object
    df = pd.read_csv(csvfile, index_col=0, sep=', ', parse_dates=['Time'], date_parser=convert_time)

    # Load the csv data
    depths= np.array([float(ii) for ii in df.columns.values])
    rho_obs_tmp = df[:].values.astype(float)
    time = df.index[:]

    # Clip the top
    rho_obs_2d = rho_obs_tmp[:,:]

    # Remove some nan
    fill_value = 1024.
    rho_obs_2d[np.isnan(rho_obs_2d)] = fill_value
    
    return xr.DataArray(rho_obs_2d,dims=('time', 'depth'),
            coords={'time':time.values,'depth':depths})


