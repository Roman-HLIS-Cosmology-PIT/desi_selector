import opencosmo as oc
import numpy as np
import pandas as pd
from scipy import interpolate
from astropy import table
from pathlib import Path
from astropy.cosmology import LambdaCDM
from scipy.interpolate import interp1d
from diffsky.experimental import lc_utils
from diffsky.data_loaders.hacc_utils import lightcone_utils
import jax.random as jran
import treecorr as tc
import healpy as hp


class DesiPhotSelector:
    

    # Cosmology used for the Diffsky sims 
    OMEGA_C = 0.26067
    OMEGA_B = 0.049
    h = 0.6766
    N_S = 0.9665
    SIGMA8 = 0.8102
    RANDOM_SEED = 42
    cosmo = LambdaCDM(H0=h * 100, Om0=OMEGA_C + OMEGA_B, Ode0=1 - (OMEGA_C + OMEGA_B))
    
    def __init__(self, 
                 desi_tracer,
                 path_sim,
                 calibration_version,
                 z_range,
                 random_seed: int = RANDOM_SEED,
                 ):

        self.desi_tracer = desi_tracer
        self.path_sim = path_sim
        self.calibration_version = calibration_version
        self.z_range = z_range
        self.random_seed = int(random_seed)

    

        
        path_sim_data = Path(f"{self.path_sim}/{calibration_version}")
        list_sim_data = list(f for f in path_sim_data.glob("*.hdf5") if f.stem.startswith("lc_cores"))
        dataset = oc.open(list_sim_data)

        # Calculate the total area the mocks span on the sky 
        dataset = oc.open(list_sim_data)
        pixels = dataset.region.pixels
        nside = dataset.region.nside
        sim_area = len(pixels)*hp.nside2pixarea(nside, degrees=True)
        self.sim_area = sim_area

        print(f'The total area spanned by the mocks in {self.calibration_version} is: {self.sim_area}')


        if self.desi_tracer == 'bgs':
            columns = ['ra', 'dec', 'redshift_true', 'lsst_r', 'lc_patch']

        elif self.desi_tracer == 'elg':
            columns = ['ra', 'dec', 'redshift_true', 'lsst_g', 'lsst_r', 'lsst_z', 'lc_patch']

        
        dataset = dataset.select(columns)
        dataset = dataset.with_redshift_range(self.z_range[0], self.z_range[1])
        sim_cat = dataset.data.to_pandas()
        sim_cat['distance'] = DesiPhotSelector.cosmo.comoving_distance(sim_cat['redshift_true']).value
        self.sim_cat = sim_cat


    def produce_desi_mock(self):

        
        if self.desi_tracer == 'bgs':
            
            R_MAG_LIMIT = 19.5
            mask_photometry = self.sim_cat['lsst_r'] <  R_MAG_LIMIT
        
        elif self.desi_tracer == 'elg':
            
            COLOR_CUT1_SLOPE = 0.50
            COLOR_CUT1_INTERCEPT = 0.1
            COLOR_CUT2_SLOPE = -1.20
            COLOR_CUT2_INTERCEPT = 1.3
            G_FIBER_LIMIT = 24.1 
            G_FIBER_MAG_OFFSET = 0.65
            G_MAG_LIMIT = G_FIBER_LIMIT - G_FIBER_MAG_OFFSET # Based on average difference between fiber and mag 
            mask_color = np.logical_and((self.sim_cat['lsst_g'] - self.sim_cat['lsst_r']) < COLOR_CUT1_SLOPE*(self.sim_cat['lsst_r'] - self.sim_cat['lsst_z']) + COLOR_CUT1_INTERCEPT, 
                                            (self.sim_cat['lsst_g'] - self.sim_cat['lsst_r']) < COLOR_CUT2_SLOPE*(self.sim_cat['lsst_r'] - self.sim_cat['lsst_z']) + COLOR_CUT2_INTERCEPT)
            mask_g_mag = self.sim_cat['lsst_g'] < G_MAG_LIMIT
            mask_photometry = np.logical_and(mask_color, mask_g_mag)


        mock_cat = self.sim_cat[mask_photometry]
        
        return mock_cat

    
    def produce_desi_rands(self, mock_cat=None):

        sim_patches = np.unique(self.sim_cat['lc_patch'])

        RAND_TO_DATA_RATIO = 10
        npatches = len(sim_patches)
        ntot = int(len(mock_cat)* RAND_TO_DATA_RATIO / npatches)
        lc_path = '/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/lc_metadata/lc_cores-decomposition.txt'
        lc_cores_decomp = lightcone_utils.read_lc_ra_dec_patch_decomposition(lc_path)[0]
        theta_low = lc_cores_decomp[:,1]
        theta_high = lc_cores_decomp[:,2]
        phi_low = lc_cores_decomp[:,3]
        phi_high = lc_cores_decomp[:,4]
    
    
        ra_min, dec_max = lightcone_utils.get_ra_dec_from_theta_phi(theta_low, phi_low)
        ra_max, dec_min = lightcone_utils.get_ra_dec_from_theta_phi(theta_high, phi_high)
        ran_key = jran.PRNGKey(self.random_seed)
    
        
        list_ra = []
        list_dec = []
        
        for patch in sim_patches:
            
            ra_loop, dec_loop = lc_utils.mc_lightcone_random_ra_dec(ran_key=ran_key, npts=ntot, ra_min=ra_min[patch],
            ra_max=ra_max[patch], dec_min=dec_min[patch], dec_max=dec_max[patch])
    
            list_ra.append(ra_loop)
            list_dec.append(dec_loop)
                                        
            
        rand_ra = np.concatenate(list_ra)
        rand_dec = np.concatenate(list_dec)
            
        list_rand_cols = np.column_stack([rand_ra, rand_dec])
        rand_cat = pd.DataFrame(list_rand_cols, columns=['ra', 'dec'])
        rand_cat = rand_cat.reset_index(drop=True) 
        mock_cat_temp = mock_cat.reset_index(drop=True).sample(
            len(rand_cat), replace=True, random_state=self.random_seed
        )
        rand_cat['distance'] = mock_cat_temp['distance'].to_numpy()
        rand_cat['redshift_true'] = mock_cat_temp['redshift_true'].to_numpy()
    
        return rand_cat

   
    def measure_autocorr(self, mock_cat, rand_cat, z_range_clustering=[0.8, 1.1], min_sep=1, max_sep=200, nbins=100):
    

        
        mask_mock_z_cut = np.logical_and(mock_cat['redshift_true'] > z_range_clustering[0], mock_cat['redshift_true'] < z_range_clustering[1]) 
        mock_cat = mock_cat[mask_mock_z_cut]
    
        mask_rand_z_cut = np.logical_and(rand_cat['redshift_true'] > z_range_clustering[0], rand_cat['redshift_true'] < z_range_clustering[1]) 
        rand_cat = rand_cat[mask_rand_z_cut]
    
        ra = mock_cat['ra']
        dec = mock_cat['dec']
        s = mock_cat['distance']
        
        ra_rand = rand_cat['ra']
        dec_rand = rand_cat['dec']
        s_rand = rand_cat['distance']
    
        print('The ratio of data to randoms is:', len(ra_rand)/ len(ra))
    
        # calculate the correlation using s
    
        dg_to_r = np.pi / 180
        
        tc_cat_s = tc.Catalog(ra=ra*dg_to_r, dec=dec*dg_to_r, r=s, ra_units='radians', dec_units='radians')
        tc_rnd_s = tc.Catalog(ra=ra_rand*dg_to_r, dec=dec_rand*dg_to_r, r=s_rand, ra_units='radians', dec_units='radians')
        
        
        dd_s = tc.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        dr_s = tc.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        rr_s = tc.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins)
        
        dd_s.process(tc_cat_s, metric='Euclidean')
        dr_s.process(tc_cat_s, tc_rnd_s, metric='Euclidean')
        rr_s.process(tc_rnd_s, metric='Euclidean')
        xi_s, var_xi = dd_s.calculateXi(rr=rr_s, dr=dr_s)
        xi_s_naive, var_xi_naive = dd_s.calculateXi(rr=rr_s)
        sep = DesiPhotSelector.cosmo.h * np.exp(dd_s.meanlogr)
    
        return {'sep': sep,
               'xi': xi_s,
                'var_xi': var_xi,
               'dd': dd_s,
               'dr': dr_s,
               "rr": rr_s,
               "xi_naive": xi_s_naive,
               'var_xi_naive': var_xi_naive}
        

        