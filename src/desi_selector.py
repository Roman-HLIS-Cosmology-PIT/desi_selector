import os
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


class DesiSelector:
    

    # Cosmology used for the Diffsky sims 
    OMEGA_C = 0.26067
    OMEGA_B = 0.049
    h = 0.6766
    N_S = 0.9665
    SIGMA8 = 0.8102
    RANDOM_SEED = 42
    cosmo = LambdaCDM(H0=h * 100, Om0=OMEGA_C + OMEGA_B, Ode0=1 - (OMEGA_C + OMEGA_B))
    
    def __init__(self, 
                 desi_tracer: str,
                 path_desi_tracer: dict[str, str],
                 path_sim: str,
                 calibration_version: str,
                 z_range: list[int], 
                 z_grid_points: int,
                 select_biggest: bool=True,
                 synth_cores: bool=False,
                 reload_oc: bool=True,
                 threshold_col: str=None,
                 sigma_dex: float=None,
                 weight: float=None,
                 cache_root: str | Path | None = None,
                 random_seed: int = RANDOM_SEED,
                 ):

        self.desi_tracer = desi_tracer
        self.path_desi_tracer = path_desi_tracer
        self.path_sim = path_sim
        self.calibration_version = calibration_version
        self.z_range = z_range
        self.z_grid_points = z_grid_points
        self.select_biggest = select_biggest
        self.synth_cores = synth_cores
        self.reload_oc = reload_oc
        self.threshold_col = threshold_col
        self.sigma_dex = sigma_dex
        self.weight = weight
        self.random_seed = int(random_seed)
        default_cache = os.environ.get(
            "DESI_SELECTOR_CACHE_ROOT",
            "/pscratch/sd/l/lhior/nz_clustering_e2e/desi_cache",
        )
        self.cache_root = Path(cache_root if cache_root is not None else default_cache)
        self._list_sim_data = None

        
        if self.threshold_col is None:
            
            # column to use for threshold calculation
            if self.desi_tracer == 'bgs':
                self.threshold_col = 'log_peak_sub_halo_mass'
    
            elif self.desi_tracer == 'lrg':
                self.threshold_col = 'log_peak_sub_halo_mass'
    
            elif self.desi_tracer == 'elg':
                self.threshold_col = 'log_sfr'
    
            elif self.desi_tracer == 'qso':
                self.threshold_col = 'black_hole_mass'

        
        
        path_sim_data = Path(f"{self.path_sim}/{self.calibration_version}")
        list_sim_data = list(f for f in path_sim_data.glob("*.hdf5") if f.stem.startswith("lc_cores"))
        self._list_sim_data = list_sim_data

        # Calculate the total area the mocks span on the sky 
        dataset = oc.open(list_sim_data, synth_cores=self.synth_cores)
        pixels = dataset.region.pixels
        nside = dataset.region.nside
        sim_area = len(pixels)*hp.nside2pixarea(nside, degrees=True)
        self.sim_area = sim_area

        print(f'The total area spanned by the mocks in {self.calibration_version} is: {self.sim_area}')
        
        
        if self.desi_tracer == 'bgs':
            columns = ['ra', 'dec', 'redshift_true', 'lsst_r', 'logsm_obs', 'logmp_obs', 'logmp_obs_host', 'central', 'lc_patch']

        elif self.desi_tracer == 'lrg':
            columns = ['ra', 'dec', 'redshift_true', 'logmp_obs', 'logmp_obs_host', 'central', 'lc_patch']

        elif self.desi_tracer == 'elg':
            columns = ['ra', 'dec', 'redshift_true', 'logsm_obs', 'logssfr_obs', 'lsst_g', 'lsst_r', 'lsst_z', 'logmp_obs', 'logmp_obs_host', 'central', 'lc_patch']

        elif self.desi_tracer == 'qso':
            columns = ['ra', 'dec', 'redshift_true', 'black_hole_mass', 'logmp_obs', 'logmp_obs_host', 'central', 'lc_patch']
              


        if self.reload_oc:
            z_low, z_high = self.z_range[0], self.z_range[1]
            dataset = dataset.select(columns)
            # lc_cores HDF5 uses redshift_true; with_redshift_range expects redshift
            dataset = dataset.filter(
                oc.col("redshift_true") > z_low,
                oc.col("redshift_true") < z_high,
            )
            sim_cat = dataset.get_data('pandas')
            sim_cat['distance'] = DesiSelector.cosmo.comoving_distance(sim_cat['redshift_true']).value

            sim_cat_filename = self._sim_cat_path()
            sim_cat_filename.parent.mkdir(parents=True, exist_ok=True)
            sim_cat.to_parquet(sim_cat_filename)
        else:
            sim_cat_filename = self._resolve_sim_cat_path()
            if sim_cat_filename.exists():
                sim_cat = pd.read_parquet(sim_cat_filename)
            else:
                raise FileNotFoundError(
                    f"Cached sim catalog not found at {sim_cat_filename}; "
                    f"set reload_oc: true for tracer {self.desi_tracer}"
                )


        # Columns to abundance match with
        if self.desi_tracer == 'bgs':
            sim_cat.rename(columns={'logmp_obs': 'log_peak_sub_halo_mass'}, inplace=True)
            sim_cat.rename(columns={'logsm_obs': 'log_stellar_mass'}, inplace=True)

            if self.sigma_dex is not None:
                rng = self._noise_rng()
                noise = rng.normal(loc=0, scale=self.sigma_dex, size=len(sim_cat))
                sim_cat['log_peak_sub_halo_mass_noisy'] = sim_cat['log_peak_sub_halo_mass'] + noise
        
        if self.desi_tracer == 'lrg':
            sim_cat.rename(columns={'logmp_obs': 'log_peak_sub_halo_mass'}, inplace=True)

            if self.sigma_dex is not None:
                rng = self._noise_rng()
                noise = rng.normal(loc=0, scale=self.sigma_dex, size=len(sim_cat))
                sim_cat['log_peak_sub_halo_mass_noisy'] = sim_cat['log_peak_sub_halo_mass'] + noise

        if self.desi_tracer == 'elg':
            sim_cat['log_sfr'] = sim_cat['logsm_obs'] + sim_cat['logssfr_obs']

            if self.weight is not None:
                sim_cat['log_sfr_times_wmass'] = sim_cat['log_sfr'] + self.weight*sim_cat['logsm_obs']

            if self.sigma_dex is not None:
                rng = self._noise_rng()
                noise = rng.normal(loc=0, scale=self.sigma_dex, size=len(sim_cat))
                sim_cat['log_sfr_noisy'] = sim_cat['log_sfr'] + noise
            else:
                sim_cat['log_sfr_noisy'] = sim_cat['log_sfr']

        if self.desi_tracer == 'qso':
            
            if self.sigma_dex is not None:

                rng = self._noise_rng()
                sigma_nat = self.sigma_dex * np.log(10)
                noise = rng.lognormal(mean=0, sigma=sigma_nat, size=len(sim_cat))
                sim_cat['black_hole_mass_noisy'] = sim_cat['black_hole_mass'] * noise    

        self.sim_cat = sim_cat

    def _noise_rng(self) -> np.random.Generator:
        """Fixed-seed RNG for tracer noise (reproducible across runs)."""
        return np.random.default_rng(self.random_seed)

    def _rng(self) -> np.random.Generator:
        """Fixed-seed NumPy Generator used for random catalogs."""
        return np.random.default_rng(self.random_seed)

    def _sim_cat_path(self) -> Path:
        z_low, z_high = self.z_range[0], self.z_range[1]
        return (
            self.cache_root
            / "sim_data"
            / self.desi_tracer
            / f"{self.calibration_version}_{self.sim_area}_z{z_low}_{z_high}.parquet"
        )

    def _resolve_sim_cat_path(self) -> Path:
        """Resolve sim catalog cache path, with legacy threshold_col fallback."""
        primary = self._sim_cat_path()
        if primary.exists():
            return primary
        legacy = (
            self.cache_root
            / "sim_data"
            / self.desi_tracer
            / f"{self.calibration_version}_{self.sim_area}_{self.threshold_col}.parquet"
        )
        if legacy.exists():
            print(f"Using legacy sim cache: {legacy.name}")
            return legacy
        return primary

    def _z_center_path(self) -> Path:
        return (
            self.cache_root
            / "selection_z_centers"
            / self.desi_tracer
            / f"{self.z_grid_points}_centers.npy"
        )

    def _threshold_path(self) -> Path:
        return (
            self.cache_root
            / "selection_thresholds"
            / self.desi_tracer
            / f"{self.threshold_col}_thres.npy"
        )

    def _lc_metadata_path(self) -> Path:
        return self.cache_root / "lc_metadata" / "lc_cores-decomposition.txt"
    
    def rebin_desi_tracer(self):
        
        # load the desi tracer data we are abundance matching to and get bin edges and centers

        # DESI footprint areas
        NORTH_AREA = 4400
        SOUTH_DECAL_AREA = 8500
        SOUTH_DES_AREA = 1100
        TOTAL_DESI_AREA = 14000
        
        
        if self.desi_tracer == 'bgs':
            tracer_data = np.loadtxt(self.path_desi_tracer['north'])
            z_bin_center = tracer_data[:,0]
            z_bin_min = tracer_data[:,1]
            z_bin_max = tracer_data[:,2]

        
        elif self.desi_tracer == 'lrg':
            tracer_data = pd.read_csv(self.path_desi_tracer['path'], index_col=False)
            # ignore the fist redshift bin since it is negative and does not contain galaxies
            z_bin_min = tracer_data['zmin'].to_numpy()[1:] 
            z_bin_max = tracer_data['zmax'].to_numpy()[1:]
            z_bin_center = (z_bin_min + z_bin_max) / 2

        
        elif self.desi_tracer == 'elg':
            tracer_data = table.Table.read(self.path_desi_tracer['path'],  format='ascii.ecsv')
            z_bin_min = tracer_data['ZMIN']
            z_bin_max = tracer_data['ZMAX']
            z_bin_center = (z_bin_min + z_bin_max) / 2 

        
        elif self.desi_tracer == 'qso':
            tracer_data = table.Table.read(self.path_desi_tracer['path'],  format='ascii.ecsv')
            z_mask = np.logical_and(tracer_data['z'] > self.z_range[0], tracer_data['z'] < self.z_range[1])
            z_bin_center = tracer_data['z'][z_mask]
            z_bin_min = z_bin_center - 0.050/2
            z_bin_max = z_bin_center + 0.050/2

        
        # get the desi tracer number/deg2 data we want to match to 
        if self.desi_tracer == 'bgs':
            
            EFF_AREA_NORTH = 5108.0437685335755
            EFF_AREA_SOUTH = 2071.9122137829345
            FRAC_AREA_NORTH = NORTH_AREA / TOTAL_DESI_AREA
            FRAC_AREA_SOUTH = (SOUTH_DECAL_AREA + SOUTH_DES_AREA) / TOTAL_DESI_AREA
            
            path_north = self.path_desi_tracer['north']
            path_south = self.path_desi_tracer['south']
            data_north = np.loadtxt(path_north)
            data_south = np.loadtxt(path_south)
            
            n_bin_north = data_north[:,4]
            n_bin_south = data_south[:,4]
            nz_north = n_bin_north / EFF_AREA_NORTH
            nz_south = n_bin_south / EFF_AREA_SOUTH
            nz_avg = nz_north*FRAC_AREA_NORTH + nz_south*FRAC_AREA_SOUTH

            fiber_rate_bgs = 0.636
            num_goodz_bgs = 300043
            area_bgs = 7473
            correct_norm_bgs = (num_goodz_bgs / area_bgs) / fiber_rate_bgs
            wrong_norm_bgs = np.sum(nz_avg)
            nz_avg = nz_avg*(correct_norm_bgs/wrong_norm_bgs)

        
        elif self.desi_tracer == 'lrg':
            # ignore first bin since there are no galaxies
            nz_avg = tracer_data['n_desi_lrg'].to_numpy()[1:]

        
        elif self.desi_tracer == 'elg':
 
            lop_north = tracer_data['ELG_LOP_NORTH']
            lop_south_decal = tracer_data['ELG_LOP_SOUTH_DECALS']
            lop_south_des = tracer_data['ELG_LOP_SOUTH_DES']
            nz_avg = (lop_north * NORTH_AREA + lop_south_decal * SOUTH_DECAL_AREA  + lop_south_des * SOUTH_DES_AREA)/(TOTAL_DESI_AREA)
            fiber_rate_elg = 0.69 
            nz_avg = nz_avg / fiber_rate_elg 


        elif self.desi_tracer == 'qso':
            
            z_mask = np.logical_and(tracer_data['z'] > self.z_range[0], tracer_data['z'] < self.z_range[1])
            nz_north = tracer_data['n_z_north'][z_mask]
            nz_south = tracer_data['n_z_south'][z_mask]
            nz_avg = (nz_north*NORTH_AREA + nz_south*(SOUTH_DECAL_AREA + SOUTH_DES_AREA)) / TOTAL_DESI_AREA 


        print(f'The redshift range of the tracer being emulated is {np.min(z_bin_min)} - {np.max(z_bin_max)}')
        
        repeat_n = int((self.z_grid_points-1)/len(z_bin_center))
        new_z_bin_min = np.linspace(np.min(z_bin_min), np.max(z_bin_max),  self.z_grid_points)[:-1]
        new_z_bin_max = np.linspace(np.min(z_bin_min), np.max(z_bin_max),  self.z_grid_points)[1:]
        new_z_center = (new_z_bin_max + new_z_bin_min) / 2
        
        z_bin_center_pad = np.insert(z_bin_center, 0, z_bin_center[0] - (z_bin_center[1] - z_bin_center[0])/2)
        z_bin_center_pad = np.append(z_bin_center_pad , z_bin_center[-1] + (z_bin_center[-1] - z_bin_center[-2])/2)
        
        nz_avg_pad = np.insert(nz_avg, 0, nz_avg[0])
        nz_avg_pad = np.append(nz_avg_pad, nz_avg[-1])

        interp_nz_avg_func = interp1d(z_bin_center_pad, nz_avg_pad, fill_value=0, bounds_error=False)
        interp_nz_avg = interp_nz_avg_func(new_z_center) / repeat_n

        zgrid = np.linspace(np.min(z_bin_min), np.max(z_bin_max), self.z_grid_points)
        values, edges = np.histogram(self.sim_cat['redshift_true'], bins=zgrid)
        values_sim = values / self.sim_area
        z_frac = interp_nz_avg / values_sim
        z_frac = np.nan_to_num(z_frac, nan=0.0, posinf=0.0, neginf=0.0)
        z_frac = np.minimum(z_frac, np.ones(len(z_frac))*0.99)

        print(f'The max value in z_frac is {np.max(z_frac)}')

        self.new_z_bin_min = new_z_bin_min
        self.new_z_center = new_z_center
        self.new_z_bin_max = new_z_bin_max
        self.z_frac = z_frac
        self.nz_avg = nz_avg

        
        # save the new z center
        path_z_center = self._z_center_path()
        path_z_center.parent.mkdir(parents=True, exist_ok=True)
        np.save(path_z_center, self.new_z_center)
        
    
    def generate_threshold(self):
        
        thres_list = []
        
        for i in range(len(self.new_z_center)):
            
            this_zmin = self.new_z_bin_min[i]
            this_zmax = self.new_z_bin_max[i]
            this_cat = self.sim_cat[np.logical_and(self.sim_cat['redshift_true']>this_zmin, self.sim_cat['redshift_true']<this_zmax)]
        
            if len(this_cat) == 0:
                
                print(f"Empty bin: zmin={this_zmin}, zmax={this_zmax}")
                this_thres = 10**40 # set threshold to high value to not select anything

            else:

                if self.select_biggest:
                
                    this_thres = np.percentile(a = this_cat[self.threshold_col], q = 100-self.z_frac[i]*100)
    
                else:
                    
                    this_thres = np.percentile(a = this_cat[self.threshold_col], q = self.z_frac[i]*100)
        
            thres_list.append(this_thres)
                        
                     
        self.thres_list = thres_list


    def produce_desi_mock(self):
      
        thres_of_z = interpolate.interp1d(self.new_z_center, self.thres_list,  fill_value="extrapolate", bounds_error=False)
        threshold_all = thres_of_z(self.sim_cat['redshift_true'])

        # save the threshold values
        path_threshold = self._threshold_path()
        path_threshold.parent.mkdir(parents=True, exist_ok=True)
        np.save(path_threshold, threshold_all)

        if self.select_biggest:
            
            mask_abundance = self.sim_cat[self.threshold_col] > threshold_all

        else:

            mask_abundance = self.sim_cat[self.threshold_col] < threshold_all
        
        mock_cat = self.sim_cat[mask_abundance]

        return mock_cat

    def _patch_ra_dec_bounds(self) -> dict[int, tuple[float, float, float, float]]:
        bounds: dict[int, tuple[float, float, float, float]] = {}
        for patch in np.unique(self.sim_cat["lc_patch"]):
            patch_data = self.sim_cat[self.sim_cat["lc_patch"] == patch]
            bounds[int(patch)] = (
                float(patch_data["ra"].min()),
                float(patch_data["ra"].max()),
                float(patch_data["dec"].min()),
                float(patch_data["dec"].max()),
            )
        return bounds

    def produce_desi_rands(self, mock_cat=None):
        
        sim_patches = np.unique(self.sim_cat['lc_patch'])

        RAND_TO_DATA_RATIO = 10
        npatches = len(sim_patches)
        ntot = int(len(mock_cat)* RAND_TO_DATA_RATIO / npatches)
        lc_path = self._lc_metadata_path()
        if lc_path.exists():
            lc_cores_decomp = lightcone_utils.read_lc_ra_dec_patch_decomposition(str(lc_path))[0]
            theta_low = lc_cores_decomp[:,1]
            theta_high = lc_cores_decomp[:,2]
            phi_low = lc_cores_decomp[:,3]
            phi_high = lc_cores_decomp[:,4]
            ra_min, dec_max = lightcone_utils.get_ra_dec_from_theta_phi(theta_low, phi_low)
            ra_max, dec_min = lightcone_utils.get_ra_dec_from_theta_phi(theta_high, phi_high)
            use_theta_phi = True
        else:
            patch_bounds = self._patch_ra_dec_bounds()
            use_theta_phi = False
        ran_key = jran.PRNGKey(self.random_seed)
    
        
        list_ra = []
        list_dec = []
        
        for patch in sim_patches:
            patch_idx = int(patch)
            if use_theta_phi:
                ra_loop, dec_loop = lc_utils.mc_lightcone_random_ra_dec(
                    ran_key=ran_key,
                    npts=ntot,
                    ra_min=ra_min[patch_idx],
                    ra_max=ra_max[patch_idx],
                    dec_min=dec_min[patch_idx],
                    dec_max=dec_max[patch_idx],
                )
            else:
                ra_lo, ra_hi, dec_lo, dec_hi = patch_bounds[patch_idx]
                ra_loop, dec_loop = lc_utils.mc_lightcone_random_ra_dec(
                    ran_key=ran_key,
                    npts=ntot,
                    ra_min=ra_lo,
                    ra_max=ra_hi,
                    dec_min=dec_lo,
                    dec_max=dec_hi,
                )
    
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
        sep = DesiSelector.cosmo.h * np.exp(dd_s.meanlogr)
    
        return {'sep': sep,
               'xi': xi_s,
                'var_xi': var_xi,
               'dd': dd_s,
               'dr': dr_s,
               "rr": rr_s,
               "xi_naive": xi_s_naive,
               'var_xi_naive': var_xi_naive}
    
    

class DesiSelectorE2E:
    

    # Cosmology used for the Diffsky sims 
    OMEGA_C = 0.26067
    OMEGA_B = 0.049
    h = 0.6766
    N_S = 0.9665
    SIGMA8 = 0.8102
    cosmo = LambdaCDM(H0=h * 100, Om0=OMEGA_C + OMEGA_B, Ode0=1 - (OMEGA_C + OMEGA_B))
    
    def __init__(self, 
                 desi_tracer: str,
                 path_desi_tracer: dict[str, str],
                 path_sim: str,
                 sim_area: float,
                 z_range: list[int], 
                 z_grid_points: int,
                 select_biggest: bool=True,
                 threshold_col: str=None,
                 ):

        self.desi_tracer = desi_tracer
        self.path_desi_tracer = path_desi_tracer
        self.path_sim = path_sim
        self.sim_area = sim_area
        self.z_range = z_range
        self.z_grid_points = z_grid_points
        self.select_biggest = select_biggest
        self.threshold_col = threshold_col


        
        if self.threshold_col is None:
            
            # column to use for threshold calculation
            if self.desi_tracer == 'bgs':
                self.threshold_col = 'log_peak_sub_halo_mass'
    
            elif self.desi_tracer == 'lrg':
                self.threshold_col = 'log_peak_sub_halo_mass'
    
            elif self.desi_tracer == 'elg':
                self.threshold_col = 'log_sfr'
    
            elif self.desi_tracer == 'qso':
                self.threshold_col = 'black_hole_mass'
 

        
        sim_cat = pd.read_parquet(self.path_sim)
        columns = [self.threshold_col, 'redshift_true', 'ra', 'dec', 'lc_patch']
        sim_cat = sim_cat[columns]
        sim_cat['distance'] = DesiSelector.cosmo.comoving_distance(sim_cat['redshift_true']).value
        mask_redshift = np.logical_and(sim_cat['redshift_true'] > self.z_range[0], sim_cat['redshift_true'] < self.z_range[1])
        self.sim_cat = sim_cat[mask_redshift]


    
    def rebin_desi_tracer(self):
        
        # load the desi tracer data we are abundance matching to and get bin edges and centers

        # DESI footprint areas
        NORTH_AREA = 4400
        SOUTH_DECAL_AREA = 8500
        SOUTH_DES_AREA = 1100
        TOTAL_DESI_AREA = 14000
        
        
        if self.desi_tracer == 'bgs':
            tracer_data = np.loadtxt(self.path_desi_tracer['north'])
            z_bin_center = tracer_data[:,0]
            z_bin_min = tracer_data[:,1]
            z_bin_max = tracer_data[:,2]

        
        elif self.desi_tracer == 'lrg':
            tracer_data = pd.read_csv(self.path_desi_tracer['path'], index_col=False)
            # ignore the fist redshift bin since it is negative and does not contain galaxies
            z_bin_min = tracer_data['zmin'].to_numpy()[1:] 
            z_bin_max = tracer_data['zmax'].to_numpy()[1:]
            z_bin_center = (z_bin_min + z_bin_max) / 2

        
        elif self.desi_tracer == 'elg':
            tracer_data = table.Table.read(self.path_desi_tracer['path'],  format='ascii.ecsv')
            z_bin_min = tracer_data['ZMIN']
            z_bin_max = tracer_data['ZMAX']
            z_bin_center = (z_bin_min + z_bin_max) / 2 

        
        elif self.desi_tracer == 'qso':
            tracer_data = table.Table.read(self.path_desi_tracer['path'],  format='ascii.ecsv')
            z_mask = np.logical_and(tracer_data['z'] > self.z_range[0], tracer_data['z'] < self.z_range[1])
            z_bin_center = tracer_data['z'][z_mask]
            z_bin_min = z_bin_center - 0.050/2
            z_bin_max = z_bin_center + 0.050/2

        
        # get the desi tracer number/deg2 data we want to match to 
        if self.desi_tracer == 'bgs':
            
            EFF_AREA_NORTH = 5108.0437685335755
            EFF_AREA_SOUTH = 2071.9122137829345
            FRAC_AREA_NORTH = NORTH_AREA / TOTAL_DESI_AREA
            FRAC_AREA_SOUTH = (SOUTH_DECAL_AREA + SOUTH_DES_AREA) / TOTAL_DESI_AREA
            
            path_north = self.path_desi_tracer['north']
            path_south = self.path_desi_tracer['south']
            data_north = np.loadtxt(path_north)
            data_south = np.loadtxt(path_south)
            
            n_bin_north = data_north[:,4]
            n_bin_south = data_south[:,4]
            nz_north = n_bin_north / EFF_AREA_NORTH
            nz_south = n_bin_south / EFF_AREA_SOUTH
            nz_avg = nz_north*FRAC_AREA_NORTH + nz_south*FRAC_AREA_SOUTH

            fiber_rate_bgs = 0.636
            num_goodz_bgs = 300043
            area_bgs = 7473
            correct_norm_bgs = (num_goodz_bgs / area_bgs) / fiber_rate_bgs
            wrong_norm_bgs = np.sum(nz_avg)
            nz_avg = nz_avg*(correct_norm_bgs/wrong_norm_bgs)

        
        elif self.desi_tracer == 'lrg':
            # ignore first bin since there are no galaxies
            nz_avg = tracer_data['n_desi_lrg'].to_numpy()[1:]

        
        elif self.desi_tracer == 'elg':
 
            lop_north = tracer_data['ELG_LOP_NORTH']
            lop_south_decal = tracer_data['ELG_LOP_SOUTH_DECALS']
            lop_south_des = tracer_data['ELG_LOP_SOUTH_DES']
            nz_avg = (lop_north * NORTH_AREA + lop_south_decal * SOUTH_DECAL_AREA  + lop_south_des * SOUTH_DES_AREA)/(TOTAL_DESI_AREA)
            fiber_rate_elg = 0.69 
            nz_avg = nz_avg / fiber_rate_elg 


        elif self.desi_tracer == 'qso':
            
            z_mask = np.logical_and(tracer_data['z'] > self.z_range[0], tracer_data['z'] < self.z_range[1])
            nz_north = tracer_data['n_z_north'][z_mask]
            nz_south = tracer_data['n_z_south'][z_mask]
            nz_avg = (nz_north*NORTH_AREA + nz_south*(SOUTH_DECAL_AREA + SOUTH_DES_AREA)) / TOTAL_DESI_AREA 


        print(f'The redshift range of the tracer being emulated is {np.min(z_bin_min)} - {np.max(z_bin_max)}')
        
        repeat_n = int((self.z_grid_points-1)/len(z_bin_center))
        new_z_bin_min = np.linspace(np.min(z_bin_min), np.max(z_bin_max),  self.z_grid_points)[:-1]
        new_z_bin_max = np.linspace(np.min(z_bin_min), np.max(z_bin_max),  self.z_grid_points)[1:]
        new_z_center = (new_z_bin_max + new_z_bin_min) / 2
        
        z_bin_center_pad = np.insert(z_bin_center, 0, z_bin_center[0] - (z_bin_center[1] - z_bin_center[0])/2)
        z_bin_center_pad = np.append(z_bin_center_pad , z_bin_center[-1] + (z_bin_center[-1] - z_bin_center[-2])/2)
        
        nz_avg_pad = np.insert(nz_avg, 0, nz_avg[0])
        nz_avg_pad = np.append(nz_avg_pad, nz_avg[-1])

        interp_nz_avg_func = interp1d(z_bin_center_pad, nz_avg_pad, fill_value=0, bounds_error=False)
        interp_nz_avg = interp_nz_avg_func(new_z_center) / repeat_n

        zgrid = np.linspace(np.min(z_bin_min), np.max(z_bin_max), self.z_grid_points)
        values, edges = np.histogram(self.sim_cat['redshift_true'], bins=zgrid)
        values_sim = values / self.sim_area
        z_frac = interp_nz_avg / values_sim
        z_frac = np.nan_to_num(z_frac, nan=0.0, posinf=0.0, neginf=0.0)
        z_frac = np.minimum(z_frac, np.ones(len(z_frac))*0.99)

        print(f'The max value in z_frac is {np.max(z_frac)}')

        self.new_z_bin_min = new_z_bin_min
        self.new_z_center = new_z_center
        self.new_z_bin_max = new_z_bin_max
        self.z_frac = z_frac
        self.nz_avg = nz_avg

        
        # save the new z center
        # path_z_center = f'/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/selection_z_centers/{self.desi_tracer}/{self.z_grid_points}_centers.npy'
        # np.save(path_z_center, self.new_z_center)
        
    
    def generate_threshold(self):
        
        thres_list = []
        
        for i in range(len(self.new_z_center)):
            
            this_zmin = self.new_z_bin_min[i]
            this_zmax = self.new_z_bin_max[i]
            this_cat = self.sim_cat[np.logical_and(self.sim_cat['redshift_true']>this_zmin, self.sim_cat['redshift_true']<this_zmax)]
        
            if len(this_cat) == 0:
                
                print(f"Empty bin: zmin={this_zmin}, zmax={this_zmax}")
                this_thres = 10**40 # set threshold to high value to not select anything

            else:

                if self.select_biggest:
                
                    this_thres = np.percentile(a = this_cat[self.threshold_col], q = 100-self.z_frac[i]*100)
    
                else:
                    
                    this_thres = np.percentile(a = this_cat[self.threshold_col], q = self.z_frac[i]*100)
        
            thres_list.append(this_thres)
                        
                     
        self.thres_list = thres_list


    def produce_desi_mock(self):
      
        thres_of_z = interpolate.interp1d(self.new_z_center, self.thres_list,  fill_value="extrapolate", bounds_error=False)
        threshold_all = thres_of_z(self.sim_cat['redshift_true'])

        # save the threshold values
        # path_threshold = f'/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/selection_thresholds/{self.desi_tracer}/{self.threshold_col}_thres.npy'
        # np.save(path_threshold, threshold_all)

        if self.select_biggest:
            
            mask_abundance = self.sim_cat[self.threshold_col] > threshold_all

        else:

            mask_abundance = self.sim_cat[self.threshold_col] < threshold_all
        
        mock_cat = self.sim_cat[mask_abundance]

        return mock_cat

    def produce_desi_rands(self, mock_cat=None):

        
        RAND_TO_DATA_RATIO = 10
        ntot = int(len(mock_cat)* RAND_TO_DATA_RATIO)
        
        ra_min = np.min(self.sim_cat['ra'])
        ra_max = np.max(self.sim_cat['ra'])
        
        rng = self._rng()
        rand_ra = ra_min + (ra_max - ra_min)*rng.random(size=ntot)
        cth_min = np.min(np.sin(np.radians(self.sim_cat['dec'])))
        cth_max = np.max(np.sin(np.radians(self.sim_cat['dec'])))
        cth_rand = cth_min + (cth_max - cth_min)*rng.random(size=ntot)
        rand_dec = np.degrees(np.arcsin(cth_rand))        
            
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
        sep = DesiSelector.cosmo.h * np.exp(dd_s.meanlogr)
    
        return {'sep': sep,
               'xi': xi_s,
                'var_xi': var_xi,
               'dd': dd_s,
               'dr': dr_s,
               "rr": rr_s,
               "xi_naive": xi_s_naive,
               'var_xi_naive': var_xi_naive}