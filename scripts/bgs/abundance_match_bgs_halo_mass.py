import sys
sys.path.append('/global/homes/y/yoki/roman/desi_selector/')
from src import desi_selector
from importlib import reload
reload(desi_selector)
from src.desi_selector import DesiSelector as ds
import numpy as np
from time import time 

desi_tracer = 'bgs'
z_range = [0.10, 0.40]
z_grid_points = 301
path_desi_tracer = {'north':'/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/desi_sv_data/desi_bgs_ts_zenodo/BGS_BRIGHT-21.5_NGC_nz.txt',
                     'south':'/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/desi_sv_data/desi_bgs_ts_zenodo/BGS_BRIGHT-21.5_SGC_nz.txt'}
path_sim = '/global/cfs/cdirs/hacc/OpenCosmo/LastJourney/synthetic_galaxies/'        
calibration = 'hlwas_cosmos_260215_02_17_2026'
select_biggest = True
threshold_col = 'log_peak_sub_halo_mass_noisy'
synth_cores = False
reload_oc = False
sigma_arr = np.array([0, 0.02, 0.04, 0.06, 0.08, 0.1])


for sigma in sigma_arr:
    
        
    # start the timer 
    start = time()
    mock_cat_selector = ds(desi_tracer=desi_tracer, 
                             path_desi_tracer=path_desi_tracer,
                             path_sim=path_sim, 
                             calibration_version=calibration, 
                             z_range=z_range,
                             z_grid_points=z_grid_points,
                             select_biggest=select_biggest,
                             threshold_col=threshold_col,
                             synth_cores=synth_cores,
                             reload_oc=reload_oc,
                             sigma_dex=sigma)
    end = time()
    elapsed = end - start
    print(f"The time it took to init the class and load in the sim data was: {elapsed:.3f}s")
    
    start = time()
    mock_cat_selector.rebin_desi_tracer()
    end = time()
    elapsed = end - start
    print(f"The time it took to run rebin_desi_tracer was: {elapsed:.3f}s")
    
    start = time()
    mock_cat_selector.generate_threshold()
    end = time()
    elapsed = end - start
    
    start = time()
    mock_cat = mock_cat_selector.produce_desi_mock()
    end = time()
    elapsed = end - start
    print(f"The time it took to run produce_desi_mock was: {elapsed:.3f}s")
    
    start = time()
    rand_cat = mock_cat_selector.produce_desi_rands(mock_cat=mock_cat)
    end = time()
    elapsed = end - start
    print(f"The time it took to run produce_desi_rands was: {elapsed:.3f}s")
    
    sim_area =  mock_cat_selector.sim_area
    path_mock_output = f'/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/mock_cats/{desi_tracer}/mock_{desi_tracer}_cat_{calibration}_{threshold_col}_{sim_area}deg2_{sigma}sigma.parquet'
    path_rand_output = f'/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/mock_cats/{desi_tracer}/rand_{desi_tracer}_cat_{calibration}_{threshold_col}_{sim_area}deg2_{sigma}sigma.parquet'
    mock_cat.to_parquet(path_mock_output)
    rand_cat.to_parquet(path_rand_output)
    
    # args to be passed into auto-correlation function
    z_range_clustering = [0.1, 0.4]
    NBINS = 100
    MIN_SEP = 1
    MAX_SEP = 200
    
    start = time()
    corr_results = mock_cat_selector.measure_autocorr(mock_cat=mock_cat, rand_cat=rand_cat, z_range_clustering=z_range_clustering, min_sep=MIN_SEP, max_sep=MAX_SEP, nbins=NBINS)
    path_corr_results = f'/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/corr_results/{desi_tracer}/z_{z_range_clustering[0]}_{z_range_clustering[1]}_{calibration}_{threshold_col}_{sim_area}deg2_{sigma}sigma.npz'
    np.savez(path_corr_results, sep=corr_results['sep'], xi=corr_results['xi'])
    end = time()
    elapsed = end - start
    print(f"The time it took to run measure auto_corr was: {elapsed:.3f}s")
