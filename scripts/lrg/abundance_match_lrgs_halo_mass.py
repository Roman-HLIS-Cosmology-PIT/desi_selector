import sys
sys.path.append('/global/homes/y/yoki/roman/desi_selector/')
from src import desi_selector
from importlib import reload
reload(desi_selector)
from src.desi_selector import DesiSelector as ds
import numpy as np
 
desi_tracer = 'lrg'
path_desi_tracer = {'path':'/pscratch/sd/y/yoki/desi_like_data_diffsky/data/desi_sv_data/desi_lrg_ts_zenodo/fig_1_histograms.csv'}
path_sim = '/global/cfs/cdirs/hacc/OpenCosmo/LastJourney/synthetic_galaxies/'        
calibration = 'hlwas_cosmos_260215_02_17_2026'
z_grid_points = 601
select_biggest = True
z_range = [0, 1.5]
threshold_col = 'log_peak_sub_halo_mass_noisy'
reload_oc = False
sigma_arr = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
                             


for sigma in sigma_arr:
    
    mock_cat_selector = ds(desi_tracer=desi_tracer, 
                             path_desi_tracer=path_desi_tracer,
                             path_sim=path_sim, 
                             calibration_version=calibration,
                             select_biggest=select_biggest,
                             z_range=z_range,
                             z_grid_points=z_grid_points,
                             threshold_col=threshold_col,
                             reload_oc=reload_oc,
                             sigma_dex=sigma)
    
    
    mock_cat_selector.rebin_desi_tracer()
    mock_cat_selector.generate_threshold()
    
    mock_cat = mock_cat_selector.produce_desi_mock()
    rand_cat = mock_cat_selector.produce_desi_rands(mock_cat=mock_cat)

    sim_area =  mock_cat_selector.sim_area
    path_mock_output = f'/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/mock_cats/{desi_tracer}/mock_{desi_tracer}_cat_{calibration}_{threshold_col}_{sim_area}deg2_{sigma}sigma.parquet'
    path_rand_output = f'/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/mock_cats/{desi_tracer}/rand_{desi_tracer}_cat_{calibration}_{threshold_col}_{sim_area}deg2_{sigma}sigma.parquet'
    
    mock_cat.to_parquet(path_mock_output)
    rand_cat.to_parquet(path_rand_output)

    # args to be passed into auto-correlation function
    z_range_clustering = [[0.4, 0.6], [0.6, 0.8], [0.8, 1.1]]
    NBINS = 100
    MIN_SEP = 1
    MAX_SEP = 200

    for z_range_cluster in z_range_clustering:
        corr_results = mock_cat_selector.measure_autocorr(mock_cat=mock_cat, rand_cat=rand_cat, z_range_clustering=z_range_cluster, min_sep=MIN_SEP, max_sep=MAX_SEP, nbins=NBINS)
        path_corr_results = f'/global/homes/y/yoki/roman/desi_like_samples/diffsky/data/corr_results/{desi_tracer}/z_{z_range_cluster[0]}_{z_range_cluster[1]}_{calibration}_{threshold_col}_{sim_area}deg2_{sigma}sigma.npz'
        np.savez(path_corr_results, sep=corr_results['sep'], xi=corr_results['xi'])