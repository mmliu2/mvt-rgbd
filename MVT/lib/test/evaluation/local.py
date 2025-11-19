from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.depthtrack_path = '/mnt/mvt-rgbd/data/depthtrack/test'
    settings.got10k_lmdb_path = '/mnt/mvt-rgbd/data/got10k_lmdb'
    settings.got10k_path = '/mnt/mvt-rgbd/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/mnt/mvt-rgbd/data/itb'
    settings.lasot_extension_subset_path_path = '/mnt/mvt-rgbd/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/mnt/mvt-rgbd/data/lasot_lmdb'
    settings.lasot_path = '/mnt/mvt-rgbd/data/lasot'
    settings.network_path = '/mnt/mvt-rgbd/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/mnt/mvt-rgbd/data/nfs'
    settings.otb_path = '/mnt/mvt-rgbd/data/otb'
    settings.prj_dir = '/mnt/mvt-rgbd/MVT'
    settings.result_plot_path = '/mnt/mvt-rgbd/output/test/result_plots'
    settings.results_path = '/mnt/mvt-rgbd/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/mnt/mvt-rgbd/output'
    settings.segmentation_path = '/mnt/mvt-rgbd/output/test/segmentation_results'
    settings.tc128_path = '/mnt/mvt-rgbd/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/mnt/mvt-rgbd/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/mnt/mvt-rgbd/data/trackingnet'
    settings.uav_path = '/mnt/mvt-rgbd/data/uav'
    settings.vot18_path = '/mnt/mvt-rgbd/data/vot2018'
    settings.vot22_path = '/mnt/mvt-rgbd/data/vot2022'
    settings.vot_path = '/mnt/mvt-rgbd/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

