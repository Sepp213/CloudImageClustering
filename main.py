import image_processing as ip
import analysis as ana

sample_size = 10000
k_images = 5
d2_stop = 100
k_patches = [500]
patchsize = [8]
stepsize = [[1]]
tresh_patch_dist = 12
tresh_cm_cod = 0.3
tresh_cm_cot = 0.3
path_cm_results_k7 = '/Users/sz/Documents/Uni/Master/Masterarbeit/Masterarbeit_Projekt/ImageClusteringMain/clustering_results/cmask/k7/csv/CM_metrics_sample_size10000_k_images7_k_patches500_patchsize8_stepsize1.csv'
path_cot_results = '/Users/sz/Documents/Uni/Master/Masterarbeit/Masterarbeit_Projekt/ImageClusteringMain/clustering_results/cot/COT_metrics_sample_size10000_k_images7_k_patches500_patchsize8_stepsize1.csv'
path_cot_results_physicals = '/Users/sz/Documents/Uni/Master/Masterarbeit/Masterarbeit_Projekt/ImageClusteringMain/clustering_results/cot/10k_physical_metrics_COT_k7.csv'

def main():
    # ip.cmask_clustering_fast(sample_size, k_images, d2_stop, k_patches, patchsize, stepsize, tresh_patch_dist)
    # ip.cot_clustering_fast(sample_size, k_images, d2_stop, k_patches, patchsize, stepsize, tresh_patch_dist, tresh_cm_cot)
    # ana.boxplots_cloud_fraction(path_cm_results_k7)
    # ana.boxplots_iorg(path_cm_results_k7)
    # ana.boxplots_cod(path_cot_results)
    # ana.boxplots_maxls_meanls(path_cm_results_k7)
    # ana.merge_physicals(path1, path2)
    # ana.plot_cluster_range_csv(path_cm_results_k7, 'Spectral_r', 7, 10, False, True)
    # ana.plot_cod_distr(path_cot_results)
    ana.plot_cmap('Spectral_r')

# Run main
if __name__ == "__main__":
    main()