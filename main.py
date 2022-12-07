import image_processing as ip
import analysis as ana
import numpy as np

sample_size = 10000
k_images = 14
d2_stop = 10
k_patches = [500]
patchsize = [8]
stepsize = [[1]]
path_csv_folder = "/Users/sz/Documents/Uni/Master/Masterarbeit/Masterarbeit_Projekt/ImageClusteringMain/clustering_results/cod/csv"
path_csv_single = "/Users/sz/Documents/Uni/Master/Masterarbeit/Masterarbeit_Projekt/ImageClusteringMain/COD_metrics_sample_size10000_k_images7_k_patches500_patchsize8_stepsize1.csv"
path_csv_single_test = "/Users/sz/Documents/Uni/Master/Masterarbeit/Masterarbeit_Projekt/ImageClusteringMain/CM_metrics_COT_sample_size10000_k_images14_k_patches500_patchsize8_stepsize1.csv"

def main():
    # ip.cmask_clustering_fast(sample_size, k_images, d2_stop, k_patches, patchsize, stepsize)
    # ip.cod_clustering_fast(sample_size, k_images, d2_stop, k_patches, patchsize, stepsize)
    ip.cot_clustering_fast(sample_size, k_images, d2_stop, k_patches, patchsize, stepsize)
    # ana.plot_from_csv(path_csv_single)
    # ana.plot_all(path_csv_folder)
    # ana.nof(path_csv_folder)
    # ana.boxplots_cloud_fraction(path_csv_single)
    # ana.boxplots_osmean(path_csv_single)
    # ana.boxplots_meanls(path_csv_single)
    # ana.boxplots_orientation(path_csv_single)
    # ana.boxplots_iorg(path_csv_single)
    # ana.boxplots_cod(path_csv_single)
    # ana.plot_clustering_csv2(path_csv_single_test, 'Spectral_r', 'cot')

# Run main
if __name__ == "__main__":
    main()