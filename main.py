import image_processing as ip
import analysis as ana

sample_size = 100
k_images = 8
d2_stop = 20
k_patches = [100]
patchsize = [8]
stepsize = [[1]]
path_csv_folder = "/Users/sz/Documents/Uni/Master/Masterarbeit/Masterarbeit_Projekt/ImageClusteringMain/cmask_clustering_results/csv/"
path_csv_single = "/Users/sz/Documents/Uni/Master/Masterarbeit/Masterarbeit_Projekt/ImageClusteringMain/cmask_clustering_results/csv/05_D2_metrics_k_images8_k_patches100_patchsize8_steps_patches4.csv"

def main():
    # ip.cmask_clustering(sample_size, k_images, d2_stop, k_patches, patchsize, stepsize)
    ip.cod_clustering(sample_size, k_images, d2_stop, k_patches, patchsize, stepsize)
    # ana.plot_from_csv(path_csv_single)
    # ana.plot_all(path_csv_folder)
    # ana.nof(path_csv_folder)
    

# Run main
if __name__ == "__main__":
    main()