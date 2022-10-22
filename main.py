import image_processing as ip
import analysis as ana

def main():
    # ip.cmask_clustering(1000, 8, 20, [100, 500], [3, 8, 32], [[1, 2, 3], [1, 4, 8], [1, 16, 32]])
    # ana.plot_from_csv("/Users/sz/Documents/Uni/Master/Masterarbeit/Masterarbeit_Projekt/ImageClusteringMain/cmask_clustering_results/csv/05_D2_metrics_k_images8_k_patches100_patchsize8_steps_patches4.csv")
    # ana.plot_all("/Users/sz/Documents/Uni/Master/Masterarbeit/Masterarbeit_Projekt/ImageClusteringMain/cmask_clustering_results/csv/")
    ana.nof("/Users/sz/Documents/Uni/Master/Masterarbeit/Masterarbeit_Projekt/ImageClusteringMain/cmask_clustering_results/csv/")

# Run main
if __name__ == "__main__":
    main()