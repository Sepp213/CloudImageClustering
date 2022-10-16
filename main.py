import image_processing as ip
import analysis as ana

def main():
    ip.cmask_clustering(1000, 8, 20, [100, 500], [3, 8, 32], [[1, 2, 3], [1, 4, 8], [1, 16, 32]])
    # ana.plot_from_csv('metrics_adj.csv')

# Run main
if __name__ == "__main__":
    main()