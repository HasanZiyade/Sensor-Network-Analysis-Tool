from functions import SensorNetwork

def main():
    """Main execution function to demonstrate sensor network functionalities."""
    # Create sensor network with data from file and a custom point
    network = SensorNetwork('Nodes.txt', custom_point=[0.2, 0.2])
    
    # Plot basic network visualizations
    network.plot_network_connectivity()
    network.plot_network_coverage()
    
    # Compute Voronoi and Delaunay diagrams
    network.compute_voronoi()
    network.compute_delaunay()
    
    # Find maximal breach path and maximal support path
    network.find_maximal_breach_path()
    network.find_maximal_support_path()

if __name__ == "__main__":
    main()