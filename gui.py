import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import threading

from functions import SensorNetwork

class SensorNetworkGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sensor Network Analysis Tool")
        self.root.geometry("900x700")  # Keep current window size
        
        # Initialize empty points list for custom node placement
        self.custom_points = []
        
        # Default values for connectivity and coverage radius
        self.connectivity_radius = 0.25
        self.coverage_radius = 0.2
        
        # Bottom controls frame - create FIRST and pack at BOTTOM
        self.bottom_frame = ttk.Frame(root)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        # Create parameters frame for sliders in bottom frame
        self.params_frame = ttk.LabelFrame(self.bottom_frame, text="Network Parameters")
        self.params_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create control panel in bottom frame
        self.control_frame = ttk.Frame(self.bottom_frame)
        self.control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create status bar at the very bottom
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Click to place nodes")
        self.status_bar = ttk.Label(self.bottom_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        # Main frame with the notebook - create AFTER bottom frame
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create the main notebook with fixed height
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.placement_tab = ttk.Frame(self.notebook)
        self.connectivity_tab = ttk.Frame(self.notebook)
        self.coverage_tab = ttk.Frame(self.notebook)
        self.breach_tab = ttk.Frame(self.notebook)
        self.support_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.placement_tab, text="Node Placement")
        self.notebook.add(self.connectivity_tab, text="Network Connectivity")
        self.notebook.add(self.coverage_tab, text="Network Coverage")
        self.notebook.add(self.breach_tab, text="Maximal Breach Path")
        self.notebook.add(self.support_tab, text="Maximal Support Path")
        
        # Setup parameter sliders
        self.setup_parameter_controls()
        
        # Setup the plots
        self.setup_plots()
        
        # Node placement controls are now integrated in setup_plots
        
        # Setup analysis controls
        self.setup_controls()
        
        # Initialize the placement plot
        self.plot_placement_grid()
        
        # No network instance at first - it will be created after node placement
        self.network = None

        # Set a fixed maximum height for the notebook to leave room for controls
        self.root.update()  # Force an update to get correct dimensions
        notebook_height = self.root.winfo_height() - 200  # Reserve 200px for bottom controls
        self.notebook.configure(height=notebook_height)
    
    def setup_parameter_controls(self):
        """Setup sliders for connectivity and coverage radius."""
        # Connectivity radius slider
        ttk.Label(self.params_frame, text="Connectivity Radius (RC):").grid(
            row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.rc_var = tk.DoubleVar(value=self.connectivity_radius)
        rc_slider = ttk.Scale(
            self.params_frame, from_=0.05, to=0.5, 
            orient=tk.HORIZONTAL, length=200, variable=self.rc_var,
            command=self.update_rc_value)
        rc_slider.grid(row=0, column=1, padx=5, pady=5, sticky="we")
        
        self.rc_value_label = ttk.Label(self.params_frame, text=f"{self.connectivity_radius:.2f}")
        self.rc_value_label.grid(row=0, column=2, padx=5, pady=5)
        
        # Coverage radius slider
        ttk.Label(self.params_frame, text="Coverage Radius (RS):").grid(
            row=1, column=0, padx=5, pady=5, sticky="w")
        
        self.rs_var = tk.DoubleVar(value=self.coverage_radius)
        rs_slider = ttk.Scale(
            self.params_frame, from_=0.05, to=0.5, 
            orient=tk.HORIZONTAL, length=200, variable=self.rs_var,
            command=self.update_rs_value)
        rs_slider.grid(row=1, column=1, padx=5, pady=5, sticky="we")
        
        self.rs_value_label = ttk.Label(self.params_frame, text=f"{self.coverage_radius:.2f}")
        self.rs_value_label.grid(row=1, column=2, padx=5, pady=5)
        
        # Apply button
        ttk.Button(
            self.params_frame, text="Apply Parameters",
            command=self.apply_parameters
        ).grid(row=0, column=3, rowspan=2, padx=10, pady=5)
        
        # Configure grid
        self.params_frame.columnconfigure(1, weight=1)
        
    def update_rc_value(self, value):
        """Update connectivity radius value label."""
        value = float(value)
        self.rc_value_label.config(text=f"{value:.2f}")
        
    def update_rs_value(self, value):
        """Update coverage radius value label."""
        value = float(value)
        self.rs_value_label.config(text=f"{value:.2f}")
        
    def apply_parameters(self):
        """Apply the new parameter values and update plots if network exists."""
        self.connectivity_radius = self.rc_var.get()
        self.coverage_radius = self.rs_var.get()
        
        self.status_var.set(f"Parameters updated: RC={self.connectivity_radius:.2f}, RS={self.coverage_radius:.2f}")
        
        if self.network:
            if self.notebook.index(self.notebook.select()) == 1:  # Connectivity tab
                self.plot_connectivity()
            elif self.notebook.index(self.notebook.select()) == 2:  # Coverage tab
                self.plot_coverage()
    
    def setup_plots(self):
        """Setup the matplotlib figures for each tab with fixed square dimensions."""
        
        # Node placement tab with left, center, right layout
        placement_main_frame = ttk.Frame(self.placement_tab)
        placement_main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left controls panel
        left_controls = ttk.Frame(placement_main_frame)
        left_controls.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # Add buttons to left panel
        ttk.Button(left_controls, text="Clear All\nNodes", 
                  command=self.clear_nodes, width=10).pack(side=tk.TOP, padx=5, pady=10)
        
        ttk.Button(left_controls, text="Add Start/End\nNodes", 
                  command=self.add_start_end_nodes, width=10).pack(side=tk.TOP, padx=5, pady=10)
        
        ttk.Button(left_controls, text="Add Random\nNodes", 
                  command=lambda: self.add_random_nodes(10), width=10).pack(side=tk.TOP, padx=5, pady=10)
        
        # Center plot frame with fixed dimensions to maintain square aspect
        plot_frame = ttk.Frame(placement_main_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create canvas with fixed size for placement tab
        canvas_size = 400  # Fixed size in pixels
        self.placement_fig = Figure(figsize=(5, 5), dpi=80, constrained_layout=True)
        self.placement_canvas = FigureCanvasTkAgg(self.placement_fig, master=plot_frame)
        canvas_widget = self.placement_canvas.get_tk_widget()
        canvas_widget.config(width=canvas_size, height=canvas_size)
        canvas_widget.pack(side=tk.TOP, anchor=tk.CENTER, padx=10, pady=10)
        
        # Right controls panel
        right_controls = ttk.Frame(placement_main_frame)
        right_controls.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        
        # Add buttons to right panel
        ttk.Button(right_controls, text="Initialize\nNetwork", 
                  command=self.initialize_network, width=10).pack(side=tk.TOP, padx=5, pady=10)
        
        ttk.Button(right_controls, text="Save\nNodes", 
                  command=self.save_nodes, width=10).pack(side=tk.TOP, padx=5, pady=10)
        
        ttk.Button(right_controls, text="Load\nNodes", 
                  command=self.load_nodes, width=10).pack(side=tk.TOP, padx=5, pady=10)
        
        # Node count display
        self.node_count_label = ttk.Label(right_controls, text="Nodes: 0")
        self.node_count_label.pack(side=tk.TOP, padx=10, pady=20)
        
        # Add save button at the bottom
        save_frame = ttk.Frame(self.placement_tab)
        save_frame.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(save_frame, text="Save Plot", 
                  command=lambda: self.save_plot(self.placement_fig, "node_placement")).pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Enable mouse click events for node placement
        self.placement_canvas.mpl_connect('button_press_event', self.on_click)
        
        # Create the other tab plots with same square dimensions
        self._create_square_plot_tab(self.connectivity_tab, "connectivity", "network_connectivity")
        self._create_square_plot_tab(self.coverage_tab, "coverage", "network_coverage")
        self._create_square_plot_tab(self.breach_tab, "breach", "maximal_breach_path")
        self._create_square_plot_tab(self.support_tab, "support", "maximal_support_path")

    def _create_square_plot_tab(self, tab, attr_prefix, save_filename):
        """Create a tab with a square plot and a save button."""
        plot_frame = ttk.Frame(tab)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create canvas with fixed pixel dimensions (400x400)
        canvas_size = 400
        fig = Figure(figsize=(5, 5), dpi=80, constrained_layout=True)
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.config(width=canvas_size, height=canvas_size)
        canvas_widget.pack(side=tk.TOP, anchor=tk.CENTER, padx=10, pady=10)
        
        # Save references
        setattr(self, f"{attr_prefix}_fig", fig)
        setattr(self, f"{attr_prefix}_canvas", canvas)
        
        # Add save button
        save_frame = ttk.Frame(tab)
        save_frame.pack(side=tk.BOTTOM, fill=tk.X)
        ttk.Button(save_frame, text="Save Plot", 
                  command=lambda: self.save_plot(getattr(self, f"{attr_prefix}_fig"), save_filename)).pack(
                      side=tk.RIGHT, padx=10, pady=5)
    
    def setup_placement_controls(self):
        """This method is now empty as controls are created in setup_plots."""
        pass
    
    def setup_controls(self):
        """Setup control buttons for analyses in multiple rows for better visibility."""
        # Use a more structured grid layout instead of pack
        control_inner = ttk.Frame(self.control_frame)
        control_inner.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Row 1 - Analysis buttons
        ttk.Button(control_inner, text="Show Network Connectivity", 
                  command=lambda: self.run_task(self.plot_connectivity, "Plotting network connectivity...")
                  ).grid(row=0, column=0, padx=3, pady=3, sticky="ew")
        
        ttk.Button(control_inner, text="Show Network Coverage", 
                  command=lambda: self.run_task(self.plot_coverage, "Plotting network coverage...")
                  ).grid(row=0, column=1, padx=3, pady=3, sticky="ew")
        
        ttk.Button(control_inner, text="Find Maximal Breach Path", 
                  command=lambda: self.run_task(self.plot_breach_path, "Finding maximal breach path...")
                  ).grid(row=0, column=2, padx=3, pady=3, sticky="ew")
        
        # Row 2 - More analysis and save buttons
        ttk.Button(control_inner, text="Find Maximal Support Path", 
                  command=lambda: self.run_task(self.plot_support_path, "Finding maximal support path...")
                  ).grid(row=1, column=0, padx=3, pady=3, sticky="ew")
        
        ttk.Button(control_inner, text="Run All Analyses", 
                  command=lambda: self.run_task(self.run_all_analyses, "Running all analyses...")
                  ).grid(row=1, column=1, padx=3, pady=3, sticky="ew")
        
        # Save all plots button
        ttk.Button(control_inner, text="Save All Plots", 
                  command=self.save_all_plots
                  ).grid(row=1, column=2, padx=3, pady=3, sticky="ew")
        
        # Configure the grid columns to expand evenly
        for i in range(3):
            control_inner.columnconfigure(i, weight=1)
    
    def plot_placement_grid(self):
        """Plot the grid for node placement."""
        self.placement_fig.clear()
        self.ax_placement = self.placement_fig.add_subplot(111)
        
        # Draw the unit square with grid
        self.ax_placement.set_xlim([0, 1])
        self.ax_placement.set_ylim([0, 1])
        self.ax_placement.grid(True, linestyle='--', alpha=0.7)
        
        self.ax_placement.set_title("Click to Place Nodes", fontsize=14)
        self.ax_placement.set_xlabel("x-Coordinate (L)", fontsize=12)
        self.ax_placement.set_ylabel("y-Coordinate (L)", fontsize=12)
        
        # Plot any existing nodes
        if self.custom_points:
            points = np.array(self.custom_points)
            self.ax_placement.plot(points[:, 0], points[:, 1], 'bo', markersize=8)
            
            # Highlight start and end nodes if they exist
            if len(self.custom_points) >= 2:
                # Show the first point placed as the start node
                self.ax_placement.plot(self.custom_points[0][0], self.custom_points[0][1], 
                                      'go', markersize=10, label="Start")
                
                # Show the second point placed as the end node
                self.ax_placement.plot(self.custom_points[1][0], self.custom_points[1][1], 
                                      'mo', markersize=10, label="End")
                
                self.ax_placement.legend()
        
        self.placement_canvas.draw()
    
    def on_click(self, event):
        """Handle mouse click events for node placement."""
        if event.xdata is not None and event.ydata is not None:
            # Check if click is within bounds
            if 0 <= event.xdata <= 1 and 0 <= event.ydata <= 1:
                # Add the node
                self.custom_points.append([event.xdata, event.ydata])
                self.status_var.set(f"Node added at ({event.xdata:.3f}, {event.ydata:.3f})")
                
                # Update the node count label
                self.node_count_label.config(text=f"Nodes: {len(self.custom_points)}")
                
                # Redraw the plot
                self.plot_placement_grid()
    
    def clear_nodes(self):
        """Clear all placed nodes."""
        self.custom_points = []
        self.node_count_label.config(text="Nodes: 0")
        self.status_var.set("All nodes cleared")
        self.plot_placement_grid()
        self.network = None
    
    def add_start_end_nodes(self):
        """Add default start and end nodes at (0,0) and (1,1)."""
        # Add start node at (0,0)
        if not any(np.allclose(np.array(p), np.array([0, 0])) for p in self.custom_points):
            self.custom_points.insert(0, [0, 0])
        
        # Add end node at (1,1)
        if not any(np.allclose(np.array(p), np.array([1, 1])) for p in self.custom_points):
            if len(self.custom_points) == 1:
                self.custom_points.append([1, 1])
            else:
                # Insert at position 1 if there are already points
                self.custom_points.insert(1, [1, 1])
        
        self.status_var.set("Start (0,0) and end (1,1) nodes added")
        self.node_count_label.config(text=f"Nodes: {len(self.custom_points)}")
        self.plot_placement_grid()
    
    def add_random_nodes(self, count=10):
        """Add random nodes within the unit square."""
        new_points = np.random.rand(count, 2)
        self.custom_points.extend(new_points.tolist())
        self.status_var.set(f"Added {count} random nodes")
        self.node_count_label.config(text=f"Nodes: {len(self.custom_points)}")
        self.plot_placement_grid()
    
    def save_nodes(self):
        """Save the current node configuration to a file."""
        if not self.custom_points:
            messagebox.showwarning("No Nodes", "No nodes to save.")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save node coordinates"
        )
        
        if filename:
            try:
                np.savetxt(filename, np.array(self.custom_points), delimiter=',')
                self.status_var.set(f"Nodes saved to {filename}")
            except Exception as e:
                self.status_var.set(f"Error saving nodes: {str(e)}")
                messagebox.showerror("Save Error", str(e))
    
    def load_nodes(self):
        """Load node configuration from a file."""
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Load node coordinates"
        )
        
        if filename:
            try:
                self.custom_points = np.genfromtxt(filename, delimiter=',').tolist()
                if isinstance(self.custom_points[0], float):  # Handle single point case
                    self.custom_points = [self.custom_points]
                self.node_count_label.config(text=f"Nodes: {len(self.custom_points)}")
                self.status_var.set(f"Loaded {len(self.custom_points)} nodes from {filename}")
                self.plot_placement_grid()
            except Exception as e:
                self.status_var.set(f"Error loading nodes: {str(e)}")
                messagebox.showerror("Load Error", str(e))
    
    def initialize_network(self):
        """Initialize the network with the placed nodes."""
        if len(self.custom_points) < 3:
            messagebox.showwarning("Not Enough Nodes", 
                                  "Need at least 3 nodes to create a network.")
            return
        
        try:
            # Create a SensorNetwork instance with the custom points
            self.network = SensorNetwork(points=np.array(self.custom_points))
            self.status_var.set(f"Network initialized with {len(self.custom_points)} nodes")
            messagebox.showinfo("Network Initialized", 
                              f"Network has been initialized with {len(self.custom_points)} nodes.\n"
                              "You can now run analyses.")
        except Exception as e:
            self.status_var.set(f"Error initializing network: {str(e)}")
            messagebox.showerror("Initialization Error", str(e))
    
    def run_task(self, task_func, status_message):
        """Run a task in a separate thread to keep the UI responsive."""
        if not self.network:
            messagebox.showwarning("No Network", 
                                  "You need to initialize the network first.")
            return
            
        self.status_var.set(status_message)
        self.root.update_idletasks()
        
        # Run the task in a separate thread to avoid freezing the UI
        threading.Thread(target=self._thread_task, args=(task_func,), daemon=True).start()
    
    def _thread_task(self, task_func):
        """Execute the task and update the status when done."""
        try:
            task_func()
            self.root.after(0, lambda: self.status_var.set("Ready"))
        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))
    
    def _show_error(self, message):
        """Show error message in status bar and dialog."""
        self.status_var.set(f"Error: {message}")
        messagebox.showerror("Error", message)
    
    def plot_connectivity(self):
        """Plot network connectivity in GUI using the current connectivity radius."""
        self.connectivity_fig.clear()
        ax = self.connectivity_fig.add_subplot(111)
        
        RC = self.connectivity_radius
        ax.plot(self.network.points[:, 0], self.network.points[:, 1], 'bo', label='Nodes')
        
        for i in range(len(self.network.points)):
            for j in range(i + 1, len(self.network.points)):
                if np.linalg.norm(self.network.points[i] - self.network.points[j]) <= RC:
                    ax.plot([self.network.points[i, 0], self.network.points[j, 0]], 
                           [self.network.points[i, 1], self.network.points[j, 1]], 'r-', alpha=0.5)
        
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.set_xlabel('x-Coordinate (L)'); ax.set_ylabel('y-Coordinate (L)')
        ax.legend(); ax.set_title(f'Network Connectivity (RC={RC:.2f})')
        
        self.connectivity_canvas.draw()
        self.notebook.select(self.connectivity_tab)
    
    def plot_coverage(self):
        """Plot network coverage in GUI using the current coverage radius."""
        self.coverage_fig.clear()
        ax = self.coverage_fig.add_subplot(111)
        
        RS = self.coverage_radius
        ax.plot(self.network.points[:, 0], self.network.points[:, 1], 'bo', label='Nodes')
        
        for i in range(len(self.network.points)):
            ax.add_patch(plt.Circle((self.network.points[i, 0], self.network.points[i, 1]), 
                                    RS, color='g', fill=True, alpha=0.2))
        
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.set_xlabel('x-Coordinate (L)'); ax.set_ylabel('y-Coordinate (L)')
        ax.legend(); ax.set_title(f'Network Coverage (RS={RS:.2f})')
        
        self.coverage_canvas.draw()
        self.notebook.select(self.coverage_tab)
    
    def plot_breach_path(self):
        """Find and plot maximal breach path in GUI."""
        # Compute Voronoi if needed
        if not hasattr(self.network, 'regions') or not hasattr(self.network, 'vertices'):
            self.network.compute_voronoi()
        
        # This will compute the maximal breach path
        mbp, max_cost = self.network.find_maximal_breach_path(
            save=False, connectivity_radius=self.connectivity_radius)
        
        # We'll manually plot it in the GUI
        self.breach_fig.clear()
        ax = self.breach_fig.add_subplot(111)
        
        # Plot Voronoi regions
        for region in self.network.regions:
            polygon = [self.network.vertices[i] for i in region]
            ax.fill(*zip(*polygon), alpha=0.1, edgecolor='orange', linewidth=1)
        
        # Plot sensor points
        ax.plot(self.network.points[:, 0], self.network.points[:, 1], 'bo', label='Sensors')
        ax.plot(0, 0, 'go', markersize=10, label='Start (0,0)')
        ax.plot(1, 1, 'mo', markersize=10, label='End (1,1)')
        
        # Plot the breach path
        if mbp:
            points_array = np.array(mbp)
            ax.plot(points_array[:, 0], points_array[:, 1], 'r-', linewidth=2.5, 
                    label=f'Maximal Breach Path (min dist: {max_cost:.3f})')
        
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.set_xlabel('x-Coordinate (L)', fontsize=12); ax.set_ylabel('y-Coordinate (L)', fontsize=12)
        ax.legend(fontsize=10)
        ax.set_title('Maximal Breach Path on Voronoi Diagram', fontsize=14)
        
        self.breach_canvas.draw()
        self.notebook.select(self.breach_tab)
    
    def plot_support_path(self):
        """Find and plot maximal support path in GUI."""
        # Compute Delaunay if needed
        if not hasattr(self.network, 'tri'):
            self.network.compute_delaunay()
        
        # This will compute the maximal support path
        msp_indices, max_cost = self.network.find_maximal_support_path(
            save=False, coverage_radius=self.coverage_radius)
        
        # We'll manually plot it in the GUI
        self.support_fig.clear()
        ax = self.support_fig.add_subplot(111)
        
        # Plot Delaunay triangulation
        ax.triplot(self.network.points[:, 0], self.network.points[:, 1], self.network.tri.simplices, 'g-', alpha=0.2)
        
        # Find start and end indices
        start_idx = np.argmin(np.sum((self.network.points - np.array([0, 0]))**2, axis=1))
        end_idx = np.argmin(np.sum((self.network.points - np.array([1, 1]))**2, axis=1))
        
        # Plot sensor points
        ax.plot(self.network.points[:, 0], self.network.points[:, 1], 'bo', label='Sensors')
        ax.plot(self.network.points[start_idx, 0], self.network.points[start_idx, 1], 'go', 
                markersize=10, label='Start')
        ax.plot(self.network.points[end_idx, 0], self.network.points[end_idx, 1], 'mo', 
                markersize=10, label='End')
        
        # Plot the support path
        if msp_indices:
            msp = [self.network.points[i] for i in msp_indices]
            msp_array = np.array(msp)
            ax.plot(msp_array[:, 0], msp_array[:, 1], 'r-', linewidth=2.5, 
                    label=f'Maximal Support Path (min coverage: {max_cost:.3f})')
        
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
        ax.set_xlabel('x-Coordinate (L)', fontsize=12); ax.set_ylabel('y-Coordinate (L)', fontsize=12)
        ax.legend(fontsize=10)
        ax.set_title('Maximal Support Path on Delaunay Triangulation', fontsize=14)
        
        self.support_canvas.draw()
        self.notebook.select(self.support_tab)
    
    def run_all_analyses(self):
        """Run all analyses in sequence."""
        self.plot_connectivity()  # Will use the current connectivity_radius
        self.plot_coverage()      # Will use the current coverage_radius
        self.plot_breach_path()
        self.plot_support_path()
        messagebox.showinfo("Analysis Complete", "All analyses have been completed successfully!")
    
    def save_plot(self, fig, default_name):
        """Save the current plot to an image file."""
        if fig is None:
            messagebox.showwarning("No Plot", "No plot to save.")
            return
            
        # Ask user for file location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg"),
                ("PDF files", "*.pdf"),
                ("SVG files", "*.svg"),
                ("All files", "*.*")
            ],
            initialfile=default_name
        )
        
        if file_path:
            try:
                # Save the figure
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                self.status_var.set(f"Plot saved to {file_path}")
                messagebox.showinfo("Success", f"Plot saved successfully to:\n{file_path}")
            except Exception as e:
                self.status_var.set(f"Error saving plot: {str(e)}")
                messagebox.showerror("Save Error", str(e))
    
    def save_all_plots(self):
        """Save all available plots to a directory."""
        if not self.network:
            messagebox.showwarning("No Network", "You need to initialize the network first.")
            return
        
        # Ask user for directory
        directory = filedialog.askdirectory(title="Select directory to save all plots")
        
        if not directory:
            return
        
        try:
            # Save all available plots
            saved_count = 0
            
            # Node placement
            if hasattr(self, 'placement_fig'):
                self.placement_fig.savefig(f"{directory}/node_placement.png", dpi=300, bbox_inches='tight')
                saved_count += 1
            
            # Network connectivity
            if hasattr(self, 'connectivity_fig'):
                # Make sure connectivity plot is generated
                self.plot_connectivity()
                self.connectivity_fig.savefig(f"{directory}/network_connectivity.png", dpi=300, bbox_inches='tight')
                saved_count += 1
            
            # Network coverage
            if hasattr(self, 'coverage_fig'):
                # Make sure coverage plot is generated
                self.plot_coverage()
                self.coverage_fig.savefig(f"{directory}/network_coverage.png", dpi=300, bbox_inches='tight')
                saved_count += 1
            
            # Breach path
            if hasattr(self, 'breach_fig'):
                # Make sure breach path plot is generated
                self.plot_breach_path()
                self.breach_fig.savefig(f"{directory}/maximal_breach_path.png", dpi=300, bbox_inches='tight')
                saved_count += 1
            
            # Support path
            if hasattr(self, 'support_fig'):
                # Make sure support path plot is generated
                self.plot_support_path()
                self.support_fig.savefig(f"{directory}/maximal_support_path.png", dpi=300, bbox_inches='tight')
                saved_count += 1
            
            self.status_var.set(f"Saved {saved_count} plots to {directory}")
            messagebox.showinfo("Success", f"Successfully saved {saved_count} plots to:\n{directory}")
        
        except Exception as e:
            self.status_var.set(f"Error saving plots: {str(e)}")
            messagebox.showerror("Save Error", str(e))

def main():
    root = tk.Tk()
    app = SensorNetworkGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()