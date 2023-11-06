import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import progressbar
import sys

nx, ny, nz = 62, 62, 300  # Adjust grid size as needed
lx = ly = 62
lz = 300

epsilon = 11.7

bias_n_well = -90
bias_p = 0
output_file = sys.argv[1]

grid = np.zeros((nx, ny, nz))


class Boundary:
    def __init__(self, nx, ny, nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.boundaries = np.zeros((nx, ny, nz))
        self.fixed_bounds = []

    def add_box(self, origin, dimensions, potential):
        x_start, y_start, z_start = origin
        x_end, y_end, z_end = (x_start + dimensions[0], y_start + dimensions[1], z_start + dimensions[2])
        self.boundaries[x_start:x_end, y_start:y_end, z_start:z_end] = potential
        fixed_points = np.array(np.where(self.boundaries != 0)).T
        self.fixed_bounds = [tuple(point) for point in fixed_points]

    def is_fixed(self, point):
        return tuple(point) in self.fixed_bounds

    def potential(self, point):
        return self.boundaries[point[0], point[1], point[2]]

    def present_yourself(self, x):
        fig, axes = plt.subplots(3)
        # sns.heatmap(self.boundaries[:, x, :], ax=axes[0]).set(
        #     title='y slice')
        # plt.show()
        # sns.heatmap(self.boundaries[x, :, :], ax=axes[1]).set(
        #     title='x slice')
        # plt.show()
        sns.heatmap(self.boundaries[:, :, x], ax=axes[2]).set(
            title='z slice')
        plt.show()


# Define a function to solve the Laplace equation in 3D using the finite difference method
def laplace_solver_3d(nx, ny, nz, boundaries, max_iterations=1000, tolerance=1e-5):
    grid = np.empty((nx, ny, nz), dtype=float)

    # Initialize the grid with user-defined boundary conditions
    for x, y, z in boundaries.fixed_bounds:
        grid[x, y, z] = boundaries.potential((x, y, z))

    # Create an array to store the Laplace equation update values
    laplace_update = np.zeros((nx, ny, nz), dtype=float)

    bar = progressbar.ProgressBar(maxval=max_iterations,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for iteration in range(max_iterations):
        # Calculate the Laplace equation update values for interior grid points.
        # We exclude the boundary points (0 and nx-1 in x, 0 and ny-1 in y, 0 and nz-1 in z)
        # as they are fixed and do not change during the Laplace equation solving process.

        # The Laplace equation for a 3D grid at point (x, y, z) is defined as:
        #   laplace = (grid[x + 1, y, z] + grid[x - 1, y, z] +
        #              grid[x, y + 1, z] + grid[x, y - 1, z] +
        #              grid[x, y, z + 1] + grid[x, y, z - 1] - 6 * grid[x, y, z])

        # To efficiently calculate the Laplace update for all interior grid points (excluding boundaries),
        # we use NumPy array slicing and operations:
        #   1. We create an array called 'laplace_update' with the same shape as 'grid' to store
        #      the Laplace equation update values for each interior point.
        #   2. We use array slicing to compute the Laplace equation update for all interior points
        #      simultaneously. This avoids the need for nested loops and makes the code more efficient.
        #   3. For each interior point (x, y, z), we calculate the sum of potentials from its neighbors:
        #      - grid[x + 1, y, z] + grid[x - 1, y, z] in the x-direction
        #      - grid[x, y + 1, z] + grid[x, y - 1, z] in the y-direction
        #      - grid[x, y, z + 1] + grid[x, y, z - 1] in the z-direction
        #   4. We subtract 6 times the potential at the current point (6 * grid[x, y, z]) as per the
        #      Laplace equation.
        #   5. The resulting Laplace update values are stored in the 'laplace_update' array.

        # After this calculation, 'laplace_update' contains the Laplace equation update values for
        # all interior grid points. These updates are then applied to the 'grid' to move towards
        # a solution for the Laplace equation.

        laplace_update[1:nx - 1, 1:ny - 1, 1:nz - 1] = (
                grid[2:nx, 1:ny - 1, 1:nz - 1] + grid[0:nx - 2, 1:ny - 1, 1:nz - 1] +
                grid[1:nx - 1, 2:ny, 1:nz - 1] + grid[1:nx - 1, 0:ny - 2, 1:nz - 1] +
                grid[1:nx - 1, 1:ny - 1, 2:nz] + grid[1:nx - 1, 1:ny - 1, 0:nz - 2] - 6 * grid[1:nx - 1, 1:ny - 1,
                                                                                               1:nz - 1]
        )

        # suppress updates of fixed boundaries
        for x, y, z in boundaries.fixed_bounds:
            laplace_update[x, y, z] = 0

        # Update the grid using the Laplace equation update values
        grid[1:nx - 1, 1:ny - 1, 1:nz - 1] += laplace_update[1:nx - 1, 1:ny - 1, 1:nz - 1] / 6.0

        # Check for convergence by comparing the new grid to the previous one
        max_diff = np.max(np.abs(laplace_update))
        bar.update(iteration)
        # print('iteration ', iteration, 'with delta ', max_diff)
        if max_diff < tolerance:
            bar.finish()
            print('achieved convergence')
            break

    return grid


def calculate_electric_field(potential):
    # Calculate the electric field components using central differences
    D = np.gradient(potential, 1E-4) # the D-field, by definition we applied a mesh with 1um resolution
    # the 1E-4 gets us the wanted units for the D-field (V/cm)
    E = np.multiply(D, epsilon)  # convert D field to E

    return E


def generate_init_file(output_file, e_field):
    with open(output_file, 'w') as f:
        f.write(f'primitive_laplace_solver_nWell_{bias_n_well}V_p_{bias_p}V\n'
                '##SEED##  ##EVENTS##\n'
                '##TURN## ##TILT## 1.0\n'
                '0.00 0.0 0.00\n'
                f'{nz} {nx} {ny} '
                '293. 0.0 1.12 1 '
                f'{nx} {ny} {nz} 0\n')

        for z in range(nz):
            for y in range(ny):
                for x in range(nx):
                    f.write(
                        f'{x + 1} {y + 1} {z + 1} {e_field[0][x, y, z]} {e_field[1][x, y, z]} {e_field[2][x, y, z]}\n')


# Plot the 2D vector field of the electric field
def plot_2d_vector_field(Ex, Ey, x_slice, **kwargs):
    # Extract the components of the electric field at the specified z-slice
    Ex_slice = Ex[x_slice, :, :]
    Ey_slice = Ey[x_slice, :, :]

    # Calculate the magnitude of the electric field vectors
    magnitude = np.sqrt(Ex_slice ** 2 + Ey_slice ** 2)

    # Normalize the electric field components for scaling arrow lengths
    mask = magnitude != 0
    Ex_normalized = np.zeros_like(Ex_slice)
    Ey_normalized = np.zeros_like(Ey_slice)
    Ex_normalized[mask] = Ex_slice[mask] / magnitude[mask]
    Ey_normalized[mask] = Ey_slice[mask] / magnitude[mask]

    # Create a grid for x and y coordinates
    x, y = np.meshgrid(np.arange(Ex_slice.shape[0]), np.arange(Ex_slice.shape[1]))

    # Create the vector plot with auto-adjusted arrow lengths
    plt.figure(figsize=(8, 6))
    plt.quiver(x, y, Ex_normalized, Ey_normalized, angles='xy', scale_units='xy', scale=.5, color='b')
    plt.title('Electric Field Vector Plot - Slice at {}'.format(x_slice))
    plt.xlabel(kwargs['x'])
    plt.ylabel(kwargs['y'])
    plt.xlim(0, Ex_slice.shape[0])
    plt.ylim(0, Ex_slice.shape[1])


# Plot a 2D slice of the 3D potential at a specific x value
def plot_2d_slice(solution, x_slice):
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Slice of Potential at X = {}'.format(x_slice))

    X_slice = solution[x_slice, :, :]  # Extract the 2D slice at a specific x value
    ax = sns.heatmap(X_slice)
    ax.invert_yaxis()
    plt.show()


bounds = Boundary(nx, ny, nz)
bounds.add_box((10, 10, 285), (40, 40, 15), bias_n_well)
# bounds.add_box((1,1,0),(60, 1, 1), bias_p)
# bounds.add_box((1,1,0),(1, 60, 1), bias_p)
# bounds.add_box((1,60,0),(60, 1, 1), bias_p)
# bounds.add_box((60,1,0),(1, 60, 1), bias_p)
bounds.add_box((0, 0, 100), (nx, ny, 1), 0)
bounds.present_yourself(290)

potential = laplace_solver_3d(nx, ny, nz, bounds, 10000)
E = calculate_electric_field(potential)
generate_init_file(output_file, E)
print('saved E-field to ', output_file)
plot_2d_slice(potential, 30)
