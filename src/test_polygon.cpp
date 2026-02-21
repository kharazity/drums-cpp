#include <iostream>
#include <vector>
#include "geometry.h"
#include "fem.h"
#include "solver.h"

int main() {
    Mesh mesh;
    FEM fem;
    ModeData modes;

    // Create a 5-pointed star
    std::vector<Vertex> star = {
        {0.0, 1.0}, {0.22, 0.31}, {0.95, 0.31},
        {0.36, -0.12}, {0.59, -0.81}, {0.0, -0.38},
        {-0.59, -0.81}, {-0.36, -0.12}, {-0.95, 0.31},
        {-0.22, 0.31}
    };

    std::cout << "Testing generate_from_polygon with a star..." << std::endl;
    mesh.generate_from_polygon(star, 15);
    
    std::cout << "Testing FEM assembly..." << std::endl;
    fem.assemble(mesh);
    
    std::cout << "Testing Solver..." << std::endl;
    modes = Solver::solve(fem, 10);
    
    std::cout << "\nComputed lowest 10 eigenvalues:" << std::endl;
    for (int i = 0; i < modes.eigenvalues.size(); i++) {
        std::cout << "Mode " << i+1 << ": " << modes.eigenvalues[i] << std::endl;
    }
    
    std::cout << "\nTest passed successfully!" << std::endl;
    return 0;
}
