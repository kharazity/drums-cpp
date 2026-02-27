#include <iostream>
#include <vector>
#include "geometry.h"
#include "fem.h"
#include "solver.h"

int main() {
    for (int density : {5, 10, 20, 40}) {
        Mesh m;
        m.generate_annulus(1.0, 0.5, density);
        FEM f;
        f.assemble(m);
        ModeData md = Solver::solve(f, 5);
        if (md.eigenvalues.size() > 0) {
            std::cout << "Density " << density << " mode 0: " << md.eigenvalues(0) 
                      << " mode " << md.eigenvalues.size()-1 << ": " << md.eigenvalues(md.eigenvalues.size()-1) << std::endl;
        } else {
            std::cout << "Density " << density << " failed." << std::endl;
        }
    }
    return 0;
}
