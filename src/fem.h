#pragma once

#include "geometry.h"
#include <Eigen/Sparse>
#include <vector>

typedef Eigen::SparseMatrix<double> SpMat;
typedef Eigen::Triplet<double> T;

class FEM {
public:
    SpMat K, M;
    std::vector<int> free_dofs;
    
    // Assemble Stiffness and Mass matrices
    void assemble(const Mesh& mesh) {
        int n_nodes = mesh.vertices.size();
        std::vector<T> K_triplets, M_triplets;
        
        // Element Assembly
        for (const auto& el : mesh.elements) {
            // Get coordinates
            const auto& p1 = mesh.vertices[el.v[0]];
            const auto& p2 = mesh.vertices[el.v[1]];
            const auto& p3 = mesh.vertices[el.v[2]];
            
            // Area (Jacobian)
            double x1 = p1.x, y1 = p1.y;
            double x2 = p2.x, y2 = p2.y;
            double x3 = p3.x, y3 = p3.y;
            
            double area = 0.5 * std::abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2));
            
            // Gradients of shape functions (Constant on linear triangle)
            // N1 = (a1 + b1 x + c1 y) / 2A
            // b1 = y2 - y3, c1 = x3 - x2
            double b1 = y2 - y3; double c1 = x3 - x2;
            double b2 = y3 - y1; double c2 = x1 - x3;
            double b3 = y1 - y2; double c3 = x2 - x1;
            
            // Local K (Stiffness) = integral(gradNi . gradNj)
            // = (bi bj + ci cj) / (4A)
            double factor_K = 1.0 / (4.0 * area);
            double loc_K[3][3];
            double b[3] = {b1, b2, b3};
            double c[3] = {c1, c2, c3};
            
            for(int i=0; i<3; ++i) {
                for(int j=0; j<3; ++j) {
                    loc_K[i][j] = factor_K * (b[i]*b[j] + c[i]*c[j]);
                }
            }
            
            // Local M (Mass) = integral(Ni Nj)
            // = A/12 * (1 + delta_ij)
            // Diagonals: A/6, Off-diagonals: A/12
            double factor_M = area / 12.0;
            double loc_M[3][3];
             for(int i=0; i<3; ++i) {
                for(int j=0; j<3; ++j) {
                    loc_M[i][j] = factor_M * (i==j ? 2.0 : 1.0);
                }
            }
            
            // Add to triplets
            for(int i=0; i<3; ++i) {
                for(int j=0; j<3; ++j) {
                    K_triplets.push_back(T(el.v[i], el.v[j], loc_K[i][j]));
                    M_triplets.push_back(T(el.v[i], el.v[j], loc_M[i][j]));
                }
            }
        }
        
        K.resize(n_nodes, n_nodes);
        M.resize(n_nodes, n_nodes);
        K.setFromTriplets(K_triplets.begin(), K_triplets.end());
        M.setFromTriplets(M_triplets.begin(), M_triplets.end());
        
        // Identify Free DOFs
        std::vector<bool> is_fixed(n_nodes, false);
        for(int idx : mesh.boundary_nodes) is_fixed[idx] = true;
        
        free_dofs.clear();
        for(int i=0; i<n_nodes; ++i) {
            if(!is_fixed[i]) free_dofs.push_back(i);
        }
    }
    
    // Helper to get Free-DOF matrices?
    // Spectra needs A and B.
    // Ideally we slice them.
};
