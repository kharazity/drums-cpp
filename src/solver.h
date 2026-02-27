#pragma once

#include <Eigen/Sparse>
#include <Spectra/SymGEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <iostream>

#include "fem.h"

struct ModeData {
    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXd eigenvectors;
};

// Operation for (K - sigma * M)^-1 * M
// Spectra doesn't have a built-in "Generalized Sparse Shift-Invert" op class in 1.0 that takes separate K and M directly in the way I tried.
// We need to form the matrix A = K - sigma * M explicitly, then use SparseSymShiftSolve? 
// Actually, SparseSymShiftSolve solves (A - sigma I) x = y.
// But we have (K - sigma M).
// So we define a custom operation class or just form G = K - sigma * M and solve G x = M y.
// Let's implement a custom operation class that does exactly what we need.

class SparseGenRealShiftSolve {
private:
    const SpMat& m_K;
    const SpMat& m_M;
    double m_sigma;
    Eigen::SimplicialLDLT<SpMat> m_solver; // Or SparseLU

public:
    using Scalar = double;

    SparseGenRealShiftSolve(const SpMat& K, const SpMat& M, double sigma) 
        : m_K(K), m_M(M), m_sigma(sigma) 
    {
        // G = K - sigma * M
        SpMat G = m_K - m_sigma * m_M;
        m_solver.compute(G);
        if(m_solver.info() != Eigen::Success) {
            throw std::runtime_error("Decomposition failed in Shift Solve");
        }
    }

    int rows() const { return m_K.rows(); }
    int cols() const { return m_K.cols(); }

    void set_shift(double sigma) {
        m_sigma = sigma;
        SpMat G = m_K - m_sigma * m_M;
        m_solver.compute(G);
    }

    // y = (K - sigma * M)^-1 * x
    void perform_op(const double* x_in, double* y_out) const {
        Eigen::Map<const Eigen::VectorXd> x(x_in, m_K.rows());
        Eigen::Map<Eigen::VectorXd> y(y_out, m_K.rows());
        y = m_solver.solve(x);
    }
};

class Solver {
public:
    static ModeData solve(const FEM& fem, int n_modes, double sigma = -1.0) {
        int n_free = fem.free_dofs.size();
        if (n_free < n_modes) n_modes = n_free - 1;
        
        // Extract Submatrices (as before)
        std::vector<int> global_to_free(fem.K.rows(), -1);
        for(int i=0; i<n_free; ++i) global_to_free[fem.free_dofs[i]] = i;
        
        std::vector<T> K_triplets, M_triplets;
        K_triplets.reserve(fem.K.nonZeros());
        M_triplets.reserve(fem.M.nonZeros());
        
        // Slicing logic (SAME as before, omitted for brevity if not changing, but I'll include to be safe)
        for (int k=0; k<fem.K.outerSize(); ++k) {
            for (SpMat::InnerIterator it(fem.K, k); it; ++it) {
                int fr = global_to_free[it.row()];
                int fc = global_to_free[it.col()];
                if (fr != -1 && fc != -1) K_triplets.push_back(T(fr, fc, it.value()));
            }
        }
        for (int k=0; k<fem.M.outerSize(); ++k) {
            for (SpMat::InnerIterator it(fem.M, k); it; ++it) {
                int fr = global_to_free[it.row()];
                int fc = global_to_free[it.col()];
                if (fr != -1 && fc != -1) M_triplets.push_back(T(fr, fc, it.value()));
            }
        }
        
        SpMat Kf(n_free, n_free); Kf.setFromTriplets(K_triplets.begin(), K_triplets.end());
        SpMat Mf(n_free, n_free); Mf.setFromTriplets(M_triplets.begin(), M_triplets.end());
        
        // Spectra Operation: custom class
        SparseGenRealShiftSolve op(Kf, Mf, sigma);
        Spectra::SparseSymMatProd<double> Bop(Mf);
        
        // Construct the solver
        Spectra::SymGEigsShiftSolver<SparseGenRealShiftSolve, Spectra::SparseSymMatProd<double>, Spectra::GEigsMode::ShiftInvert> 
            eigs(op, Bop, n_modes, std::min(2*n_modes + 10, n_free), sigma);

        eigs.init();
        int n_conv = eigs.compute(Spectra::SortRule::LargestMagn);
        
        ModeData data;
        Eigen::VectorXd raw_eigenvalues = eigs.eigenvalues();
        Eigen::MatrixXd raw_eigenvectors = eigs.eigenvectors();
        
        // Sort eigenvalues in ascending order
        std::vector<int> indices(n_conv);
        for(int i = 0; i < n_conv; ++i) indices[i] = i;
        
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return raw_eigenvalues(a) < raw_eigenvalues(b);
        });

        data.eigenvalues.resize(n_conv);
        Eigen::MatrixXd free_eigenvectors(raw_eigenvectors.rows(), n_conv);
        for(int i = 0; i < n_conv; ++i) {
            data.eigenvalues(i) = raw_eigenvalues(indices[i]);
            free_eigenvectors.col(i) = raw_eigenvectors.col(indices[i]);
        }
        
        data.eigenvectors = Eigen::MatrixXd::Zero(fem.K.rows(), n_conv);
        
        // Map eigenvectors back to the full node domain (fixed boundary nodes remain 0)
        for (int i = 0; i < n_free; ++i) {
            if (i < free_eigenvectors.rows()) {
                data.eigenvectors.row(fem.free_dofs[i]) = free_eigenvectors.row(i);
            }
        }

        if(eigs.info() != Spectra::CompInfo::Successful) {
             std::cout << "Spectra info: " << (int)eigs.info() << std::endl;
        }
        
        return data;
    }
};
