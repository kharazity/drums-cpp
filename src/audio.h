#pragma once

#include <vector>
#include <cmath>
#include <atomic>
#include <mutex>
#include <cstring>
#include <iostream>
#include <complex>
#include <SDL.h>
#include "solver.h" // For ModeData

struct AudioState {
    std::vector<float> Z_re, Z_im;
    std::vector<float> E_re, E_im;
    std::vector<float> G_re, G_im;
    std::vector<float> pickup;
    std::vector<uint8_t> active; // 1 = active, 0 = inactive
    size_t count = 0;

    void resize(size_t n) {
        Z_re.assign(n, 0.0f);
        Z_im.assign(n, 0.0f);
        E_re.assign(n, 0.0f);
        E_im.assign(n, 0.0f);
        G_re.assign(n, 0.0f);
        G_im.assign(n, 0.0f);
        pickup.assign(n, 0.0f);
        active.assign(n, 0);
        count = n;
    }
};

class AudioEngine {
public:
    std::mutex mutex;
    AudioState active_modes;

    // ─── Physical Constants (matching Julia SynthesisParams defaults) ────────
    double tension   = 200.0;   // T (N/m)
    double rho_s     = 0.26;    // Surface density (kg/m²)
    double alpha0    = 5.0;     // Frequency-independent damping (1/s)
    double alpha1    = 0.0;     // Frequency-proportional damping (s)
    double strike_v0 = 1.0;     // Strike velocity (m/s)
    double strike_width_delta = 0.05; // Spatial Mallet strike width
    
    // Finite Time Force Tracking
    double current_strike_time = 0.0;
    double strike_duration_delta = 0.0;
    
    AudioEngine() {
        active_modes.resize(0); 
    }

    // Wave speed c = sqrt(T / rho_s)
    double wave_speed() const {
        return std::sqrt(std::max(tension, 0.1) / rho_s);
    }

    // Convert raw eigenvalue lambda to physical frequency (Hz)
    double eigenvalue_to_freq(double lambda) const {
        double c = wave_speed();
        return c * std::sqrt(std::max(lambda, 0.0)) / (2.0 * M_PI);
    }

    // Called from Main Thread
    void trigger_strike(const Mesh& mesh, const ModeData& modes, double strike_x, double strike_y, int pickup_node_idx, double force = 1.0) {
        std::lock_guard<std::mutex> lock(mutex);

        int n_modes = modes.eigenvalues.size();
        if ((int)active_modes.count != n_modes) {
            active_modes.resize(n_modes);
        }

        double c = wave_speed();
        const double dt = 1.0 / 44100.0;
        
        // --- Setup Finite Strike Time Window ---
        // A base contact time of ~2 ms is realistic for a mallet.
        // Delta relates inversely to velocity; higher velocity = sharper (shorter) impulse bump
        strike_duration_delta = 0.002 / std::max(strike_v0, 0.01);
        current_strike_time = -strike_duration_delta; // Start integrating from left tail
        
        // The integral of exp(-1/(1-(t/delta)^2)) from -delta to delta is exactly ~0.4439938 * delta.
        // To preserve the total momentum transfer, we scale the force by the inverse of this integral.
        double temporal_bump_integral = 0.4439938 * strike_duration_delta;
        double overall_force = force / temporal_bump_integral;
        
        // --- Pre-compute the spatial distribution weights (bump function) ---
        // F(xj) = exp(-1/(1 - xi^2)) where xi = r / delta_space
        std::vector<double> F(mesh.vertices.size(), 0.0);
        double F_sum = 0.0;
        
        for (size_t j = 0; j < mesh.vertices.size(); ++j) {
            double dx = mesh.vertices[j].x - strike_x;
            double dy = mesh.vertices[j].y - strike_y;
            double r = std::sqrt(dx*dx + dy*dy);
            
            if (r < strike_width_delta) {
                double xi = r / strike_width_delta;
                F[j] = std::exp(-1.0 / (1.0 - xi * xi));
                F_sum += F[j];
            }
        }
        
        // Fallback to closest point if delta is so small it misses all nodes
        if (F_sum == 0.0 && !mesh.vertices.empty()) {
            double min_r = 1e9;
            int closest = 0;
            for (size_t j = 0; j < mesh.vertices.size(); ++j) {
                double dx = mesh.vertices[j].x - strike_x;
                double dy = mesh.vertices[j].y - strike_y;
                double r = dx*dx + dy*dy;
                if (r < min_r) { min_r = r; closest = j; }
            }
            F[closest] = 1.0;
            F_sum = 1.0;
        }

        // Normalize weights
        for (double& w : F) {
            w /= F_sum;
        }

        for (int i = 0; i < n_modes; ++i) {
            if (i >= (int)active_modes.count) break;
            if (pickup_node_idx >= modes.eigenvectors.rows()) continue;

            double lambda_raw = modes.eigenvalues[i];
            double freq_hz = eigenvalue_to_freq(lambda_raw);         // Physical frequency (Hz)
            
            // --- Compute Distributed Spatial Strike Gamma ---
            double gamma = 0.0;
            for (size_t j = 0; j < mesh.vertices.size(); ++j) {
                if (F[j] > 0.0) {
                    gamma += F[j] * modes.eigenvectors(j, i);
                }
            }
            
            double val_pickup = modes.eigenvectors(pickup_node_idx, i); // Microphone amplitude
            
            // Allow modes that aren't physically coupled to the strike to continue decaying undisturbed
            if (std::abs(gamma) < 1e-4) continue;
            
            // --- Compute Complex Physics Constants for ZOH Integration ---
            double decay = alpha0 + alpha1 * freq_hz;
            std::complex<double> lambda(-decay, 2.0 * M_PI * freq_hz);
            
            // E = exp(lambda * dt)
            std::complex<double> E = std::exp(lambda * dt);
            
            // Phasor normalization map from real scalar to complex scalar ODE
            std::complex<double> norm = 1.0 / (lambda - std::conj(lambda));
            
            // G = (E - 1) / lambda * norm * Gamma * force scale
            std::complex<double> G = 0.0;
            if (std::abs(lambda) > 1e-12) {
                G = ((E - 1.0) / lambda) * norm * (gamma * overall_force); 
            } else {
                G = dt * norm * (gamma * overall_force);
            }

            active_modes.Z_re[i] = 0.0f; // Reset initial state for a fresh attack
            active_modes.Z_im[i] = 0.0f;
            active_modes.E_re[i] = (float)E.real();
            active_modes.E_im[i] = (float)E.imag();
            active_modes.G_re[i] = (float)G.real();
            active_modes.G_im[i] = (float)G.imag();
            active_modes.pickup[i] = (float)val_pickup;
            active_modes.active[i] = 1;
        }
    }

    // SDL Audio Callback (real-time thread)
    static void AudioCallback(void* userdata, Uint8* stream, int len) {
        AudioEngine* engine = (AudioEngine*)userdata;
        float* buffer = (float*)stream;
        int samples = len / sizeof(float);

        memset(stream, 0, len);

        std::lock_guard<std::mutex> lock(engine->mutex);

        const double dt = 1.0 / 44100.0;

        for (int i = 0; i < samples; ++i) {
            double sample = 0.0;
            
            // --- Evaluate current temporal bump force f(t) ---
            double f_t = 0.0;
            if (engine->current_strike_time < engine->strike_duration_delta) {
                double xi = engine->current_strike_time / engine->strike_duration_delta;
                if (xi > -1.0 && xi < 1.0) {
                    f_t = std::exp(-1.0 / (1.0 - xi * xi));
                }
                engine->current_strike_time += dt;
            }

            float f_t_f = (float)f_t;

            // --- Optimized SoA SIMD Loop ---
            bool can_deactivate = engine->current_strike_time >= engine->strike_duration_delta;
            
            for (size_t m = 0; m < engine->active_modes.count; ++m) {
                if (!engine->active_modes.active[m]) continue;

                // Load state
                float zr = engine->active_modes.Z_re[m];
                float zi = engine->active_modes.Z_im[m];
                float er = engine->active_modes.E_re[m];
                float ei = engine->active_modes.E_im[m];
                float gr = engine->active_modes.G_re[m];
                float gi = engine->active_modes.G_im[m];

                // Update Phasor State Z = Z*E + G*f(t)
                float next_zr = (zr * er - zi * ei) + gr * f_t_f;
                float next_zi = (zr * ei + zi * er) + gi * f_t_f;

                engine->active_modes.Z_re[m] = next_zr;
                engine->active_modes.Z_im[m] = next_zi;

                // Synthesize Output y = 2 * pickup * Real(Z)
                sample += engine->active_modes.pickup[m] * 2.0 * next_zr;

                // Deactivate dead modes (Squared mag < 1e-16 saves a slow sqrt!)
                if (can_deactivate && (next_zr * next_zr + next_zi * next_zi) < 1e-16f) {
                    engine->active_modes.active[m] = 0;
                }
            }

            // Soft clamp to prevent clipping
            // The physical displacements are very small numerically,
            // so we scale up significantly for audio output.
            sample *= 2000.0; // Master volume
            if (sample > 1.0) sample = 1.0;
            if (sample < -1.0) sample = -1.0;

            buffer[i] = (float)sample;
        }
    }
};
