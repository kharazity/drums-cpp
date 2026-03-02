#pragma once

#include <vector>
#include <cmath>
#include <atomic>
#include <mutex>
#include <cstring>
#include <iostream>
#include <complex>
#include <random>
#include <SDL.h>
#include "solver.h" // For ModeData
#include "fem.h"    // For FEM (mass matrix)
#include <fstream>

// Standard 16-bit PCM WAV Writer
inline bool write_wav_file(const char* filename, const std::vector<float>& samples, int sample_rate) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) return false;

    // Trim silent tail: scan backwards for the last sample above ~-80 dBFS (1e-4 magnitude).
    // This removes the long inaudible exponential decay from the output.
    size_t num_samples = samples.size();
    const float silence_threshold = 1e-4f;
    while (num_samples > 0 && std::fabs(samples[num_samples - 1]) < silence_threshold) {
        --num_samples;
    }
    // Add a small 50ms fade-out tail so the file doesn't click at the end.
    size_t fade_samples = static_cast<size_t>(sample_rate * 0.05);
    num_samples = std::min(num_samples + fade_samples, samples.size());

    // We'll write 16-bit integers
    std::vector<int16_t> pcm_data;
    pcm_data.reserve(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        float v = samples[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        pcm_data.push_back(static_cast<int16_t>(v * 32767.0f));
    }

    uint32_t data_size = pcm_data.size() * sizeof(int16_t);
    uint32_t file_size = 36 + data_size;
    uint32_t sample_rate_u32 = sample_rate;
    uint16_t num_channels = 1;
    uint16_t bits_per_sample = 16;
    uint32_t byte_rate = sample_rate_u32 * num_channels * (bits_per_sample / 8);
    uint16_t block_align = num_channels * (bits_per_sample / 8);

    // RIFF chunk
    file.write("RIFF", 4);
    file.write(reinterpret_cast<const char*>(&file_size), 4);
    file.write("WAVE", 4);

    // fmt chunk
    file.write("fmt ", 4);
    uint32_t fmt_size = 16;
    file.write(reinterpret_cast<const char*>(&fmt_size), 4);
    uint16_t audio_format = 1; // PCM
    file.write(reinterpret_cast<const char*>(&audio_format), 2);
    file.write(reinterpret_cast<const char*>(&num_channels), 2);
    file.write(reinterpret_cast<const char*>(&sample_rate_u32), 4);
    file.write(reinterpret_cast<const char*>(&byte_rate), 4);
    file.write(reinterpret_cast<const char*>(&block_align), 2);
    file.write(reinterpret_cast<const char*>(&bits_per_sample), 2);

    // data chunk
    file.write("data", 4);
    file.write(reinterpret_cast<const char*>(&data_size), 4);
    file.write(reinterpret_cast<const char*>(pcm_data.data()), data_size);

    return true;
}

// ─── Far-Field Radiation Coefficient Set ────────────────────────────────────
// Per-mode coefficients used by the audio callback.
// Three instances exist for lock-free triple-buffering.
static constexpr int MAX_MODES = 2048;

struct CoeffSet {
    float Phi_re[MAX_MODES] = {};   // Radiation weight Φ_n (unconjugated, real part)
    float Phi_im[MAX_MODES] = {};   // Radiation weight Φ_n (unconjugated, imag part)
    float s2_re[MAX_MODES]  = {};   // s_n² real: d_n² - ω_n²
    float s2_im[MAX_MODES]  = {};   // s_n² imag: -2·d_n·ω_n
    float E_re[MAX_MODES]   = {};   // Oscillator transition exp(s·dt) real
    float E_im[MAX_MODES]   = {};   // Oscillator transition exp(s·dt) imag
    float G_re[MAX_MODES]   = {};   // Forcing coefficient real
    float G_im[MAX_MODES]   = {};   // Forcing coefficient imag
    float sB_re[MAX_MODES]  = {};   // Forcing correction s·B real
    float sB_im[MAX_MODES]  = {};   // Forcing correction s·B imag
    float s_re[MAX_MODES]   = {};   // s_n real part: -d_n
    float s_im[MAX_MODES]   = {};   // s_n imag part: ω_n
    
    // Extracted raw physical parameters for per-mode state-dependent updates
    float omega[MAX_MODES]   = {};  // Angular frequency (rad/s)
    float damping[MAX_MODES] = {};  // Damping coefficient (1/s)
    float pickup_psi[MAX_MODES] = {}; // Mode shape magnitude at the pickup location
};

// ─── Modal State (phasor Z, activity flags) ─────────────────────────────────
struct ModalState {
    float Z_re[MAX_MODES] = {};
    float Z_im[MAX_MODES] = {};
    float strike_scale[MAX_MODES] = {};  // Per-mode: gamma * overall_force (set per strike)
    uint8_t active[MAX_MODES] = {};
    int count = 0;
};

// ─── Biquad Filter State ────────────────────────────────────────────────────
struct BiquadCoeffs {
    float b0 = 1.0f;
    float b1 = 0.0f;
    float b2 = 0.0f;
    float a1 = 0.0f;
    float a2 = 0.0f;
};

struct BiquadState {
    float x1 = 0.0f;
    float x2 = 0.0f;
    float y1 = 0.0f;
    float y2 = 0.0f;
    
    float process(float in, const BiquadCoeffs& c) {
        float out = c.b0 * in + c.b1 * x1 + c.b2 * x2 - c.a1 * y1 - c.a2 * y2;
        x2 = x1;
        x1 = in;
        y2 = y1;
        y1 = out;
        return out;
    }
};

// ─── Physical Model Parameters (toggle flags + striker constants) ────────────
struct PhysicalModelParams {
    bool use_contact_model = false;
    bool use_mode_dependent_damping = false;
    bool use_shell_bank = false;

    // Striker / contact
    double striker_mass = 0.02;          // kg
    double striker_stiffness = 5e5;      // K
    double striker_damping = 10.0;       // R (contact damping)
    double striker_exponent = 1.5;       // p (Hertz power-law exponent)
    double striker_initial_velocity = 1.0; // m/s

    // Damping (Milestone 2)
    double edge_loss_weight = 0.0;
    double air_loss_weight = 1.0;

    // Shell Bank (Milestone 3)
    int shell_mode_count = 4;
    double shell_mix = 0.2;
};

// ─── Striker State (single degree of freedom, updated per sample) ────────────
struct StrikerState {
    bool active = false;
    double y = 0.0;   // striker position (positive = into membrane)
    double v = 0.0;   // striker velocity
};

// ─── Shell Resonator Bank (Milestone 3) ─────────────────────────────────────
struct ShellMode {
    float omega = 0.0f;
    float damping = 0.0f;
    float gain = 1.0f;

    float x = 0.0f;   // state (displacement)
    float v = 0.0f;   // state (velocity)
};

struct ShellResonatorBank {
    static constexpr int MAX_SHELL_MODES = 8;
    ShellMode modes[MAX_SHELL_MODES];
    int count = 0;

    void reset() {
        for (int i = 0; i < MAX_SHELL_MODES; ++i) {
            modes[i].x = 0.0f;
            modes[i].v = 0.0f;
        }
    }

    float process(float drive, float dt) {
        float out = 0.0f;
        for (int i = 0; i < count; ++i) {
            ShellMode& m = modes[i];
            // Symplectic Euler
            float force = drive - (m.omega * m.omega * m.x) - (2.0f * m.damping * m.omega * m.v);
            m.v += dt * force;
            m.x += dt * m.v;
            out += m.gain * m.x;
        }
        return out;
    }
};

class AudioEngine {
public:
    std::mutex mutex;               // Only for strike/WAV export (not hot path coefficients)
    std::vector<float> last_strike_audio;

    // ─── Physical Constants ─────────────────────────────────────────────────
    double tension   = 200.0;       // T (N/m)
    double rho_s     = 0.26;        // Surface density (kg/m²)
    double alpha0    = 5.0;         // Frequency-independent damping (1/s)
    double alpha1    = 0.01;        // Frequency-proportional damping coefficient
    double beta      = 1.0;         // Frequency damping power law exponent
    double strike_v0 = 1.0;         // Strike velocity (m/s)
    double strike_width_delta = 0.05; // Spatial mallet strike width (m)
    double strike_duration_ms = 5.0;  // Temporal strike duration (ms)

    // ─── Physical Model Params & Striker ────────────────────────────────────
    PhysicalModelParams phys;
    StrikerState striker;
    float contact_gamma[MAX_MODES] = {};  // Force injection weights (MU-based)
    float contact_psi[MAX_MODES]   = {};  // Displacement readout weights (U-based)

    // ─── Far-Field Radiation Parameters ─────────────────────────────────────
    double c_air = 343.0;           // Speed of sound in air (m/s)
    double rho_air = 1.225;         // Air density (kg/m³)
    double listener_distance = 1.0; // Far-field distance (m) — affects amplitude only
    double listener_elevation = 90.0; // Degrees (90° = directly overhead)
    double listener_azimuth   = 0.0;  // Degrees
    double master_volume = 10.0;    // User-adjustable master volume
    
    // ─── Pickup Mix Parameters (Displacement, Velocity, Acceleration) ───────
    double pickup_x = 0.0;          // Pickup coordinate X
    double pickup_y = 0.0;          // Pickup coordinate Y
    double pickup_radius = 0.02;    // Pickup radius for area-weighted sampling
    double mix_vel  = 0.65;         // Velocity mix ratio
    double mix_accel = 0.25;        // Far-field acceleration mix ratio
    double mix_disp  = 0.10;        // Displacement mix ratio

    // ─── Stochastic Frequency Detuning ──────────────────────────────────────
    double detune_amount = 0.005;   // Base detuning amount (e.g., 0.005 for 0.5%)
    std::vector<double> mode_detune_factors; // Per-mode uniformly distributed [-1, 1]

    // ─── Shell Resonance Filter Parameters ──────────────────────────────────
    double shell_freq = 200.0;      // Resonance Frequency (Hz) (Old Biquad)
    double shell_q = 5.0;           // Q Factor
    double shell_gain_db = 0.0;     // Gain (dB)
    ShellResonatorBank shell_bank;  // New Multi-mode Resonator Bank

    double peak_envelope = 0.0;     // Auto-gain peak tracker (callback-only)
    bool diagnosis_mode = false;    // Bypass nonlinear stages and unity gain master volume

    // ─── Precomputed Geometry ───────────────────────────────────────────────
    Eigen::MatrixXd MU;             // M·U (J × N_modes), precomputed at mesh rebuild
    const Eigen::MatrixXd* U_ptr = nullptr; // Pointer to eigenvectors (NOT owned)
    std::vector<double> xi;         // Phase-centered ξ_j = n_∥·s_j - mean(ξ)
    int n_modes_cached = 0;

    // ─── Mode-Dependent Damping Data (Milestone 2) ──────────────────────────
    std::vector<int> boundary_nodes;
    float modal_edge_participation[MAX_MODES] = {};
    float modal_air_participation[MAX_MODES] = {};

    // ─── Tension Modulation Grid ─────────────────────────────────────────────
    static constexpr int TENSION_GRID_SIZE = 3;
    static constexpr float TENSION_GRID_VALS[TENSION_GRID_SIZE] = {0.95f, 1.0f, 1.05f};
    static constexpr int BASE_TENSION_IDX = 1;   // index of multiplier 1.0

    // Double-buffered grid: two buffers each containing TENSION_GRID_SIZE CoeffSet objects
    CoeffSet tension_coeffs[2][TENSION_GRID_SIZE];
    std::atomic<int> active_tension_grid{0};

    // User parameter: nonlinearity (0 = off) and smoothed energy
    float nonlinearity = 0.0f;
    float energy_smooth = 0.0f;
    float energy_alpha = 0.0001f;   // smoothing factor (can be fixed)

    // Double-buffered pickup weights (tension-independent)
    float pickup_psi_buffers[2][MAX_MODES];
    std::atomic<int> active_pickup_idx{0};

    // Multi-mode filter for entire drum body resonance
    BiquadCoeffs filter_coeff[3];
    std::atomic<int> cur_filter_idx{0};
    std::atomic<int> fade_filter_idx{1};
    std::atomic<bool> filter_update_ready{false};
    BiquadState cur_filter_state;
    BiquadState fade_filter_state;

    // ─── Crossfade State (only touched by callback) ─────────────────────────
    bool fading_filter = false;
    float fade_filter_alpha = 0.0f;
    static constexpr float FADE_DURATION_S = 0.02f; // 20ms crossfade

    // ─── Modal State (shared, protected by mutex for strike init) ───────────
    ModalState modes_state;

    // Finite Time Force Tracking
    double current_strike_time = 0.0;
    double strike_duration_delta = 0.0;

    AudioEngine() {}

    // Wave speed c = sqrt(T / rho_s)
    double wave_speed() const {
        return std::sqrt(std::max(tension, 0.1) / rho_s);
    }

    // Convert raw eigenvalue Λ to physical frequency (Hz)
    double eigenvalue_to_freq(double lambda) const {
        double c = wave_speed();
        return c * std::sqrt(std::max(lambda, 0.0)) / (2.0 * M_PI);
    }

    // ─── Precompute MU = M · eigenvectors (call on mesh rebuild) ────────────
    void precompute_MU(const FEM& fem, const ModeData& modes) {
        int n_modes = modes.eigenvalues.size();
        int J = modes.eigenvectors.rows();
        n_modes_cached = n_modes;
        U_ptr = &modes.eigenvectors;  // Store pointer for contact readout later

        // Single sparse×dense multiply: MU = M * U
        MU = fem.M * modes.eigenvectors;  // (J × N_modes)

        // Mass-normalization check: ensure u_n^T M u_n = 1
        for (int n = 0; n < n_modes; ++n) {
            double norm_sq = MU.col(n).dot(modes.eigenvectors.col(n));
            if (std::abs(norm_sq - 1.0) > 1e-6 && norm_sq > 1e-12) {
                double s = 1.0 / std::sqrt(norm_sq);
                // Note: we can't modify modes.eigenvectors (const), so we scale MU
                // and rely on consistent usage. For strike projection via MU, this is fine.
                MU.col(n) *= s;
            }
        }

        // Generate reproducible stochastic mode detuning factors
        mode_detune_factors.resize(n_modes);
        std::mt19937 gen(42); // Fixed seed for reproducibility when rebuilding
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (int n = 0; n < n_modes; ++n) {
            mode_detune_factors[n] = dist(gen);
        }
    }

    // ─── Compute ξ_j (call on init and direction change) ────────────────────
    void compute_xi(const Mesh& mesh) {
        double theta_rad = listener_elevation * M_PI / 180.0;
        double phi_rad   = listener_azimuth   * M_PI / 180.0;
        double nx = std::cos(theta_rad) * std::cos(phi_rad);
        double ny = std::cos(theta_rad) * std::sin(phi_rad);

        int J = mesh.vertices.size();
        xi.resize(J);

        double xi_sum = 0.0;
        for (int j = 0; j < J; ++j) {
            xi[j] = nx * mesh.vertices[j].x + ny * mesh.vertices[j].y;
            xi_sum += xi[j];
        }

        // Phase centering: subtract mean to remove global phase offset
        double xi_mean = xi_sum / J;
        for (int j = 0; j < J; ++j) {
            xi[j] -= xi_mean;
        }
    }

    // ─── Compute pickup weights (call on pickup position change) ─────────────
    // Writes into staging buffer. Sets active_pickup_idx for callback to pick up.
    void compute_pickup_weights(const Mesh& mesh, const ModeData& modes) {
        int J = mesh.vertices.size();
        if (J == 0) return;

        int n_modes = modes.eigenvalues.size();
        if (n_modes > MAX_MODES) n_modes = MAX_MODES;
        
        int active = active_pickup_idx.load(std::memory_order_acquire);
        int next = 1 - active;

        // 1. Compute nodal areas (1/3 of adjacent triangle areas)
        std::vector<double> nodal_area(J, 0.0);
        for (const auto& tri : mesh.elements) {
            double x1 = mesh.vertices[tri.v[0]].x, y1 = mesh.vertices[tri.v[0]].y;
            double x2 = mesh.vertices[tri.v[1]].x, y2 = mesh.vertices[tri.v[1]].y;
            double x3 = mesh.vertices[tri.v[2]].x, y3 = mesh.vertices[tri.v[2]].y;
            double area = 0.5 * std::abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2));
            double node_weight = area / 3.0;
            nodal_area[tri.v[0]] += node_weight;
            nodal_area[tri.v[1]] += node_weight;
            nodal_area[tri.v[2]] += node_weight;
        }

        // 2. Find vertices within pickup_radius and compute total area
        double total_area = 0.0;
        std::vector<int> active_nodes;
        
        for (int j = 0; j < J; ++j) {
            double dx = mesh.vertices[j].x - pickup_x;
            double dy = mesh.vertices[j].y - pickup_y;
            double r = std::sqrt(dx*dx + dy*dy);
            if (r <= pickup_radius) {
                total_area += nodal_area[j];
                active_nodes.push_back(j);
            }
        }

        // Fallback to closest vertex if none found within radius
        if (active_nodes.empty()) {
            double min_r = 1e9;
            int closest = 0;
            for (int j = 0; j < J; ++j) {
                double dx = mesh.vertices[j].x - pickup_x;
                double dy = mesh.vertices[j].y - pickup_y;
                double r = dx*dx + dy*dy;
                if (r < min_r) { min_r = r; closest = j; }
            }
            active_nodes.push_back(closest);
            total_area = nodal_area[closest];
        }

        // 3. Compute area-weighted average for each mode
        for (int n = 0; n < n_modes; ++n) {
            double weighted_sum = 0.0;
            for (int j : active_nodes) {
                weighted_sum += modes.eigenvectors(j, n) * nodal_area[j];
            }
            if (total_area > 0.0) {
                pickup_psi_buffers[next][n] = (float)(weighted_sum / total_area);
            } else {
                pickup_psi_buffers[next][n] = 0.0f;
            }
        }
        
        // Signal the audio thread that the new buffer is ready
        active_pickup_idx.store(next, std::memory_order_release);
    }

    // ─── Compute radiation weights (call on tension/direction change) ────────
    // Writes into staging buffer. Sets update_ready for callback to pick up.
    void compute_coeffs_for_tension(double base_tension, const ModeData& modes, CoeffSet& out) {
        double c = std::sqrt(std::max(base_tension, 0.1) / rho_s);
        const double dt = 1.0 / 44100.0;
        const double nyquist = 0.45 * 44100.0;
        int J = MU.rows();
        int n_modes = std::min((int)modes.eigenvalues.size(), MAX_MODES);

        for (int n = 0; n < n_modes; ++n) {
            double Lambda_n = modes.eigenvalues[n];
            // Apply per-mode stochastic frequency detuning
            double detune_mult = 1.0;
            if (n < (int)mode_detune_factors.size()) {
                detune_mult += mode_detune_factors[n] * detune_amount;
            }
            double omega_n = c * std::sqrt(std::max(Lambda_n, 0.0)) * detune_mult;
            double freq_hz = omega_n / (2.0 * M_PI);

            // Nyquist guard
            if (freq_hz > nyquist) {
                out.E_re[n] = out.E_im[n] = 0.f;
                out.G_re[n] = out.G_im[n] = 0.f;
                out.s2_re[n] = out.s2_im[n] = 0.f;
                out.sB_re[n] = out.sB_im[n] = 0.f;
                out.Phi_re[n] = out.Phi_im[n] = 0.f;
                out.s_re[n] = out.s_im[n] = 0.f;
                continue;
            }

            // Damping – use existing build_mode_damping
            double d_n = build_mode_damping(n, omega_n, freq_hz);
            d_n = std::min(d_n, 0.95 * omega_n);

            out.omega[n] = (float)omega_n;
            out.damping[n] = (float)d_n;

            // s = -d + iω
            out.s_re[n] = (float)(-d_n);
            out.s_im[n] = (float)(omega_n);

            // s²
            out.s2_re[n] = (float)(d_n * d_n - omega_n * omega_n);
            out.s2_im[n] = (float)(-2.0 * d_n * omega_n);

            // E = exp(s·dt)
            double exp_decay = std::exp(-d_n * dt);
            out.E_re[n] = (float)(exp_decay * std::cos(omega_n * dt));
            out.E_im[n] = (float)(exp_decay * std::sin(omega_n * dt));

            // G and sB
            std::complex<double> s_n(-d_n, omega_n);
            std::complex<double> E_n(out.E_re[n], out.E_im[n]);
            std::complex<double> norm = 1.0 / (s_n - std::conj(s_n));
            std::complex<double> G_base = (std::abs(s_n) > 1e-12) ? ((E_n - 1.0) / s_n) * norm : dt * norm;
            out.G_re[n] = (float)G_base.real();
            out.G_im[n] = (float)G_base.imag();

            std::complex<double> sB = s_n * norm;
            out.sB_re[n] = (float)sB.real();
            out.sB_im[n] = (float)sB.imag();

            // Radiation weight Φ
            double k_n = omega_n / c_air;
            double phi_re = 0.0, phi_im = 0.0;
            for (int j = 0; j < J; ++j) {
                double angle = k_n * xi[j];
                double cos_a = std::cos(angle);
                double sin_a = std::sin(angle);
                double mu_jn = MU(j, n);
                phi_re += mu_jn * cos_a;
                phi_im += mu_jn * sin_a;
            }
            out.Phi_re[n] = (float)phi_re;
            out.Phi_im[n] = (float)phi_im;
        }
    }

    void rebuild_tension_grid(const Mesh& mesh, const ModeData& modes) {
        int J = mesh.vertices.size();
        if (J == 0) return;

        // Populate modal participations (requires MU, precomputed in precompute_MU)
        int n_modes = std::min((int)modes.eigenvalues.size(), MAX_MODES);
        for (int n = 0; n < n_modes; ++n) {
            double I_vol = 0.0;
            for (int j = 0; j < J; ++j) {
                I_vol += MU(j, n);
            }
            modal_air_participation[n] = (float)I_vol;
        }

        // Get the inactive buffer index
        int active = active_tension_grid.load(std::memory_order_acquire);
        int next = 1 - active;

        // Build a CoeffSet for each tension multiplier
        int current_pickup = active_pickup_idx.load(std::memory_order_acquire);
        for (int j = 0; j < TENSION_GRID_SIZE; ++j) {
            double grid_tension = tension * TENSION_GRID_VALS[j];
            compute_coeffs_for_tension(grid_tension, modes, tension_coeffs[next][j]);
            
            // Mirror current pickup weights into the tension grid
            for (int n = 0; n < n_modes; ++n) {
                tension_coeffs[next][j].pickup_psi[n] = pickup_psi_buffers[current_pickup][n];
            }
        }

        active_tension_grid.store(next, std::memory_order_release);
    }

    // ─── Compute Biquad Filter Coefficients (called on UI param change) ─────
    void compute_filter_coeffs() {

        const double fs = 44100.0;
        double w0 = 2.0 * M_PI * shell_freq / fs;
        double R = std::exp(-M_PI * shell_freq / (std::max(shell_q, 0.001) * fs));

        // Two-pole resonator coefficients for H_res(z) = 1 / (1 + a1 z^-1 + a2 z^-2)
        double a1 = -2.0 * R * std::cos(w0);
        double a2 = R * R;

        // The peak gain of the raw resonator is approximately 1 / (1 - R).
        // We want the total filter H_total = 1 + c * H_res to have a peak gain
        // of 10^(gain_db / 20) at resonance.
        // So 1 + c / (1 - R) = 10^(gain_db / 20) => c = (10^(gain_db / 20) - 1) * (1 - R).
        double linear_boost = std::pow(10.0, shell_gain_db / 20.0);
        double c = (linear_boost - 1.0) * (1.0 - R);

        BiquadCoeffs& stg = filter_coeff[2];

        // Parallel formulation gives mathematically perfect zero-phase pass-through when c=0
        stg.b0 = (float)(1.0 + c);
        stg.b1 = (float)a1;
        stg.b2 = (float)a2;
        stg.a1 = (float)a1;
        stg.a2 = (float)a2;

        filter_update_ready.store(true, std::memory_order_release);
    }

    // ─── Reset all modal activity cleanly ────────────────────────────────────
    void reset_modal_state() {
        for (int i = 0; i < MAX_MODES; ++i) {
            modes_state.Z_re[i] = 0.0f;
            modes_state.Z_im[i] = 0.0f;
            modes_state.strike_scale[i] = 0.0f;
            modes_state.active[i] = 0;
            contact_gamma[i] = 0.0f;
            contact_psi[i] = 0.0f;
        }
        striker.active = false;
        striker.y = 0.0;
        striker.v = 0.0;
        current_strike_time = strike_duration_delta; // disable old bump
    }

    // ─── Precompute strike coupling arrays ──────────────────────────────────
    // gamma_force[n]: mass-consistent force projection (MU)
    // psi_contact[n]: mode-shape displacement readout (MU)
    void prepare_strike_coupling(
        const Mesh& mesh,
        const ModeData& modes,
        double strike_x,
        double strike_y
    ) {
        int n_modes = modes.eigenvalues.size();
        if (n_modes > MAX_MODES) n_modes = MAX_MODES;
        int J = mesh.vertices.size();

        // 1. Compute nodal areas recursively (1/3 of adjacent triangle areas)
        std::vector<double> nodal_area(J, 0.0);
        for (const auto& tri : mesh.elements) {
            double x1 = mesh.vertices[tri.v[0]].x, y1 = mesh.vertices[tri.v[0]].y;
            double x2 = mesh.vertices[tri.v[1]].x, y2 = mesh.vertices[tri.v[1]].y;
            double x3 = mesh.vertices[tri.v[2]].x, y3 = mesh.vertices[tri.v[2]].y;
            double area = 0.5 * std::abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2));
            double node_weight = area / 3.0;
            nodal_area[tri.v[0]] += node_weight;
            nodal_area[tri.v[1]] += node_weight;
            nodal_area[tri.v[2]] += node_weight;
        }

        // 2. Build spatial bump shape F[j]
        std::vector<double> F(J, 0.0);
        std::vector<int> active_nodes;
        double F_integral = 0.0;

        for (int j = 0; j < J; ++j) {
            double dx = mesh.vertices[j].x - strike_x;
            double dy = mesh.vertices[j].y - strike_y;
            double r = std::sqrt(dx*dx + dy*dy);
            if (r < strike_width_delta) {
                double xi_r = r / strike_width_delta;
                F[j] = std::exp(-1.0 / (1.0 - xi_r * xi_r));
                F_integral += F[j] * nodal_area[j];
                active_nodes.push_back(j);
            }
        }

        // Fallback to closest node
        if (F_integral == 0.0 && J > 0) {
            double min_r = 1e9;
            int closest = 0;
            for (int j = 0; j < J; ++j) {
                double dx = mesh.vertices[j].x - strike_x;
                double dy = mesh.vertices[j].y - strike_y;
                double r = dx*dx + dy*dy;
                if (r < min_r) { min_r = r; closest = j; }
            }
            F[closest] = 1.0;
            F_integral = nodal_area[closest];
            active_nodes.push_back(closest);
        }

        // 3. Normalize F to be a proper force density (integral F dA = 1)
        if (F_integral > 0.0) {
            for (int j : active_nodes) {
                F[j] /= F_integral;
            }
        }

        // 4. Project onto modes using Galerkin area-weighting (MU)
        for (int n = 0; n < n_modes; ++n) {
            double gamma = 0.0;
            for (int j : active_nodes) {
                // MU is M * U. So F^T MU computes the discretized integral of F * U dA
                gamma += MU(j, n) * F[j];   
            }
            // By Maxwell-Betti reciprocity, displacement readout is perfectly symmetric with force projection
            contact_gamma[n] = (float)gamma;
            contact_psi[n]   = (float)gamma;
        }
    }

    // ─── Mode and Boundary Precomputation (Milestone 2) ─────────────────────
    void compute_mode_descriptors(const Mesh& mesh, const ModeData& modes) {
        // 1. Find boundary nodes (nodes appearing on fewer than 2 * expected element borders, or just use explicitly marked ones if available)
        // For our simple 2D meshes, we can find nodes shared by edges that only belong to 1 triangle.
        
        struct Edge {
            int v1, v2;
            bool operator==(const Edge& o) const {
                return (v1 == o.v1 && v2 == o.v2) || (v1 == o.v2 && v2 == o.v1);
            }
        };
        std::vector<Edge> all_edges;
        for (const auto& tri : mesh.elements) {
            all_edges.push_back({tri.v[0], tri.v[1]});
            all_edges.push_back({tri.v[1], tri.v[2]});
            all_edges.push_back({tri.v[2], tri.v[0]});
        }
        
        std::vector<int> edge_counts(all_edges.size(), 0);
        for (size_t i = 0; i < all_edges.size(); ++i) {
            for (size_t j = 0; j < all_edges.size(); ++j) {
                if (all_edges[i] == all_edges[j]) edge_counts[i]++;
            }
        }
        
        std::vector<int> bnd_nodes;
        for (size_t i = 0; i < all_edges.size(); ++i) {
            if (edge_counts[i] == 1) { // Boundary edge connects 2 boundary nodes
                bnd_nodes.push_back(all_edges[i].v1);
                bnd_nodes.push_back(all_edges[i].v2);
            }
        }
        
        // Remove duplicates
        std::sort(bnd_nodes.begin(), bnd_nodes.end());
        bnd_nodes.erase(std::unique(bnd_nodes.begin(), bnd_nodes.end()), bnd_nodes.end());
        boundary_nodes = bnd_nodes;

        // 2. Compute E_n^{edge}
        const Eigen::MatrixXd& U = modes.eigenvectors;
        int n_modes = modes.eigenvalues.size();
        if (n_modes > MAX_MODES) n_modes = MAX_MODES;

        for (int n = 0; n < n_modes; ++n) {
            double edge_energy = 0.0;
            for (int j : boundary_nodes) {
                double u = U(j, n);
                edge_energy += u * u;
            }
            // Normalize somewhat by number of boundary nodes to keep scale sane
            if (!boundary_nodes.empty()) {
                edge_energy /= boundary_nodes.size();
            }
            modal_edge_participation[n] = (float)edge_energy;
        }
    }

    double build_mode_damping(int n, double omega_n, double freq_hz) {
        double d_base = alpha0 + alpha1 * std::pow(freq_hz, beta);
        
        double I_vol = modal_air_participation[n];
        double d_rad = (rho_air * omega_n * omega_n * I_vol * I_vol) / (8.0 * M_PI * c_air * rho_s);

        if (phys.use_mode_dependent_damping) {
            double d_air  = phys.air_loss_weight * d_rad;
            // E_edge is inherently quite small (average mode amp squared), apply a scaling factor
            // A suitable scaling factor makes the tuning slider intuitive (0 -> 100 range)
            double edge_scale = 1000.0; 
            double d_edge = phys.edge_loss_weight * modal_edge_participation[n] * edge_scale;
            return d_base + d_air + d_edge;
        } else {
            return d_base;
        }
    }

    // ─── Centralized coefficient rebuild ────────────────────────────────────
    void rebuild_physical_coeffs(const Mesh& mesh, const ModeData& modes) {
        compute_mode_descriptors(mesh, modes);
        compute_xi(mesh);
        rebuild_tension_grid(mesh, modes);
    }

    // ─── Strike (called from UI thread) ─────────────────────────────────────
    void trigger_strike(const Mesh& mesh, const ModeData& modes, double strike_x, double strike_y, double force = 1.0) {
        std::lock_guard<std::mutex> lock(mutex);

        int n_modes = modes.eigenvalues.size();
        if (n_modes > MAX_MODES) n_modes = MAX_MODES;
        modes_state.count = n_modes;

        // Clear the buffer used for WAV export
        last_strike_audio.clear();

        // Fully reset all modal state to prevent stale leakage
        reset_modal_state();

        // Initialize shell bank with actual drum modes from the staging buffer
        // This is explicitly done per-strike so that tuning changes do not pop the ongoing acoustic ring.
        shell_bank.count = std::min(4, modes_state.count);
        for (int k = 0; k < shell_bank.count; ++k) {
            float om = tension_coeffs[active_tension_grid.load(std::memory_order_acquire)][BASE_TENSION_IDX].omega[k];
            float gain = 1.0f - (k * 0.2f); 
            float damp = 0.05f - (k * 0.01f);
            
            shell_bank.modes[k].omega = om;
            shell_bank.modes[k].damping = damp;
            shell_bank.modes[k].gain = gain;
            shell_bank.modes[k].x = 0.0f;
            shell_bank.modes[k].v = 0.0f;
        }

        // Precompute spatial coupling for this strike location
        prepare_strike_coupling(mesh, modes, strike_x, strike_y);

        if (phys.use_contact_model) {
            // --- Contact Model: initialize striker DOF ---
            striker.active = true;
            striker.y = 0.0;
            striker.v = phys.striker_initial_velocity * force;

            // Activate all coupled modes
            for (int i = 0; i < n_modes; ++i) {
                if (std::abs(contact_gamma[i]) > 1e-6f) {
                    modes_state.active[i] = 1;
                }
            }
        } else {
            // --- Old prescribed bump model ---
            strike_duration_delta = std::max(0.1, strike_duration_ms) / 1000.0;
            current_strike_time = -strike_duration_delta;

            double temporal_bump_integral = 0.4439938 * strike_duration_delta;
            double overall_force = force / temporal_bump_integral;

            for (int i = 0; i < n_modes; ++i) {
                if (std::abs(contact_gamma[i]) < 1e-4f) continue;
                modes_state.strike_scale[i] = (float)(contact_gamma[i] * overall_force);
                modes_state.active[i] = 1;
            }
        }
    }

    // ─── SDL Audio Callback (real-time thread) ──────────────────────────────
    static void AudioCallback(void* userdata, Uint8* stream, int len) {
        AudioEngine* engine = (AudioEngine*)userdata;
        float* buffer = (float*)stream;
        int samples = len / sizeof(float);

        memset(stream, 0, len);

        // Try to acquire lock. If UI thread is busy (e.g., rebuilding mesh or processing 
        // a new strike), we do NOT block the audio thread. We just output the zeroed buffer.
        std::unique_lock<std::mutex> lock(engine->mutex, std::try_to_lock);
        if (!lock.owns_lock()) {
            return;
        }

        const double dt = 1.0 / 44100.0;
        const float alpha_step = 1.0f / (FADE_DURATION_S * 44100.0f);

        // --- Snapshot indices once per block ---
        int ci = engine->active_tension_grid.load(std::memory_order_acquire);
        int pi_idx = engine->active_pickup_idx.load(std::memory_order_acquire);
        auto& grid = engine->tension_coeffs[ci];
        auto& pickup = engine->pickup_psi_buffers[pi_idx];

        // --- Check for new filter update (once per block) ---
        int filter_ci = engine->cur_filter_idx.load(std::memory_order_acquire);
        int filter_fi = engine->fade_filter_idx.load(std::memory_order_relaxed);

        if (engine->filter_update_ready.load(std::memory_order_acquire)) {
            // Unconditionally adopt the newest target coefficients
            std::memcpy(&engine->filter_coeff[filter_fi], &engine->filter_coeff[2], sizeof(BiquadCoeffs));
            engine->filter_update_ready.store(false, std::memory_order_release);
            
            // Only prime a new fade cycle if we aren't already crossfading.
            // If we are actively crossfading, simply altering the target coefficients 
            // allows the slider value to track smoothly without interrupting the ongoing alpha interpolation.
            if (!engine->fading_filter) {
                engine->fading_filter = true;
                engine->fade_filter_alpha = 0.0f;
                // Prime fade filter state to match current filter state to avoid popping early
                engine->fade_filter_state = engine->cur_filter_state;
            }
        }

        const BiquadCoeffs& cur_filter_c = engine->filter_coeff[filter_ci];
        const BiquadCoeffs& fade_filter_c = engine->filter_coeff[filter_fi];

        for (int i = 0; i < samples; ++i) {
            double sample = 0.0;

            // --- Compute Total Modale Energy (E = w^2 * z^2 + v^2) ---
            float current_energy = 0.0f;
            for (int m = 0; m < engine->modes_state.count; ++m) {
                if (engine->modes_state.active[m]) {
                    float zr = engine->modes_state.Z_re[m];
                    float zi = engine->modes_state.Z_im[m];
                    // omega from the base tension grid
                    float om = grid[BASE_TENSION_IDX].omega[m];
                    float z_mag2 = zr*zr + zi*zi;
                    // approx modal energy ~ omega^2 * |Z|^2  (since Z is scaled phasor)
                    current_energy += z_mag2 * om * om; 
                }
            }
            
            // Exponential smoothing to avoid audio-rate zipper noise on the pitch drop
            engine->energy_smooth = engine->energy_smooth + engine->energy_alpha * (current_energy - engine->energy_smooth);

            // --- Map Smoothed Energy to Tension Multiplier ---
            // Base multiplier is 1.0. Adds (nonlinearity * energy)
            float tension_mult = 1.0f + engine->nonlinearity * engine->energy_smooth;
            
            // Clamp to our precomputed grid bounds
            float min_mult = TENSION_GRID_VALS[0];
            float max_mult = TENSION_GRID_VALS[TENSION_GRID_SIZE - 1];
            if (tension_mult < min_mult) tension_mult = min_mult;
            if (tension_mult > max_mult) tension_mult = max_mult;

            // --- Find Grid Indices for Interpolation ---
            int idx_low = 0;
            int idx_high = 1;
            for (int g = 0; g < TENSION_GRID_SIZE - 1; ++g) {
                if (tension_mult >= TENSION_GRID_VALS[g] && tension_mult <= TENSION_GRID_VALS[g + 1]) {
                    idx_low = g;
                    idx_high = g + 1;
                    break;
                }
            }

            float val_low = TENSION_GRID_VALS[idx_low];
            float val_high = TENSION_GRID_VALS[idx_high];
            float t = (tension_mult - val_low) / (val_high - val_low);

            const CoeffSet& low = grid[idx_low];
            const CoeffSet& high = grid[idx_high];

            // ─── Determine per-sample forcing ────────────────────────────
            float Fc = 0.0f;  // Contact force (used only in contact model)
            float f_t_f = 0.0f; // Old bump force (used only in prescribed bump)
            bool can_deactivate = true;

            if (engine->phys.use_contact_model) {
                // --- Contact model: compute w_c and w_dot_c from modal state ---
                if (engine->striker.active) {
                    double w_contact = 0.0;
                    double w_contact_vel = 0.0;
                    for (int m = 0; m < engine->modes_state.count; ++m) {
                        if (!engine->modes_state.active[m]) continue;
                        float zr = engine->modes_state.Z_re[m];
                        float zi = engine->modes_state.Z_im[m];
                        float psi = engine->contact_psi[m];
                        // q_n ≈ 2·Re(Z_n)
                        w_contact += 2.0 * (double)(zr * psi);
                        // q_dot_n ≈ 2·Re(s_n · Z_n)
                        
                        // Interpolate s for velocity
                        float sr = (1-t) * low.s_re[m] + t * high.s_re[m];
                        float si = (1-t) * low.s_im[m] + t * high.s_im[m];
                        
                        float sZ_re = sr * zr - si * zi;
                        w_contact_vel += 2.0 * (double)(sZ_re * psi);
                    }

                    double delta = engine->striker.y - w_contact;
                    double delta_dot = engine->striker.v - w_contact_vel;

                    if (delta > 0.0) {
                        double delta_p = std::pow(delta, engine->phys.striker_exponent);
                        double force_val = engine->phys.striker_stiffness * delta_p
                                         + engine->phys.striker_damping * delta_p * delta_dot;
                        Fc = (float)std::max(0.0, force_val);
                    }

                    // Update striker dynamics (symplectic Euler)
                    engine->striker.v -= dt * (double)Fc / engine->phys.striker_mass;
                    engine->striker.y += dt * engine->striker.v;

                    // Deactivation: striker has rebounded and separated
                    if (Fc == 0.0f && engine->striker.v <= 0.0 && engine->striker.y <= 0.0) {
                        engine->striker.active = false;
                    }
                }
                can_deactivate = !engine->striker.active;
            } else {
                // --- Old prescribed bump ---
                if (engine->current_strike_time < engine->strike_duration_delta) {
                    double xi_t = engine->current_strike_time / engine->strike_duration_delta;
                    if (xi_t > -1.0 && xi_t < 1.0) {
                        f_t_f = (float)std::exp(-1.0 / (1.0 - xi_t * xi_t));
                    }
                    engine->current_strike_time += dt;
                }
                can_deactivate = engine->current_strike_time >= engine->strike_duration_delta;
            }

            // ─── Modal evolution loop ────────────────────────────────────
            double sum_radiation = 0.0;
            double sum_disp = 0.0;
            double sum_vel = 0.0;

            for (int m = 0; m < engine->modes_state.count; ++m) {
                if (!engine->modes_state.active[m]) continue;

                float zr = engine->modes_state.Z_re[m];
                float zi = engine->modes_state.Z_im[m];

                // Determine per-mode forcing
                float forcing;
                if (engine->phys.use_contact_model) {
                    forcing = engine->contact_gamma[m] * Fc;
                } else {
                    forcing = engine->modes_state.strike_scale[m] * f_t_f;
                }

                // Interpolate E and G
                float E_re = (1-t) * low.E_re[m] + t * high.E_re[m];
                float E_im = (1-t) * low.E_im[m] + t * high.E_im[m];
                float G_re = (1-t) * low.G_re[m] + t * high.G_re[m];
                float G_im = (1-t) * low.G_im[m] + t * high.G_im[m];

                // Evolution: Z' = E·Z + G·forcing
                float next_zr = (zr * E_re - zi * E_im) + G_re * forcing;
                float next_zi = (zr * E_im + zi * E_re) + G_im * forcing;

                engine->modes_state.Z_re[m] = next_zr;
                engine->modes_state.Z_im[m] = next_zi;

                // ---- Pickup Extraction (Displacement and Velocity) ----
                float psi = pickup[m];
                float disp_n = 2.0f * next_zr;
                
                float s_re = (1-t) * low.s_re[m] + t * high.s_re[m];
                float s_im = (1-t) * low.s_im[m] + t * high.s_im[m];
                float vel_n  = 2.0f * (s_re * next_zr - s_im * next_zi);
                
                sum_disp += disp_n * psi;
                sum_vel  += vel_n * psi;

                // ---- Far-Field Extraction (Acceleration) ----
                // Acceleration: a = s²·Z
                float s2_re = (1-t) * low.s2_re[m] + t * high.s2_re[m];
                float s2_im = (1-t) * low.s2_im[m] + t * high.s2_im[m];
                float ar = s2_re * next_zr - s2_im * next_zi;
                float ai = s2_re * next_zi + s2_im * next_zr;

                // Forcing correction during active forcing window
                if (forcing != 0.0f) {
                    float sB_re = (1-t) * low.sB_re[m] + t * high.sB_re[m];
                    float sB_im = (1-t) * low.sB_im[m] + t * high.sB_im[m];
                    ar += sB_re * forcing;
                    ai += sB_im * forcing;
                }

                // Radiation mapping
                float pr = (1-t) * low.Phi_re[m] + t * high.Phi_re[m];
                float pi = (1-t) * low.Phi_im[m] + t * high.Phi_im[m];

                // sum_radiation = Σ 2·Re(conj(Φ)·a)
                sum_radiation += 2.0 * (pr * ar + pi * ai);

                // Deactivate dead modes
                if (can_deactivate && (next_zr * next_zr + next_zi * next_zi) < 1e-16f) {
                    engine->modes_state.active[m] = 0;
                    engine->modes_state.Z_re[m] = 0.0f;
                    engine->modes_state.Z_im[m] = 0.0f;
                }
            }

            // Apply spatial scaling to the far-field term
            sum_radiation *= engine->rho_air / (2.0 * M_PI * engine->listener_distance);

            // Mix down final output using triple-mix coefficients
            sample = (engine->mix_disp * sum_disp) 
                   + (engine->mix_vel * sum_vel) 
                   + (engine->mix_accel * sum_radiation);

            // --- Apply Shell Processing (A/B testing block) ---
            if (engine->phys.use_shell_bank) {
                // New multi-mode resonator bank
                float shell_out = engine->shell_bank.process((float)sample, (float)dt);
                sample = (1.0 - engine->phys.shell_mix) * sample + engine->phys.shell_mix * shell_out;
            } else {
                // Old Biquad Filter
                float filtered_sample = engine->cur_filter_state.process((float)sample, cur_filter_c);
                
                if (engine->fading_filter) {
                    float fade_filtered = engine->fade_filter_state.process((float)sample, fade_filter_c);
                    // Clamp alpha between 0 and 1 during rapid updates
                    float alpha = std::min(engine->fade_filter_alpha, 1.0f);
                    filtered_sample = (1.0f - alpha) * filtered_sample + alpha * fade_filtered;
                    engine->fade_filter_alpha += alpha_step;
                }
                sample = filtered_sample;
            }

            // Apply user master volume and soft clip
            if (engine->diagnosis_mode) {
                // In diagnosis mode, output pure unaltered samples
                // No master volume, no soft-clipping
            } else {
                sample *= engine->master_volume;
                sample = std::tanh(sample);
            }

            buffer[i] = (float)sample;
            engine->last_strike_audio.push_back((float)sample);
        }

        // --- Commit fade at block boundary ---
        // Fading logic for tension coefficients (cur_idx, fade_idx, update_ready) has been removed
        // and replaced by interpolated double-buffering via active_tension_grid.

        // --- Commit filter fade at block boundary ---
        if (engine->fading_filter && engine->fade_filter_alpha >= 1.0f) {
            engine->cur_filter_idx.store(filter_fi, std::memory_order_release);
            engine->fade_filter_idx.store(filter_ci, std::memory_order_relaxed);
            engine->fading_filter = false;
            engine->fade_filter_alpha = 0.0f;
            engine->cur_filter_state = engine->fade_filter_state; // Keep state continuous
        }
    }
};
