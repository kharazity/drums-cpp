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
static constexpr int MAX_MODES = 256;

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
    
    // ─── Shell Resonance Filter Parameters ──────────────────────────────────
    double shell_freq = 200.0;      // Resonance Frequency (Hz) (Old Biquad)
    double shell_q = 5.0;           // Q Factor
    double shell_gain_db = 0.0;     // Gain (dB)
    ShellResonatorBank shell_bank;  // New Multi-mode Resonator Bank

    double peak_envelope = 0.0;     // Auto-gain peak tracker (callback-only)

    // ─── Precomputed Geometry ───────────────────────────────────────────────
    Eigen::MatrixXd MU;             // M·U (J × N_modes), precomputed at mesh rebuild
    const Eigen::MatrixXd* U_ptr = nullptr; // Pointer to eigenvectors (NOT owned)
    std::vector<double> xi;         // Phase-centered ξ_j = n_∥·s_j - mean(ξ)
    int n_modes_cached = 0;

    // ─── Mode-Dependent Damping Data (Milestone 2) ──────────────────────────
    std::vector<int> boundary_nodes;
    float modal_edge_participation[MAX_MODES] = {};
    float modal_air_participation[MAX_MODES] = {};

    // ─── Three-Buffer Coefficients (lock-free, fixed staging) ────────────────
    // coeff[0], coeff[1] = live buffers rotated between cur/fade by callback
    // coeff[2] = ALWAYS staging (UI writes here, callback never reads directly)
    CoeffSet coeff[3];
    std::atomic<int> cur_idx{0};        // Callback reads E/G/s²/Phi from this
    std::atomic<int> fade_idx{1};       // Callback crossfades Phi toward this
    // No staging_idx: staging is always coeff[2]
    std::atomic<bool> update_ready{false};  // Mailbox flag: UI sets, callback clears

    BiquadCoeffs filter_coeff[3];
    std::atomic<int> cur_filter_idx{0};
    std::atomic<int> fade_filter_idx{1};
    std::atomic<bool> filter_update_ready{false};
    BiquadState cur_filter_state;
    BiquadState fade_filter_state;

    // ─── Crossfade State (only touched by callback) ─────────────────────────
    bool fading = false;
    float fade_alpha = 0.0f;
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

    // ─── Compute radiation weights (call on tension/direction change) ────────
    // Writes into staging buffer. Sets update_ready for callback to pick up.
    void compute_radiation_weights(const ModeData& modes) {
        // Mailbox discipline: skip if callback hasn't consumed the last update
        if (update_ready.load(std::memory_order_acquire)) {
            return;  // Mailbox full — try again next frame
        }

        int n_modes = modes.eigenvalues.size();
        if (n_modes > MAX_MODES) n_modes = MAX_MODES;

        double c = wave_speed();
        const double dt = 1.0 / 44100.0;
        const double nyquist_limit = 0.45 * 44100.0;  // Nyquist guard
        int J = MU.rows();

        CoeffSet& stg = coeff[2];  // Staging is ALWAYS coeff[2]

        for (int n = 0; n < n_modes; ++n) {
            double Lambda_n = modes.eigenvalues[n];
            double omega_n = c * std::sqrt(std::max(Lambda_n, 0.0));
            double freq_hz = omega_n / (2.0 * M_PI);

            // Nyquist guard: mute modes that would alias
            if (freq_hz > nyquist_limit) {
                stg.E_re[n] = 0.0f; stg.E_im[n] = 0.0f;
                stg.G_re[n] = 0.0f; stg.G_im[n] = 0.0f;
                stg.s2_re[n] = 0.0f; stg.s2_im[n] = 0.0f;
                stg.sB_re[n] = 0.0f; stg.sB_im[n] = 0.0f;
                stg.Phi_re[n] = 0.0f; stg.Phi_im[n] = 0.0f;
                continue;
            }

            // Compute radiative participation (I_vol)
            double I_vol = 0.0;
            for (int j = 0; j < J; ++j) {
                I_vol += MU(j, n);
            }
            modal_air_participation[n] = (float)I_vol;

            // Damping (clamped to underdamped regime)
            double d_n = build_mode_damping(n, omega_n, freq_hz);
            d_n = std::min(d_n, 0.95 * omega_n);

            // s_n = -d_n + i·ω_n
            stg.s_re[n] = (float)(-d_n);
            stg.s_im[n] = (float)(omega_n);

            // s_n² = (d_n² - ω_n²) + i·(-2·d_n·ω_n)
            stg.s2_re[n] = (float)(d_n * d_n - omega_n * omega_n);
            stg.s2_im[n] = (float)(-2.0 * d_n * omega_n);

            // E = exp(s·dt)
            double exp_decay = std::exp(-d_n * dt);
            stg.E_re[n] = (float)(exp_decay * std::cos(omega_n * dt));
            stg.E_im[n] = (float)(exp_decay * std::sin(omega_n * dt));

            // G = ((E-1)/s) · norm
            // norm = 1/(s - s*) = 1/(2·i·ω_n)
            std::complex<double> s_n(-d_n, omega_n);
            std::complex<double> E_n(stg.E_re[n], stg.E_im[n]);
            std::complex<double> norm = 1.0 / (s_n - std::conj(s_n));
            std::complex<double> G_base = 0.0;
            if (std::abs(s_n) > 1e-12) {
                G_base = ((E_n - 1.0) / s_n) * norm;
            } else {
                G_base = dt * norm;
            }
            stg.G_re[n] = (float)G_base.real();
            stg.G_im[n] = (float)G_base.imag();

            // Forcing correction s·B = s·norm = s / (s - s*)
            std::complex<double> sB = s_n * norm;
            stg.sB_re[n] = (float)sB.real();
            stg.sB_im[n] = (float)sB.imag();

            // Radiation weight: Φ_n = Σ_j MU(j,n) · exp(i·k_n·ξ_j)
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
            stg.Phi_re[n] = (float)phi_re;
            stg.Phi_im[n] = (float)phi_im;
        }

        // Publish: signal callback that new coefficients are available
        update_ready.store(true, std::memory_order_release);
    }

    // ─── Compute Biquad Filter Coefficients (called on UI param change) ─────
    void compute_filter_coeffs() {
        if (filter_update_ready.load(std::memory_order_acquire)) {
            return;
        }

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
            return d_base + d_rad;
        }
    }

    // ─── Centralized coefficient rebuild ────────────────────────────────────
    void rebuild_physical_coeffs(const Mesh& mesh, const ModeData& modes) {
        compute_mode_descriptors(mesh, modes);
        compute_xi(mesh);
        compute_radiation_weights(modes);
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
            strike_duration_delta = 0.002 / std::max(strike_v0, 0.01);
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

        std::lock_guard<std::mutex> lock(engine->mutex);

        const double dt = 1.0 / 44100.0;
        const float alpha_step = 1.0f / (FADE_DURATION_S * 44100.0f);

        // --- Snapshot indices once per block ---
        int ci = engine->cur_idx.load(std::memory_order_acquire);
        int fi = engine->fade_idx.load(std::memory_order_relaxed);

        // --- Check for new coefficient update (once per block) ---
        if (engine->update_ready.load(std::memory_order_acquire)) {
            // Copy staging (coeff[2]) into fade target buffer
            std::memcpy(&engine->coeff[fi], &engine->coeff[2], sizeof(CoeffSet));
            // Initialize shell bank with target default modes
            engine->shell_bank.count = 4;
            engine->shell_bank.modes[0] = {2.0f * (float)M_PI * 120.0f, 0.05f, 1.0f, 0.0f, 0.0f};
            engine->shell_bank.modes[1] = {2.0f * (float)M_PI * 210.0f, 0.04f, 0.8f, 0.0f, 0.0f};
            engine->shell_bank.modes[2] = {2.0f * (float)M_PI * 340.0f, 0.03f, 0.6f, 0.0f, 0.0f};
            engine->shell_bank.modes[3] = {2.0f * (float)M_PI * 520.0f, 0.02f, 0.4f, 0.0f, 0.0f};
            engine->update_ready.store(false, std::memory_order_release);
            engine->fading = true;
            engine->fade_alpha = 0.0f;
        }

        const CoeffSet& cur_coeff = engine->coeff[ci];
        const CoeffSet& fade_coeff = engine->coeff[fi];

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
                        float sr = cur_coeff.s_re[m];
                        float si = cur_coeff.s_im[m];
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

                // Evolution: Z' = E·Z + G·forcing
                float next_zr = (zr * cur_coeff.E_re[m] - zi * cur_coeff.E_im[m]) + cur_coeff.G_re[m] * forcing;
                float next_zi = (zr * cur_coeff.E_im[m] + zi * cur_coeff.E_re[m]) + cur_coeff.G_im[m] * forcing;

                engine->modes_state.Z_re[m] = next_zr;
                engine->modes_state.Z_im[m] = next_zi;

                // Acceleration: a = s²·Z
                float ar = cur_coeff.s2_re[m] * next_zr - cur_coeff.s2_im[m] * next_zi;
                float ai = cur_coeff.s2_re[m] * next_zi + cur_coeff.s2_im[m] * next_zr;

                // Forcing correction during active forcing window
                if (forcing != 0.0f) {
                    ar += cur_coeff.sB_re[m] * forcing;
                    ai += cur_coeff.sB_im[m] * forcing;
                }

                // Output: crossfade Phi ONLY (radiation mapping)
                float pr, pi;
                if (engine->fading) {
                    float alpha = engine->fade_alpha;
                    pr = (1.0f - alpha) * cur_coeff.Phi_re[m] + alpha * fade_coeff.Phi_re[m];
                    pi = (1.0f - alpha) * cur_coeff.Phi_im[m] + alpha * fade_coeff.Phi_im[m];
                } else {
                    pr = cur_coeff.Phi_re[m];
                    pi = cur_coeff.Phi_im[m];
                }

                // p += 2·Re(conj(Φ)·a)
                sample += 2.0 * (pr * ar + pi * ai);

                // Deactivate dead modes
                if (can_deactivate && (next_zr * next_zr + next_zi * next_zi) < 1e-16f) {
                    engine->modes_state.active[m] = 0;
                    engine->modes_state.Z_re[m] = 0.0f;
                    engine->modes_state.Z_im[m] = 0.0f;
                }
            }

            // Advance crossfade alpha
            if (engine->fading) {
                engine->fade_alpha += alpha_step;
            }

            // Physical scaling: ρ₀/(2πr) where ρ₀=1.225 kg/m³, r=1m
            sample *= engine->rho_air / (2.0 * M_PI * engine->listener_distance);

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
            sample *= engine->master_volume;
            sample = std::tanh(sample);

            buffer[i] = (float)sample;
            engine->last_strike_audio.push_back((float)sample);
        }

        // --- Commit fade at block boundary ---
        if (engine->fading && engine->fade_alpha >= 1.0f) {
            // Fade complete: swap cur and fade (both always in {0,1})
            engine->cur_idx.store(fi, std::memory_order_release);
            engine->fade_idx.store(ci, std::memory_order_relaxed);
            // Now: cur=fi, fade=ci (old cur becomes new fade target for next update)
            engine->fading = false;
            engine->fade_alpha = 0.0f;
        }

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
