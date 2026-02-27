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
};

// ─── Modal State (phasor Z, activity flags) ─────────────────────────────────
struct ModalState {
    float Z_re[MAX_MODES] = {};
    float Z_im[MAX_MODES] = {};
    float strike_scale[MAX_MODES] = {};  // Per-mode: gamma * overall_force (set per strike)
    uint8_t active[MAX_MODES] = {};
    int count = 0;
};

class AudioEngine {
public:
    std::mutex mutex;               // Only for strike/WAV export (not hot path coefficients)
    std::vector<float> last_strike_audio;

    // ─── Physical Constants ─────────────────────────────────────────────────
    double tension   = 200.0;       // T (N/m)
    double rho_s     = 0.26;        // Surface density (kg/m²)
    double alpha0    = 5.0;         // Frequency-independent damping (1/s)
    double alpha1    = 0.01;        // Frequency-proportional damping (s)
    double strike_v0 = 1.0;         // Strike velocity (m/s)
    double strike_width_delta = 0.05; // Spatial mallet strike width (m)

    // ─── Far-Field Radiation Parameters ─────────────────────────────────────
    double c_air = 343.0;           // Speed of sound in air (m/s)
    double rho_air = 1.225;         // Air density (kg/m³)
    double listener_distance = 1.0; // Far-field distance (m) — affects amplitude only
    double listener_elevation = 90.0; // Degrees (90° = directly overhead)
    double listener_azimuth   = 0.0;  // Degrees
    double master_volume = 10.0;    // User-adjustable master volume
    double peak_envelope = 0.0;     // Auto-gain peak tracker (callback-only)

    // ─── Precomputed Geometry ───────────────────────────────────────────────
    Eigen::MatrixXd MU;             // M·U (J × N_modes), precomputed at mesh rebuild
    std::vector<double> xi;         // Phase-centered ξ_j = n_∥·s_j - mean(ξ)
    int n_modes_cached = 0;

    // ─── Three-Buffer Coefficients (lock-free, fixed staging) ────────────────
    // coeff[0], coeff[1] = live buffers rotated between cur/fade by callback
    // coeff[2] = ALWAYS staging (UI writes here, callback never reads directly)
    CoeffSet coeff[3];
    std::atomic<int> cur_idx{0};        // Callback reads E/G/s²/Phi from this
    std::atomic<int> fade_idx{1};       // Callback crossfades Phi toward this
    // No staging_idx: staging is always coeff[2]
    std::atomic<bool> update_ready{false};  // Mailbox flag: UI sets, callback clears

    // ─── Crossfade State (only touched by callback) ─────────────────────────
    bool fading = false;
    float fade_alpha = 0.0f;
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

            // Damping (clamped to underdamped regime)
            double d_n = alpha0 + alpha1 * freq_hz;
            d_n = std::min(d_n, 0.95 * omega_n);

            // s_n = -d_n + i·ω_n
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

    // ─── Strike (called from UI thread) ─────────────────────────────────────
    // Uses MU for modal projection. No pickup_node_idx.
    void trigger_strike(const Mesh& mesh, const ModeData& modes, double strike_x, double strike_y, double force = 1.0) {
        std::lock_guard<std::mutex> lock(mutex);

        int n_modes = modes.eigenvalues.size();
        if (n_modes > MAX_MODES) n_modes = MAX_MODES;
        modes_state.count = n_modes;

        // Clear the buffer used for WAV export
        last_strike_audio.clear();

        double c = wave_speed();
        const double dt = 1.0 / 44100.0;

        // --- Setup Finite Strike Time Window ---
        strike_duration_delta = 0.002 / std::max(strike_v0, 0.01);
        current_strike_time = -strike_duration_delta;

        double temporal_bump_integral = 0.4439938 * strike_duration_delta;
        double overall_force = force / temporal_bump_integral;

        // --- Pre-compute the spatial distribution weights (bump function) ---
        int J = mesh.vertices.size();
        std::vector<double> F(J, 0.0);
        std::vector<int> active_nodes;
        double F_sum = 0.0;

        for (int j = 0; j < J; ++j) {
            double dx = mesh.vertices[j].x - strike_x;
            double dy = mesh.vertices[j].y - strike_y;
            double r = std::sqrt(dx*dx + dy*dy);

            if (r < strike_width_delta) {
                double xi_r = r / strike_width_delta;
                F[j] = std::exp(-1.0 / (1.0 - xi_r * xi_r));
                F_sum += F[j];
                active_nodes.push_back(j);
            }
        }

        // Fallback to closest point if delta is so small it misses all nodes
        if (F_sum == 0.0 && J > 0) {
            double min_r = 1e9;
            int closest = 0;
            for (int j = 0; j < J; ++j) {
                double dx = mesh.vertices[j].x - strike_x;
                double dy = mesh.vertices[j].y - strike_y;
                double r = dx*dx + dy*dy;
                if (r < min_r) { min_r = r; closest = j; }
            }
            F[closest] = 1.0;
            F_sum = 1.0;
            active_nodes.push_back(closest);
        }

        // Normalize
        for (int j : active_nodes) F[j] /= F_sum;

        // Read the current coefficient set for E/G values
        int ci = cur_idx.load(std::memory_order_acquire);
        const CoeffSet& cur = coeff[ci];

        for (int i = 0; i < n_modes; ++i) {
            // --- Compute modal excitation using MU (mass-consistent projection) ---
            double gamma = 0.0;
            for (int j : active_nodes) {
                gamma += MU(j, i) * F[j];
            }

            // Skip modes not coupled to the strike
            if (std::abs(gamma) < 1e-4) continue;

            modes_state.Z_re[i] = 0.0f;
            modes_state.Z_im[i] = 0.0f;
            modes_state.strike_scale[i] = (float)(gamma * overall_force);
            modes_state.active[i] = 1;
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
            engine->update_ready.store(false, std::memory_order_release);
            engine->fading = true;
            engine->fade_alpha = 0.0f;
        }

        const CoeffSet& cur_coeff = engine->coeff[ci];
        const CoeffSet& fade_coeff = engine->coeff[fi];

        for (int i = 0; i < samples; ++i) {
            double sample = 0.0;

            // --- Evaluate current temporal bump force f(t) ---
            double f_t = 0.0;
            if (engine->current_strike_time < engine->strike_duration_delta) {
                double xi_t = engine->current_strike_time / engine->strike_duration_delta;
                if (xi_t > -1.0 && xi_t < 1.0) {
                    f_t = std::exp(-1.0 / (1.0 - xi_t * xi_t));
                }
                engine->current_strike_time += dt;
            }

            float f_t_f = (float)f_t;
            bool can_deactivate = engine->current_strike_time >= engine->strike_duration_delta;

            for (int m = 0; m < engine->modes_state.count; ++m) {
                if (!engine->modes_state.active[m]) continue;

                float zr = engine->modes_state.Z_re[m];
                float zi = engine->modes_state.Z_im[m];

                // Evolution: ALWAYS uses current E/G (Policy P1: self-consistent dynamics)
                // G is base coefficient; strike_scale holds the per-mode gamma*force
                float ss = engine->modes_state.strike_scale[m];
                float next_zr = (zr * cur_coeff.E_re[m] - zi * cur_coeff.E_im[m]) + cur_coeff.G_re[m] * ss * f_t_f;
                float next_zi = (zr * cur_coeff.E_im[m] + zi * cur_coeff.E_re[m]) + cur_coeff.G_im[m] * ss * f_t_f;

                engine->modes_state.Z_re[m] = next_zr;
                engine->modes_state.Z_im[m] = next_zi;

                // Acceleration: a = s²·Z (using current s², consistent with current E)
                float ar = cur_coeff.s2_re[m] * next_zr - cur_coeff.s2_im[m] * next_zi;
                float ai = cur_coeff.s2_re[m] * next_zi + cur_coeff.s2_im[m] * next_zr;

                // Forcing correction during strike window
                if (f_t_f > 0.0f) {
                    float form_f = ss * f_t_f;
                    ar += cur_coeff.sB_re[m] * form_f;
                    ai += cur_coeff.sB_im[m] * form_f;
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

                // p += 2·Re(conj(Φ)·a) = 2·(Φ_re·a_re + Φ_im·a_im)
                // Include physical far-field scaling: ρ₀/(2πr)
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
    }
};
