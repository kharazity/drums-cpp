
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

#include <SDL.h>
#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_sdlrenderer2.h"
#include "implot.h"

#include "geometry.h"
#include "fem.h"
#include "solver.h"
#include "audio.h"

// Basic Audio Config
const int SAMPLE_RATE = 44100;
const int BUFFER_SIZE = 512;

// Drawing Mode
enum AppMode { PLAY, DRAWING, PLACING_HOLE };

// Main Loop
int main(int argc, char* argv[]) {
    // ── Simulation State ────────────────────────────────────────────────────
    std::vector<Mesh> meshes;
    std::vector<FEM> fems;
    std::vector<ModeData> all_modes;
    AudioEngine audio;
    
    // Initial Setup
    meshes.push_back(Mesh());
    meshes[0].generate_regular_polygon(4, 1.0, 20);
    fems.push_back(FEM());
    fems[0].assemble(meshes[0]);
    all_modes.push_back(Solver::solve(fems[0], 60));
    audio.precompute_MU(fems[0], all_modes[0]);
    audio.compute_xi(meshes[0]);
    audio.compute_radiation_weights(all_modes[0]);
    
    // Visualization State
    float zoom = 300.0f;
    int n_modes = 60;
    int mesh_density = 20;
    
    // Drawing State
    enum ShapeType { REGULAR_POLYGON, ELLIPSE, ISOSPECTRAL, CUSTOM, ANNULUS };
    ShapeType current_shape = REGULAR_POLYGON;
    AppMode app_mode = PLAY;
    std::vector<Vertex> draw_points; // Polygon being drawn (world coords)
    std::vector<Vertex> active_polygon; // Saved custom polygon
    
    // Shape Parameters
    float ellipse_a = 1.0f;
    float ellipse_b = 0.8f;
    int polygon_n = 4;
    float polygon_L = 1.0f;
    float annulus_outer_r = 1.0f;
    float annulus_inner_r = 0.4f;
    
    // Holes system
    struct HoleDef {
        double cx, cy;   // center
        double a, b;     // semi-major, semi-minor axes
    };
    std::vector<HoleDef> shape_holes;
    float new_hole_a = 0.15f;  // default hole size
    float new_hole_b = 0.15f;
    
    // Shape Control Points (for interactive editing in play mode)
    std::vector<Vertex> shape_control_points;
    int dragged_cp_idx = -1;
    bool cp_dirty = false;
    
    // Initialize control points for default shape (4-sided regular polygon, L=1.0)
    {
        double circumradius = polygon_L / (2.0 * std::sin(M_PI / polygon_n));
        double offset = M_PI / 2.0 - M_PI / polygon_n;
        for (int i = 0; i < polygon_n; ++i) {
            double theta = offset + 2.0 * M_PI * i / polygon_n;
            shape_control_points.push_back({circumradius * std::cos(theta), circumradius * std::sin(theta)});
        }
    }
    
    // Spectrum Data
    std::vector<float> freqs;
    std::vector<float> amps;
    int last_struck_mesh = 0;
    int dragged_mesh_idx = -1;
    int dragged_draw_point_idx = -1;
    char export_wav_path[256] = "output/strike.wav";
    
    // ── Helper: compute outer boundary polygon for current shape ────────
    auto compute_outer_boundary = [&]() -> std::vector<Vertex> {
        std::vector<Vertex> boundary;
        if (current_shape == ELLIPSE) {
            double a = ellipse_a, b = ellipse_b;
            double h = std::pow(a - b, 2) / std::pow(a + b, 2);
            double perimeter = M_PI * (a + b) * (1.0 + (3.0 * h) / (10.0 + std::sqrt(4.0 - 3.0 * h)));
            double scale = 2.0 * std::max(a, b);
            int n = std::max(16, (int)(perimeter / (scale / mesh_density)));
            for (int i = 0; i < n; ++i) {
                double theta = 2.0 * M_PI * i / n;
                boundary.push_back({a * std::cos(theta), b * std::sin(theta)});
            }
        } else if (current_shape == REGULAR_POLYGON) {
            double circumradius = polygon_L / (2.0 * std::sin(M_PI / polygon_n));
            double offset = M_PI / 2.0 - M_PI / polygon_n;
            for (int i = 0; i < polygon_n; ++i) {
                double theta = offset + 2.0 * M_PI * i / polygon_n;
                boundary.push_back({circumradius * std::cos(theta), circumradius * std::sin(theta)});
            }
        } else if (current_shape == CUSTOM && !active_polygon.empty()) {
            boundary = active_polygon;
        }
        return boundary;
    };
    
    // ── Helper: convert HoleDefs to polygon vectors ─────────────────────
    auto holes_to_polygons = [&]() -> std::vector<std::vector<Vertex>> {
        std::vector<std::vector<Vertex>> polys;
        for (const auto& h : shape_holes) {
            std::vector<Vertex> poly;
            int n = std::max(16, (int)(mesh_density * 0.5));
            for (int i = 0; i < n; ++i) {
                double theta = 2.0 * M_PI * i / n;
                poly.push_back({h.cx + h.a * std::cos(theta), h.cy + h.b * std::sin(theta)});
            }
            polys.push_back(poly);
        }
        return polys;
    };
    
    // ────────────────────────────────────────────────────────────────────────

    // 1. Init Video & Timer (Essential)
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0) {
        std::cerr << "Error: " << SDL_GetError() << std::endl;
        return -1;
    }

    // 2. Init Audio (Optional / Fallback)
    bool audio_enabled = true;
    if (SDL_InitSubSystem(SDL_INIT_AUDIO) != 0) {
        std::cerr << "Warning: Failed to init audio: " << SDL_GetError() << " (Running in silent mode)" << std::endl;
        audio_enabled = false;
    }

    // Audio Setup
    SDL_AudioDeviceID dev = 0;
    if (audio_enabled) {
        SDL_AudioSpec want, have;
        SDL_zero(want);
        want.freq = SAMPLE_RATE;
        want.format = AUDIO_F32;
        want.channels = 1;
        want.samples = BUFFER_SIZE;
        want.callback = AudioEngine::AudioCallback;
        want.userdata = &audio;
        
        dev = SDL_OpenAudioDevice(NULL, 0, &want, &have, 0);
        if (dev == 0) {
            std::cerr << "Failed to open audio device: " << SDL_GetError() << std::endl;
            audio_enabled = false;
        } else {
            SDL_PauseAudioDevice(dev, 0); // Start playing
        }
    }

    // Setup window
    SDL_WindowFlags window_flags = (SDL_WindowFlags)(SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI);
    SDL_Window* window = SDL_CreateWindow("Drums++", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, window_flags);
    if (!window) {
         std::cerr << "Warning: Failed to create window. Proceeding." << std::endl;
    }
    
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_PRESENTVSYNC | SDL_RENDERER_ACCELERATED);
    if (!renderer) {
         std::cerr << "Warning: Failed to create renderer. Proceeding." << std::endl;
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui::StyleColorsDark();

    // Setup Platform/Renderer backends
    if (window && renderer) {
        ImGui_ImplSDL2_InitForSDLRenderer(window, renderer);
        ImGui_ImplSDLRenderer2_Init(renderer);
    }

    // Initialize shell resonance filter
    audio.compute_filter_coeffs();

    
    struct BeaterPreset {
        const char* name;
        double mass, K_lo, K_hi, R_lo, R_hi, exponent, width;
    };
    static const BeaterPreset BEATER_PRESETS[] = {
        {"Stick",       0.015, 5e5, 2e6, 5.0, 20.0, 2.0, 0.01},
        {"Hard Mallet", 0.04,  2e5, 1e6, 10.0, 40.0, 1.5, 0.03},
        {"Soft Mallet", 0.06,  5e4, 3e5, 20.0, 80.0, 1.5, 0.08},
        {"Finger",      0.02,  1e4, 8e4, 30.0, 100.0, 1.2, 0.15},
    };
    int current_beater_idx = 0;
    float current_beater_hardness = 0.5f;

    struct MicPreset { 
        const char* name; 
        double elev, azim; 
    };
    static const MicPreset MIC_PRESETS[] = {
        {"Player",   60.0, 0.0},
        {"Front",    30.0, 0.0},
        {"Overhead", 90.0, 0.0},
        {"Room",     20.0, 45.0},
        {"Edge",     10.0, 0.0},
    };
    int current_mic_idx = 0;

    float current_damping_macro = 0.5f;
    float current_hit_strength = 5.0f;

    bool done = false;
    while (!done) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (window && renderer) ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT)
                done = true;
            if (window && event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE && event.window.windowID == SDL_GetWindowID(window))
                done = true;
        }

        if (window && renderer) {
            // Start the Dear ImGui frame
            ImGui_ImplSDLRenderer2_NewFrame();
            ImGui_ImplSDL2_NewFrame();
            ImGui::NewFrame();

            // ═══════════════════════════════════════════════════════════════
            // 1. Control Panel
            // ═══════════════════════════════════════════════════════════════
            {
                ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
                ImGui::SetNextWindowSize(ImVec2(300, 450), ImGuiCond_FirstUseEver);
                ImGui::Begin("Controls");
                ImGui::Text("FPS: %.1f", io.Framerate);
                ImGui::Separator();

                enum UITab { TAB_PLAY, TAB_DESIGN, TAB_ADVANCED };
                static UITab current_tab = TAB_PLAY;
                
                if (ImGui::BeginTabBar("MainTabs")) {
                    if (ImGui::BeginTabItem("Play")) { current_tab = TAB_PLAY; ImGui::EndTabItem(); }
                    if (ImGui::BeginTabItem("Design")) { current_tab = TAB_DESIGN; ImGui::EndTabItem(); }
                    if (ImGui::BeginTabItem("Advanced")) { current_tab = TAB_ADVANCED; ImGui::EndTabItem(); }
                    ImGui::EndTabBar();
                }
                ImGui::Separator();

                // ── Shape Selection (Shared across Play & Design) ────────
                if (current_tab == TAB_PLAY || current_tab == TAB_DESIGN) {
                    const char* shape_names[] = { "Regular Polygon", "Ellipse", "Isospectral Drums", "Custom (Draw)", "Annulus" };
                    int current_shape_idx = (int)current_shape;
                    
                    if (ImGui::Combo("Shape Type", &current_shape_idx, shape_names, IM_ARRAYSIZE(shape_names))) {
                        current_shape = (ShapeType)current_shape_idx;
                        shape_holes.clear();
                        if (current_shape == CUSTOM) {
                            app_mode = DRAWING;
                            draw_points.clear();
                        } else {
                            app_mode = PLAY;
                        }
                    }
                    
                    if (current_shape == ELLIPSE) {
                        float min_axis = 3.0f / (float)mesh_density;
                        ImGui::InputFloat("Semi-major (a)", &ellipse_a, 0.01f, 0.1f, "%.3f");
                        if (ellipse_a < min_axis) ellipse_a = min_axis;
                        ImGui::InputFloat("Semi-minor (b)", &ellipse_b, 0.01f, 0.1f, "%.3f");
                        if (ellipse_b < min_axis) ellipse_b = min_axis;
                        ImGui::TextDisabled("(min: %.4f based on density)", min_axis);
                    } else if (current_shape == REGULAR_POLYGON) {
                        ImGui::SliderInt("Sides (n)", &polygon_n, 3, 20);
                        ImGui::SliderFloat("Side Length (L)", &polygon_L, 0.1f, 3.0f);
                    } else if (current_shape == ANNULUS) {
                        float min_gap = 3.0f / (float)mesh_density;
                        ImGui::InputFloat("Outer Radius", &annulus_outer_r, 0.01f, 0.1f, "%.3f");
                        if (annulus_outer_r < 2.0f * min_gap) annulus_outer_r = 2.0f * min_gap;
                        ImGui::InputFloat("Inner Radius", &annulus_inner_r, 0.01f, 0.1f, "%.3f");
                        if (annulus_inner_r < min_gap) annulus_inner_r = min_gap;
                        if (annulus_inner_r >= annulus_outer_r - min_gap) annulus_inner_r = annulus_outer_r - min_gap;
                        ImGui::TextDisabled("(gap min: %.4f)", min_gap);
                    }

                    ImGui::Spacing();
                    if (ImGui::Button("Rebuild Geometry", ImVec2(-1, 0))) {
                        if (current_shape == ISOSPECTRAL) {
                            meshes = Mesh::generate_isospectral(mesh_density);
                            shape_control_points.clear();
                        } else {
                            meshes.assign(1, Mesh());
                            auto hole_polys = holes_to_polygons();
                            
                            if (current_shape == ELLIPSE) {
                                if (hole_polys.empty()) {
                                    meshes[0].generate_ellipse(ellipse_a, ellipse_b, mesh_density);
                                } else {
                                    auto outer = compute_outer_boundary();
                                    meshes[0].generate_from_polygon(outer, hole_polys, mesh_density);
                                }
                                shape_control_points = {{(double)ellipse_a, 0.0}, {0.0, (double)ellipse_b}};
                            } else if (current_shape == REGULAR_POLYGON) {
                                if (hole_polys.empty()) {
                                    meshes[0].generate_regular_polygon(polygon_n, polygon_L, mesh_density);
                                } else {
                                    auto outer = compute_outer_boundary();
                                    meshes[0].generate_from_polygon(outer, hole_polys, mesh_density);
                                }
                                shape_control_points.clear();
                                double circumradius = polygon_L / (2.0 * std::sin(M_PI / polygon_n));
                                double offset = M_PI / 2.0 - M_PI / polygon_n;
                                for (int i = 0; i < polygon_n; ++i) {
                                    double theta = offset + 2.0 * M_PI * i / polygon_n;
                                    shape_control_points.push_back({circumradius * std::cos(theta), circumradius * std::sin(theta)});
                                }
                            } else if (current_shape == CUSTOM && !active_polygon.empty()) {
                                meshes[0].generate_from_polygon(active_polygon, hole_polys, mesh_density);
                                shape_control_points = active_polygon;
                            } else if (current_shape == ANNULUS) {
                                meshes[0].generate_annulus(annulus_outer_r, annulus_inner_r, mesh_density);
                                shape_control_points = {{(double)annulus_outer_r, 0.0}, {(double)annulus_inner_r, 0.0}};
                            }
                        }
                        
                        // Rebuild FEM and Physics for all generated meshes
                        fems.clear();
                        all_modes.clear();
                        for (auto& m : meshes) {
                            if (m.elements.size() > 0) {
                                FEM f;
                                f.assemble(m);
                                fems.push_back(f);
                                all_modes.push_back(Solver::solve(f, n_modes));
                            }
                        }
                        
                        // Precompute radiation weights for first mesh
                        if (!fems.empty() && !all_modes.empty()) {
                            audio.precompute_MU(fems[0], all_modes[0]);
                            audio.compute_xi(meshes[0]);
                            audio.compute_radiation_weights(all_modes[0]);
                        }
                        
                        freqs.clear();
                        {
                            std::lock_guard<std::mutex> lock(audio.mutex);
                            audio.modes_state.count = 0;
                            memset(audio.modes_state.active, 0, sizeof(audio.modes_state.active));
                        }
                    }
                    

                }

                if (current_tab == TAB_PLAY) {
                    // ── Play Tab ─────────────────────────────────────────────
                    float tension_f = (float)audio.tension;
                    if (ImGui::InputFloat("Tension", &tension_f, 1.0f, 10.0f, "%.1f")) {
                        if (tension_f <= 0.0f) {
                            tension_f = 0.1f;
                        }
                        audio.tension = (double)tension_f;
                        // Debounced radiation weight recompute on tension change
                        if (!all_modes.empty()) {
                            audio.compute_radiation_weights(all_modes[0]);
                        }
                    }

                    ImGui::Spacing();
                    if (ImGui::Button("Strike Center", ImVec2(-1, 0))) {
                        if (!meshes.empty() && !meshes[0].vertices.empty() && !all_modes.empty()) {
                            double cx = 0, cy = 0;
                            for (const auto& v : meshes[0].vertices) { cx += v.x; cy += v.y; }
                            cx /= meshes[0].vertices.size();
                            cy /= meshes[0].vertices.size();
                            audio.trigger_strike(meshes[0], all_modes[0], cx, cy, 1.0);
                        }
                    }

                    ImGui::Separator();
                    ImGui::Text("Performance");
                    if (ImGui::SliderFloat("Hit Strength", &current_hit_strength, 0.1f, 10.0f, "%.1f", ImGuiSliderFlags_Logarithmic)) {
                        audio.strike_v0 = (double)current_hit_strength;
                        audio.phys.striker_initial_velocity = (double)current_hit_strength;
                    }
                    
                    const char* beater_names[] = { "Stick", "Hard Mallet", "Soft Mallet", "Finger" };
                    bool beater_changed = ImGui::Combo("Beater Type", &current_beater_idx, beater_names, IM_ARRAYSIZE(beater_names));
                    bool hardness_changed = ImGui::SliderFloat("Beater Hardness", &current_beater_hardness, 0.0f, 1.0f);
                    
                    if (beater_changed || hardness_changed) {
                        const auto& b = BEATER_PRESETS[current_beater_idx];
                        audio.phys.striker_mass = b.mass;
                        audio.phys.striker_stiffness = b.K_lo + current_beater_hardness * (b.K_hi - b.K_lo);
                        audio.phys.striker_damping = b.R_lo + current_beater_hardness * (b.R_hi - b.R_lo);
                        audio.phys.striker_exponent = b.exponent;
                        audio.strike_width_delta = b.width;
                    }
                    
                    bool damping_changed = ImGui::SliderFloat("Damping / Muffling", &current_damping_macro, 0.0f, 1.0f);
                    if (damping_changed) {
                        audio.alpha0 = 0.5 + current_damping_macro * 19.5;
                        audio.phys.air_loss_weight = 0.5 + current_damping_macro * 2.5;
                        audio.phys.edge_loss_weight = 0.0 + current_damping_macro * 50.0;
                        if (!all_modes.empty()) audio.compute_radiation_weights(all_modes[0]);
                    }

                    float mix_f = (float)audio.phys.shell_mix;
                    if (ImGui::SliderFloat("Body", &mix_f, 0.0f, 1.0f, "%.2f")) {
                        audio.phys.shell_mix = (double)mix_f;
                    }

                    ImGui::Separator();
                    ImGui::Text("Environment");

                    const char* mic_names[] = { "Player", "Front", "Overhead", "Room", "Edge" };
                    if (ImGui::Combo("Mic Preset", &current_mic_idx, mic_names, IM_ARRAYSIZE(mic_names))) {
                        const auto& m = MIC_PRESETS[current_mic_idx];
                        audio.listener_elevation = m.elev;
                        audio.listener_azimuth = m.azim;
                        if (!meshes.empty()) audio.compute_xi(meshes[0]);
                        if (!all_modes.empty()) audio.compute_radiation_weights(all_modes[0]);
                    }

                    float vol_f = (float)audio.master_volume;
                    if (ImGui::SliderFloat("Master Volume", &vol_f, 0.01f, 100.0f, "%.2f", ImGuiSliderFlags_Logarithmic)) {
                        audio.master_volume = (double)vol_f;
                    }
                } else if (current_tab == TAB_DESIGN) {
                    // ── Design Tab ─────────────────────────────────────────────
                    if (ImGui::InputInt("Modes", &n_modes, 1, 10)) {
                        if (n_modes < 1) n_modes = 1;
                    }
                    if (ImGui::InputInt("Mesh Density", &mesh_density)) {
                        if (mesh_density < 5) mesh_density = 5;
                        if (mesh_density > 1000000) mesh_density = 1000000;
                    }

                    if (app_mode == PLAY || app_mode == PLACING_HOLE) {
                        // ── Holes UI ─────────────────────────────────────────
                        if (current_shape != ISOSPECTRAL && current_shape != ANNULUS) {
                            ImGui::Separator();
                            bool holes_open = ImGui::TreeNodeEx("Holes", ImGuiTreeNodeFlags_DefaultOpen,
                                                                "Holes  (%d placed)", (int)shape_holes.size());
                            if (holes_open) {
                                ImGui::Indent(4.0f);
                                
                                bool hole_size_changed = false;
                                float min_gap = std::max(0.01f, 3.0f / (float)mesh_density);
                                hole_size_changed |= ImGui::InputFloat("a (semi-major)", &new_hole_a, 0.01f, 0.05f, "%.3f");
                                if (new_hole_a < min_gap) new_hole_a = min_gap;
                                hole_size_changed |= ImGui::InputFloat("b (semi-minor)", &new_hole_b, 0.01f, 0.05f, "%.3f");
                                if (new_hole_b < min_gap) new_hole_b = min_gap;

                                // Live-update all existing holes when a/b change
                                if (hole_size_changed && !shape_holes.empty()) {
                                    for (auto& h : shape_holes) {
                                        h.a = new_hole_a;
                                        h.b = new_hole_b;
                                    }
                                    meshes.assign(1, Mesh());
                                    auto hole_polys = holes_to_polygons();
                                    auto outer = compute_outer_boundary();
                                    if (!outer.empty()) {
                                        meshes[0].generate_from_polygon(outer, hole_polys, mesh_density);
                                    }
                                    fems.clear();
                                    all_modes.clear();
                                    if (meshes[0].elements.size() > 0) {
                                        FEM f;
                                        f.assemble(meshes[0]);
                                        fems.push_back(f);
                                        all_modes.push_back(Solver::solve(f, n_modes));
                                        audio.precompute_MU(fems[0], all_modes[0]);
                                        audio.compute_xi(meshes[0]);
                                        audio.compute_radiation_weights(all_modes[0]);
                                    }
                                }

                                ImGui::Spacing();
                                if (app_mode == PLACING_HOLE) {
                                    ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.0f, 1.0f), "[+] Click canvas to place hole");
                                    if (ImGui::Button("Cancel")) app_mode = PLAY;
                                } else {
                                    if (ImGui::Button("Add Hole")) app_mode = PLACING_HOLE;
                                    if (!shape_holes.empty()) {
                                        ImGui::SameLine();
                                        if (ImGui::Button("Clear All")) shape_holes.clear();
                                    }
                                }
                                ImGui::Unindent(4.0f);
                                ImGui::TreePop();
                            }
                        }

                        ImGui::Spacing();
                        ImGui::Separator();

                        ImGui::Spacing();
                        if (ImGui::Button("Strike Center", ImVec2(-1, 0))) {
                            if (!meshes.empty() && !meshes[0].vertices.empty() && !all_modes.empty()) {
                                double cx = 0, cy = 0;
                                for (const auto& v : meshes[0].vertices) { cx += v.x; cy += v.y; }
                                cx /= meshes[0].vertices.size();
                                cy /= meshes[0].vertices.size();
                                audio.trigger_strike(meshes[0], all_modes[0], cx, cy, 1.0);
                            }
                        }
                    } else {
                        // Drawing mode controls
                        ImGui::TextColored(ImVec4(1, 0.8f, 0, 1), "DRAWING MODE");
                        ImGui::Text("Left-click to add points");
                        ImGui::Text("Points: %d", (int)draw_points.size());

                        if (draw_points.size() >= 3) {
                            if (ImGui::Button("Finish & Build")) {
                                // Close polygon and mesh it
                                active_polygon = draw_points;
                                meshes.assign(1, Mesh());
                                meshes[0].generate_from_polygon(active_polygon, {}, mesh_density);
                                
                                fems.clear();
                                all_modes.clear();
                                if (meshes[0].elements.size() > 0) {
                                    FEM f;
                                    f.assemble(meshes[0]);
                                    fems.push_back(f);
                                    all_modes.push_back(Solver::solve(f, n_modes));

                                    audio.precompute_MU(fems[0], all_modes[0]);
                                    audio.compute_xi(meshes[0]);
                                    audio.compute_radiation_weights(all_modes[0]);

                                    freqs.clear();
                                    {
                                        std::lock_guard<std::mutex> lock(audio.mutex);
                                        audio.modes_state.count = 0;
                                        memset(audio.modes_state.active, 0, sizeof(audio.modes_state.active));
                                    }
                                }
                                shape_control_points = active_polygon;  // Custom polygon control points
                                app_mode = PLAY;
                                draw_points.clear();
                            }
                            ImGui::SameLine();
                        }
                        if (ImGui::Button("Cancel")) {
                            app_mode = PLAY;
                            current_shape = REGULAR_POLYGON; // fallback out of custom mode
                            draw_points.clear();
                        }
                    }
                } else if (current_tab == TAB_ADVANCED) {
                    // ── Advanced Tab ─────────────────────────────────────────
                    ImGui::Text("Physical Subsystems");
                    ImGui::Checkbox("Use Contact Striker", &audio.phys.use_contact_model);
                    if (ImGui::Checkbox("Mode-Dependent Damping", &audio.phys.use_mode_dependent_damping)) {
                        if (!all_modes.empty()) audio.compute_radiation_weights(all_modes[0]);
                    }
                    ImGui::Checkbox("Use Multi-Mode Shell Bank", &audio.phys.use_shell_bank);

                    if (audio.phys.use_contact_model) {
                        ImGui::Separator();
                        ImGui::Text("Raw Striker");
                        float sm_f = (float)audio.phys.striker_mass;
                        if (ImGui::SliderFloat("Mass (kg)", &sm_f, 0.005f, 0.2f, "%.3f", ImGuiSliderFlags_Logarithmic)) audio.phys.striker_mass = (double)sm_f;
                        float sk_f = (float)audio.phys.striker_stiffness;
                        if (ImGui::SliderFloat("Stiffness (K)", &sk_f, 1e4f, 1e7f, "%.0f", ImGuiSliderFlags_Logarithmic)) audio.phys.striker_stiffness = (double)sk_f;
                        float sr_f = (float)audio.phys.striker_damping;
                        if (ImGui::SliderFloat("Contact Damping (R)", &sr_f, 0.1f, 100.0f, "%.1f", ImGuiSliderFlags_Logarithmic)) audio.phys.striker_damping = (double)sr_f;
                        float sp_f = (float)audio.phys.striker_exponent;
                        if (ImGui::SliderFloat("Exponent (p)", &sp_f, 1.0f, 3.0f, "%.2f")) audio.phys.striker_exponent = (double)sp_f;
                        float sw_f = (float)audio.strike_width_delta;
                        if (ImGui::SliderFloat("Mallet Width", &sw_f, 0.001f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic)) audio.strike_width_delta = (double)sw_f;
                    }

                    ImGui::Separator();
                    ImGui::Text("Raw Damping");
                    float alpha0_f = (float)audio.alpha0;
                    if (ImGui::SliderFloat("Base Damping (a0)", &alpha0_f, 0.1f, 50.0f, "%.1f")) { audio.alpha0 = (double)alpha0_f; if (!all_modes.empty()) audio.compute_radiation_weights(all_modes[0]); }
                    float alpha1_f = (float)audio.alpha1;
                    if (ImGui::SliderFloat("Base Freq Damp (a1)", &alpha1_f, 0.0f, 0.05f, "%.5f")) { audio.alpha1 = (double)alpha1_f; if (!all_modes.empty()) audio.compute_radiation_weights(all_modes[0]); }
                    float beta_f = (float)audio.beta;
                    if (ImGui::SliderFloat("Base Freq Power (beta)", &beta_f, 0.5f, 3.0f, "%.2f")) { audio.beta = (double)beta_f; if (!all_modes.empty()) audio.compute_radiation_weights(all_modes[0]); }
                    
                    if (audio.phys.use_mode_dependent_damping) {
                        float air_w_f = (float)audio.phys.air_loss_weight;
                        if (ImGui::SliderFloat("Air Loss Weight", &air_w_f, 0.0f, 5.0f, "%.2f")) { audio.phys.air_loss_weight = (double)air_w_f; if (!all_modes.empty()) audio.compute_radiation_weights(all_modes[0]); }
                        float edge_w_f = (float)audio.phys.edge_loss_weight;
                        if (ImGui::SliderFloat("Edge Loss Weight", &edge_w_f, 0.0f, 100.0f, "%.1f", ImGuiSliderFlags_Logarithmic)) { audio.phys.edge_loss_weight = (double)edge_w_f; if (!all_modes.empty()) audio.compute_radiation_weights(all_modes[0]); }
                    }

                    if (!audio.phys.use_shell_bank) {
                        ImGui::Separator();
                        ImGui::Text("Old Biquad EQ");
                        float shell_freq_f = (float)audio.shell_freq;
                        if (ImGui::SliderFloat("Frequency (Hz)", &shell_freq_f, 20.0f, 1000.0f, "%.1f")) { audio.shell_freq = (double)shell_freq_f; audio.compute_filter_coeffs(); }
                        float shell_q_f = (float)audio.shell_q;
                        if (ImGui::SliderFloat("Q Factor", &shell_q_f, 0.1f, 20.0f, "%.2f")) { audio.shell_q = (double)shell_q_f; audio.compute_filter_coeffs(); }
                        float shell_gain_f = (float)audio.shell_gain_db;
                        if (ImGui::SliderFloat("Gain (dB)", &shell_gain_f, -24.0f, 24.0f, "%.1f")) { audio.shell_gain_db = (double)shell_gain_f; audio.compute_filter_coeffs(); }
                    } else {
                        ImGui::Separator();
                        ImGui::Text("Multi-Mode Shell Bank Active");
                        float mix_f = (float)audio.phys.shell_mix;
                        if (ImGui::SliderFloat("Shell Mix", &mix_f, 0.0f, 1.0f, "%.2f")) {
                            audio.phys.shell_mix = (double)mix_f;
                        }
                    }

                    ImGui::Separator();
                    ImGui::Text("Listener");
                    float elevation_f = (float)audio.listener_elevation;
                    if (ImGui::SliderFloat("Elevation (deg)", &elevation_f, 0.0f, 90.0f, "%.1f")) {
                        audio.listener_elevation = (double)elevation_f;
                        if (!meshes.empty()) audio.compute_xi(meshes[0]);
                        if (!all_modes.empty()) audio.compute_radiation_weights(all_modes[0]);
                    }
                    float azimuth_f = (float)audio.listener_azimuth;
                    if (ImGui::SliderFloat("Azimuth (deg)", &azimuth_f, 0.0f, 360.0f, "%.1f")) {
                        audio.listener_azimuth = (double)azimuth_f;
                        if (!meshes.empty()) audio.compute_xi(meshes[0]);
                        if (!all_modes.empty()) audio.compute_radiation_weights(all_modes[0]);
                    }

                    ImGui::Separator();
                    ImGui::Text("Metrics");
                    ImGui::Text("Wave Speed (c) = %.1f m/s", audio.wave_speed());
                    int total_verts = 0;
                    for (const auto& m : meshes) total_verts += m.vertices.size();
                    ImGui::Text("Meshes: %lu  Total Vertices: %d", meshes.size(), total_verts);
                }

                // ── Shared Utilities (always visible at bottom) ──────────
                ImGui::Separator();

                ImGui::Text("Audio Export");
                ImGui::InputText("WAV Path", export_wav_path, IM_ARRAYSIZE(export_wav_path));
                if (ImGui::Button("Export to .wav")) {
                    std::lock_guard<std::mutex> lock(audio.mutex);
                    if (audio.last_strike_audio.empty()) {
                         std::cerr << "Warning: No audio to export. Strike the drum first." << std::endl;
                    } else if (write_wav_file(export_wav_path, audio.last_strike_audio, SAMPLE_RATE)) {
                         std::cout << "Successfully exported " << audio.last_strike_audio.size() << " samples to " << export_wav_path << std::endl;
                    } else {
                         std::cerr << "Error: Failed to write WAV file to " << export_wav_path << std::endl;
                    }
                }

                ImGui::End();
            }
            // ═══════════════════════════════════════════════════════════════
            // 2. Drum View (Mesh Rendering & Interaction)
            // ═══════════════════════════════════════════════════════════════
            {
                ImGui::SetNextWindowPos(ImVec2(320, 10), ImGuiCond_FirstUseEver);
                ImGui::SetNextWindowSize(ImVec2(940, 700), ImGuiCond_FirstUseEver);
                ImGui::Begin("Drum View");
                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                ImVec2 p0 = ImGui::GetCursorScreenPos();
                ImVec2 sz = ImGui::GetContentRegionAvail();
                
                static float pan_x = 0.0f;
                static float pan_y = 0.0f;
                
                if (ImGui::IsWindowHovered()) {
                    ImGuiIO& io = ImGui::GetIO();
                    // Panning (Middle or Right mouse button drag)
                    if (ImGui::IsMouseDragging(ImGuiMouseButton_Middle) || ImGui::IsMouseDragging(ImGuiMouseButton_Right)) {
                        pan_x += io.MouseDelta.x;
                        pan_y += io.MouseDelta.y;
                    }
                    
                    // Zooming (Scroll wheel)
                    if (io.MouseWheel != 0.0f) {
                        float old_zoom = zoom;
                        zoom *= (1.0f + io.MouseWheel * 0.15f);
                        if (zoom < 1.0f) zoom = 1.0f;
                        if (zoom > 100000.0f) zoom = 100000.0f;
                        
                        // Zoom-towards-cursor math
                        float cx = p0.x + sz.x * 0.5f;
                        float cy = p0.y + sz.y * 0.5f;
                        
                        float world_x = (io.MousePos.x - cx - pan_x) / old_zoom;
                        float world_y = (io.MousePos.y - cy - pan_y) / old_zoom;
                        
                        pan_x = io.MousePos.x - cx - world_x * zoom;
                        pan_y = io.MousePos.y - cy - world_y * zoom;
                    }
                }
                
                ImVec2 center = ImVec2(p0.x + sz.x*0.5f + pan_x, p0.y + sz.y*0.5f + pan_y);
                if (sz.x < 50 || sz.y < 50) {
                    center = ImVec2(1280/2 + pan_x, 720/2 + pan_y);
                }
                
                // Reset View button
                ImGui::SetCursorPos(ImVec2(ImGui::GetWindowWidth() - 90, 30));
                if (ImGui::Button("Reset View")) {
                    pan_x = 0.0f;
                    pan_y = 0.0f;
                    zoom = 120.0f;
                }

                // ── DRAWING MODE ─────────────────────────────────────────
                if (app_mode == DRAWING) {
                    // Use InvisibleButton to capture all mouse interaction
                    ImGui::InvisibleButton("draw_canvas", sz);
                    bool canvas_hovered = ImGui::IsItemHovered();
                    ImVec2 mouse_pos = ImGui::GetMousePos();

                    // Convert mouse to world coords
                    double wx = (mouse_pos.x - center.x) / zoom;
                    double wy = -(mouse_pos.y - center.y) / zoom;

                    // Hit-test radius in screen pixels
                    const float HIT_RADIUS = 10.0f;

                    // Find which vertex (if any) is under the mouse
                    int hovered_vertex = -1;
                    for (int i = 0; i < (int)draw_points.size(); ++i) {
                        ImVec2 sp = ImVec2(center.x + draw_points[i].x * zoom,
                                           center.y - draw_points[i].y * zoom);
                        float dx = mouse_pos.x - sp.x, dy = mouse_pos.y - sp.y;
                        if (dx*dx + dy*dy <= HIT_RADIUS*HIT_RADIUS) {
                            hovered_vertex = i;
                            break;
                        }
                    }

                    // Mouse button down: start drag or add point
                    if (canvas_hovered && ImGui::IsMouseClicked(0)) {
                        if (hovered_vertex != -1) {
                            dragged_draw_point_idx = hovered_vertex;  // Start dragging
                        } else {
                            draw_points.push_back({wx, wy});           // Add new vertex
                        }
                    }

                    // Continue dragging
                    if (dragged_draw_point_idx != -1 && ImGui::IsMouseDown(0)) {
                        draw_points[dragged_draw_point_idx].x = wx;
                        draw_points[dragged_draw_point_idx].y = wy;
                    }

                    // Release drag
                    if (ImGui::IsMouseReleased(0)) {
                        dragged_draw_point_idx = -1;
                    }

                    // Draw edges
                    for (int i = 0; i < (int)draw_points.size(); ++i) {
                        if (i > 0) {
                            ImVec2 a = ImVec2(center.x + draw_points[i-1].x * zoom,
                                              center.y - draw_points[i-1].y * zoom);
                            ImVec2 b = ImVec2(center.x + draw_points[i].x * zoom,
                                              center.y - draw_points[i].y * zoom);
                            draw_list->AddLine(a, b, IM_COL32(255, 200, 50, 200), 2.5f);
                        }
                    }

                    // Closing edge preview
                    if (draw_points.size() >= 3) {
                        ImVec2 first = ImVec2(center.x + draw_points[0].x * zoom,
                                              center.y - draw_points[0].y * zoom);
                        ImVec2 last  = ImVec2(center.x + draw_points.back().x * zoom,
                                              center.y - draw_points.back().y * zoom);
                        draw_list->AddLine(last, first, IM_COL32(255, 200, 50, 100), 1.5f);
                    }

                    // Draw vertices (red dots; highlight hover=white, drag=cyan)
                    for (int i = 0; i < (int)draw_points.size(); ++i) {
                        ImVec2 sp = ImVec2(center.x + draw_points[i].x * zoom,
                                           center.y - draw_points[i].y * zoom);
                        ImU32 col;
                        float r;
                        if (i == dragged_draw_point_idx) {
                            col = IM_COL32(0, 220, 255, 255);  // Cyan: dragging
                            r = 8.0f;
                        } else if (i == hovered_vertex) {
                            col = IM_COL32(255, 255, 255, 255); // White: hover
                            r = 7.0f;
                        } else {
                            col = IM_COL32(220, 60, 60, 255);   // Red: normal
                            r = 5.0f;
                        }
                        draw_list->AddCircleFilled(sp, r, col);
                        draw_list->AddCircle(sp, r + 1.5f, IM_COL32(0,0,0,120), 0, 1.5f);
                    }

                    // Cursor trail from last point to mouse (only when not about to drag)
                    if (!draw_points.empty() && hovered_vertex == -1 && dragged_draw_point_idx == -1) {
                        ImVec2 last_sp = ImVec2(center.x + draw_points.back().x * zoom,
                                                center.y - draw_points.back().y * zoom);
                        draw_list->AddLine(last_sp, mouse_pos, IM_COL32(255, 255, 255, 80), 1.0f);
                    }

                    // Help text
                    const char* hint = (hovered_vertex != -1)
                        ? "Drag vertex to reshape. Click empty space to add."
                        : (draw_points.size() >= 3)
                            ? "Click to add points. Drag red dots to adjust. 'Finish & Build' when done."
                            : "Click to add polygon vertices.";
                    draw_list->AddText(ImVec2(p0.x + 10, p0.y + 10), IM_COL32(255, 200, 50, 255), hint);
                }
                // ── PLAY MODE ────────────────────────────────────────────
                else {
                    // Add invisible button to capture clicks in the window area so the window doesn't drag
                    ImGui::InvisibleButton("play_canvas", sz);
                    bool play_hovered = ImGui::IsItemHovered();
                    
                    // Interaction: Control Point Drag + Strike + Mesh Drag
                    ImVec2 mouse_pos = ImGui::GetMousePos();
                    double wx = (mouse_pos.x - center.x) / zoom;
                    double wy = -(mouse_pos.y - center.y) / zoom;
                    
                    // ── Control Point Hover Detection ────────────────────
                    const float CP_HIT_RADIUS = 12.0f;
                    int hovered_cp = -1;
                    for (int i = 0; i < (int)shape_control_points.size(); ++i) {
                        ImVec2 sp = ImVec2(center.x + shape_control_points[i].x * zoom,
                                           center.y - shape_control_points[i].y * zoom);
                        float dx = mouse_pos.x - sp.x, dy = mouse_pos.y - sp.y;
                        if (dx*dx + dy*dy <= CP_HIT_RADIUS*CP_HIT_RADIUS) {
                            hovered_cp = i;
                            break;
                        }
                    }
                    
                    // ── Mouse Click: prioritize control point grab over strike ──
                    if (play_hovered && ImGui::IsMouseClicked(0)) {
                        if (app_mode == PLACING_HOLE) {
                            // Place a hole at click position
                            shape_holes.push_back({wx, wy, (double)new_hole_a, (double)new_hole_b});
                            app_mode = PLAY;
                            
                            // Auto-rebuild mesh with the new hole
                            meshes.assign(1, Mesh());
                            auto hole_polys = holes_to_polygons();
                            auto outer = compute_outer_boundary();
                            if (!outer.empty()) {
                                meshes[0].generate_from_polygon(outer, hole_polys, mesh_density);
                            }
                            
                            fems.clear();
                            all_modes.clear();
                            if (meshes[0].elements.size() > 0) {
                                FEM f;
                                f.assemble(meshes[0]);
                                fems.push_back(f);
                                all_modes.push_back(Solver::solve(f, n_modes));
                                audio.precompute_MU(fems[0], all_modes[0]);
                                audio.compute_xi(meshes[0]);
                                audio.compute_radiation_weights(all_modes[0]);
                            }
                        } else if (hovered_cp != -1) {
                            // Grab control point
                            dragged_cp_idx = hovered_cp;
                        } else {
                            // Strike the drum
                            int hit_mesh_idx = -1;
                            for (size_t m_idx = 0; m_idx < meshes.size(); ++m_idx) {
                                const auto& sub_mesh = meshes[m_idx];
                                for (const auto& el : sub_mesh.elements) {
                                    Vertex v1 = sub_mesh.vertices[el.v[0]];
                                    Vertex v2 = sub_mesh.vertices[el.v[1]];
                                    Vertex v3 = sub_mesh.vertices[el.v[2]];
                                    double denom = ((v2.y - v3.y)*(v1.x - v3.x) + (v3.x - v2.x)*(v1.y - v3.y));
                                    if (std::abs(denom) < 1e-12) continue;
                                    double a = ((v2.y - v3.y)*(wx - v3.x) + (v3.x - v2.x)*(wy - v3.y)) / denom;
                                    double b = ((v3.y - v1.y)*(wx - v3.x) + (v1.x - v3.x)*(wy - v3.y)) / denom;
                                    double c = 1.0 - a - b;
                                    if (a >= -1e-6 && b >= -1e-6 && c >= -1e-6) {
                                        hit_mesh_idx = m_idx;
                                        break;
                                    }
                                }
                                if (hit_mesh_idx != -1) break;
                            }
                            if (hit_mesh_idx != -1) {
                                dragged_mesh_idx = hit_mesh_idx;
                                last_struck_mesh = hit_mesh_idx;
                                audio.trigger_strike(meshes[hit_mesh_idx], all_modes[hit_mesh_idx], wx, wy, 1.0);
                            }
                        }
                    }
                    
                    // ── Control Point Dragging ───────────────────────────
                    if (dragged_cp_idx != -1 && ImGui::IsMouseDown(0)) {
                        if (current_shape == ELLIPSE) {
                            // Ellipse: constrain axis handles
                            double min_gap = std::max(0.01, 3.0 / (double)mesh_density);
                            if (dragged_cp_idx == 0) {
                                shape_control_points[0].x = std::max(min_gap, std::abs(wx));
                                shape_control_points[0].y = 0.0;
                            } else {
                                shape_control_points[1].x = 0.0;
                                shape_control_points[1].y = std::max(min_gap, std::abs(wy));
                            }
                        } else if (current_shape == ANNULUS) {
                            // Annulus: both handles on x-axis, with gap constraint
                            double min_gap = 3.0 / (double)mesh_density;
                            double r = std::max(min_gap, std::abs(wx));
                            if (dragged_cp_idx == 0) {
                                // Outer radius handle
                                if (r <= shape_control_points[1].x + min_gap)
                                    r = shape_control_points[1].x + min_gap;
                                shape_control_points[0] = {r, 0.0};
                            } else {
                                // Inner radius handle
                                if (r >= shape_control_points[0].x - min_gap)
                                    r = shape_control_points[0].x - min_gap;
                                shape_control_points[1] = {r, 0.0};
                            }
                        } else {
                            // Polygon: free movement
                            shape_control_points[dragged_cp_idx].x = wx;
                            shape_control_points[dragged_cp_idx].y = wy;
                        }
                        cp_dirty = true;
                    }
                    
                    // ── Mouse Release: rebuild if control points changed ─
                    if (ImGui::IsMouseReleased(0)) {
                        if (dragged_cp_idx != -1 && cp_dirty) {
                            // Rebuild mesh from updated control points
                            meshes.assign(1, Mesh());
                            if (current_shape == ELLIPSE) {
                                ellipse_a = (float)shape_control_points[0].x;
                                ellipse_b = (float)shape_control_points[1].y;
                                auto hole_polys = holes_to_polygons();
                                if (hole_polys.empty()) {
                                    meshes[0].generate_ellipse(ellipse_a, ellipse_b, mesh_density);
                                } else {
                                    auto outer = compute_outer_boundary();
                                    meshes[0].generate_from_polygon(outer, hole_polys, mesh_density);
                                }
                            } else if (current_shape == ANNULUS) {
                                annulus_outer_r = (float)shape_control_points[0].x;
                                annulus_inner_r = (float)shape_control_points[1].x;
                                meshes[0].generate_annulus(annulus_outer_r, annulus_inner_r, mesh_density);
                            } else {
                                // Regular polygon or custom: use control points as polygon
                                active_polygon = shape_control_points;
                                current_shape = CUSTOM;
                                auto hole_polys = holes_to_polygons();
                                meshes[0].generate_from_polygon(active_polygon, hole_polys, mesh_density);
                            }
                            
                            fems.clear();
                            all_modes.clear();
                            if (meshes[0].elements.size() > 0) {
                                FEM f;
                                f.assemble(meshes[0]);
                                fems.push_back(f);
                                all_modes.push_back(Solver::solve(f, n_modes));
                                audio.precompute_MU(fems[0], all_modes[0]);
                                audio.compute_xi(meshes[0]);
                                audio.compute_radiation_weights(all_modes[0]);
                                freqs.clear();
                                {
                                    std::lock_guard<std::mutex> lock(audio.mutex);
                                    audio.modes_state.count = 0;
                                    memset(audio.modes_state.active, 0, sizeof(audio.modes_state.active));
                                }
                            }
                            cp_dirty = false;
                        }
                        dragged_cp_idx = -1;
                        dragged_mesh_idx = -1;
                    }
                    
                    // ── Mesh Dragging (whole mesh translate) ─────────────
                    if (dragged_cp_idx == -1 && ImGui::IsMouseDragging(0) && dragged_mesh_idx != -1 && dragged_mesh_idx < (int)meshes.size()) {
                        ImVec2 delta = ImGui::GetIO().MouseDelta;
                        double dx = delta.x / zoom;
                        double dy = -delta.y / zoom;
                        
                        // If dragging the primary mesh, ensure control points and holes follow
                        if (dragged_mesh_idx == 0) {
                            if (current_shape == ELLIPSE || current_shape == ANNULUS) {
                                // Auto-convert parametric primitives to CUSTOM when manually moved, 
                                // so the handles properly track the shape globally
                                active_polygon = compute_outer_boundary();
                                if (!active_polygon.empty()) {
                                    shape_control_points = active_polygon;
                                    current_shape = CUSTOM;
                                }
                            }
                            
                            for (auto& cp : shape_control_points) {
                                cp.x += dx;
                                cp.y += dy;
                            }
                            for (auto& p : active_polygon) {
                                p.x += dx;
                                p.y += dy;
                            }
                            for (auto& h : shape_holes) {
                                h.cx += dx;
                                h.cy += dy;
                            }
                        }

                        // Translate the actual mesh geometry
                        for (auto& v : meshes[dragged_mesh_idx].vertices) {
                            v.x += dx;
                            v.y += dy;
                        }
                    }
    
                    // ── Draw Mesh Triangles ──────────────────────────────
                    for (const auto& m : meshes) {
                        for (const auto& el : m.elements) {
                            ImVec2 p[3];
                            for(int i = 0; i < 3; ++i) {
                                Vertex v = m.vertices[el.v[i]];
                                p[i] = ImVec2(center.x + v.x * zoom, center.y - v.y * zoom);
                            }
                            draw_list->AddTriangleFilled(p[0], p[1], p[2], IM_COL32(40, 50, 70, 180));
                            draw_list->AddTriangle(p[0], p[1], p[2], IM_COL32(100, 150, 200, 200), 1.0f);
                        }
                        
                        // Draw boundary nodes as small dots
                        for (int idx : m.boundary_nodes) {
                            if (idx >= 0 && idx < (int)m.vertices.size()) {
                                ImVec2 bp = ImVec2(center.x + m.vertices[idx].x * zoom,
                                                   center.y - m.vertices[idx].y * zoom);
                                draw_list->AddCircleFilled(bp, 2.0f, IM_COL32(255, 100, 100, 120));
                            }
                        }
                    }
                    
                    // ── Draw Control Points ──────────────────────────────
                    for (int i = 0; i < (int)shape_control_points.size(); ++i) {
                        ImVec2 sp = ImVec2(center.x + shape_control_points[i].x * zoom,
                                           center.y - shape_control_points[i].y * zoom);
                        ImU32 col;
                        float r;
                        if (i == dragged_cp_idx) {
                            col = IM_COL32(0, 220, 255, 255);    // Cyan: dragging
                            r = 9.0f;
                        } else if (i == hovered_cp) {
                            col = IM_COL32(255, 255, 255, 255);  // White: hover
                            r = 8.0f;
                        } else {
                            col = IM_COL32(220, 60, 60, 255);    // Red: normal
                            r = 6.0f;
                        }
                        draw_list->AddCircleFilled(sp, r, col);
                        draw_list->AddCircle(sp, r + 1.5f, IM_COL32(0, 0, 0, 150), 0, 1.5f);
                        
                        // Label for ellipse/annulus handles
                        if (current_shape == ELLIPSE) {
                            const char* label = (i == 0) ? "a" : "b";
                            draw_list->AddText(ImVec2(sp.x + r + 4, sp.y - 6), IM_COL32(255, 200, 100, 200), label);
                        } else if (current_shape == ANNULUS) {
                            const char* label = (i == 0) ? "R" : "r";
                            draw_list->AddText(ImVec2(sp.x + r + 4, sp.y - 6), IM_COL32(255, 200, 100, 200), label);
                        }
                    }
                    
                    // ── Draw Hole Outlines ───────────────────────────────
                    for (const auto& h : shape_holes) {
                        int n_seg = 48;
                        for (int i = 0; i < n_seg; ++i) {
                            double t0 = 2.0 * M_PI * i / n_seg;
                            double t1 = 2.0 * M_PI * (i + 1) / n_seg;
                            ImVec2 p0(center.x + (h.cx + h.a * std::cos(t0)) * zoom,
                                      center.y - (h.cy + h.b * std::sin(t0)) * zoom);
                            ImVec2 p1(center.x + (h.cx + h.a * std::cos(t1)) * zoom,
                                      center.y - (h.cy + h.b * std::sin(t1)) * zoom);
                            draw_list->AddLine(p0, p1, IM_COL32(255, 80, 255, 200), 2.0f);
                        }
                        // Center dot
                        ImVec2 hc(center.x + h.cx * zoom, center.y - h.cy * zoom);
                        draw_list->AddCircleFilled(hc, 3.0f, IM_COL32(255, 80, 255, 180));
                    }
                    
                    // ── Preview hole at cursor during PLACING_HOLE ───────
                    if (app_mode == PLACING_HOLE && play_hovered) {
                        int n_seg = 48;
                        for (int i = 0; i < n_seg; ++i) {
                            double t0 = 2.0 * M_PI * i / n_seg;
                            double t1 = 2.0 * M_PI * (i + 1) / n_seg;
                            ImVec2 p0(center.x + (wx + new_hole_a * std::cos(t0)) * zoom,
                                      center.y - (wy + new_hole_b * std::sin(t0)) * zoom);
                            ImVec2 p1(center.x + (wx + new_hole_a * std::cos(t1)) * zoom,
                                      center.y - (wy + new_hole_b * std::sin(t1)) * zoom);
                            draw_list->AddLine(p0, p1, IM_COL32(255, 200, 50, 120), 1.5f);
                        }
                    }
                }
                
                ImGui::End();
            }

            // ═══════════════════════════════════════════════════════════════
            // 3. Spectrum Plot
            // ═══════════════════════════════════════════════════════════════
            {
                ImGui::SetNextWindowPos(ImVec2(10, 370), ImGuiCond_FirstUseEver);
                ImGui::SetNextWindowSize(ImVec2(300, 340), ImGuiCond_FirstUseEver);
                ImGui::Begin("Frequency Spectrum");
                
                // Ensure safe bounds
                if (last_struck_mesh >= (int)all_modes.size()) {
                    last_struck_mesh = 0;
                }
                
                if (!all_modes.empty()) {
                    const ModeData& active_modes_data = all_modes[last_struck_mesh];
                    
                    // Update Data Sizes
                    if (freqs.size() != active_modes_data.eigenvalues.size()) {
                        freqs.resize(active_modes_data.eigenvalues.size());
                        amps.resize(active_modes_data.eigenvalues.size());
                    }
                    
                    // ALWAYS update physical frequencies
                    for(size_t i = 0; i < active_modes_data.eigenvalues.size(); ++i) {
                        freqs[i] = (float)audio.eigenvalue_to_freq(active_modes_data.eigenvalues[i]);
                    }
                    
                    // Get current amplitudes from Audio Engine (in dB)
                    // Show radiated pressure per mode (not raw displacement)
                    {
                        std::lock_guard<std::mutex> lock(audio.mutex);
                        int ci = audio.cur_idx.load(std::memory_order_acquire);
                        const CoeffSet& cc = audio.coeff[ci];
                        double phys_scale = audio.rho_air / (2.0 * M_PI * audio.listener_distance);
                        
                        if (audio.modes_state.count == (int)active_modes_data.eigenvalues.size()) {
                            for(size_t i = 0; i < active_modes_data.eigenvalues.size(); ++i) {
                                float zr = audio.modes_state.Z_re[i];
                                float zi = audio.modes_state.Z_im[i];
                                // Acceleration: a = s²·Z
                                float ar = cc.s2_re[i] * zr - cc.s2_im[i] * zi;
                                float ai = cc.s2_re[i] * zi + cc.s2_im[i] * zr;
                                float p = 2.0f * (cc.Phi_re[i] * ar + cc.Phi_im[i] * ai) * (float)phys_scale;
                                float mag = std::fabs(p);
                                float db = (mag > 1e-20f) ? 20.0f * std::log10(mag) : -120.0f;
                                // Shift so -120 dB is drawn at y=0, loud sounds shoot upward
                                amps[i] = std::max(0.0f, db + 120.0f);
                            }
                        } else {
                            std::fill(amps.begin(), amps.end(), 0.0f);
                        }
                    }
                    
                    if (ImPlot::BeginPlot("Modes", ImVec2(-1, -1))) {
                        ImPlot::SetupAxes("Frequency (Hz)", "Relative Amplitude (dB from floor)");
                        ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, 140.0, ImPlotCond_Always);
                        ImPlot::PlotBars("Modes", freqs.data(), amps.data(), freqs.size(), 1.0f);
                        ImPlot::EndPlot();
                    }
                }
                ImGui::End();
            }
    
            // Rendering
            ImGui::Render();
            SDL_RenderSetScale(renderer, io.DisplayFramebufferScale.x, io.DisplayFramebufferScale.y);
            SDL_SetRenderDrawColor(renderer, 20, 20, 20, 255);
            SDL_RenderClear(renderer);
            ImGui_ImplSDLRenderer2_RenderDrawData(ImGui::GetDrawData());
            SDL_RenderPresent(renderer);
        } else {
             // Headless loop
             SDL_Event event;
             while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) done = true;
             }
             SDL_Delay(16);
        }
    }

    // Cleanup
    if (window && renderer) {
        ImGui_ImplSDLRenderer2_Shutdown();
        ImGui_ImplSDL2_Shutdown();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
    }
    if (audio_enabled && dev != 0) {
        SDL_CloseAudioDevice(dev);
    }
    SDL_Quit();

    return 0;
}
