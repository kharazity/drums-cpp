
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
enum AppMode { PLAY, DRAWING };

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
    enum ShapeType { REGULAR_POLYGON, ELLIPSE, ISOSPECTRAL, CUSTOM };
    ShapeType current_shape = REGULAR_POLYGON;
    AppMode app_mode = PLAY;
    std::vector<Vertex> draw_points; // Polygon being drawn (world coords)
    std::vector<Vertex> active_polygon; // Saved custom polygon
    
    // Shape Parameters
    float ellipse_a = 1.0f;
    float ellipse_b = 0.8f;
    int polygon_n = 4;
    float polygon_L = 1.0f;
    
    // Spectrum Data
    std::vector<float> freqs;
    std::vector<float> amps;
    int last_struck_mesh = 0;
    int dragged_mesh_idx = -1;
    char export_wav_path[256] = "output/strike.wav";
    
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
                ImGui::SetNextWindowSize(ImVec2(300, 350), ImGuiCond_FirstUseEver);
                ImGui::Begin("Controls");
                ImGui::Text("FPS: %.1f", io.Framerate);
                ImGui::Separator();
                
                // ── Shape Selection ──────────────────────────────────────
                const char* shape_names[] = { "Regular Polygon", "Ellipse", "Isospectral Drums", "Custom (Draw)" };
                int current_shape_idx = (int)current_shape;
                
                if (ImGui::Combo("Shape Type", &current_shape_idx, shape_names, IM_ARRAYSIZE(shape_names))) {
                    current_shape = (ShapeType)current_shape_idx;
                    if (current_shape == CUSTOM) {
                        app_mode = DRAWING;
                        draw_points.clear();
                    } else {
                        app_mode = PLAY;
                    }
                }
                
                // Shape-specific parameters
                if (current_shape == ELLIPSE) {
                    ImGui::SliderFloat("Semi-major (a)", &ellipse_a, 0.1f, 3.0f);
                    ImGui::SliderFloat("Semi-minor (b)", &ellipse_b, 0.1f, 3.0f);
                } else if (current_shape == REGULAR_POLYGON) {
                    ImGui::SliderInt("Sides (n)", &polygon_n, 3, 20);
                    ImGui::SliderFloat("Side Length (L)", &polygon_L, 0.1f, 3.0f);
                }

                // Parameters requiring Rebuild
                if (ImGui::InputInt("Modes", &n_modes, 1, 10)) {
                    if (n_modes < 1) n_modes = 1;
                }
                ImGui::InputInt("Mesh Density", &mesh_density);
                if (mesh_density < 5) mesh_density = 5;
                if (mesh_density > 1000000) mesh_density = 1000000;

                if (app_mode == PLAY) {
                    if (ImGui::Button("Rebuild Mesh")) {
                        if (current_shape == ISOSPECTRAL) {
                            meshes = Mesh::generate_isospectral(mesh_density);
                        } else {
                            meshes.assign(1, Mesh());
                            if (current_shape == ELLIPSE) {
                                meshes[0].generate_ellipse(ellipse_a, ellipse_b, mesh_density);
                            } else if (current_shape == REGULAR_POLYGON) {
                                meshes[0].generate_regular_polygon(polygon_n, polygon_L, mesh_density);
                            } else if (current_shape == CUSTOM && !active_polygon.empty()) {
                                meshes[0].generate_from_polygon(active_polygon, mesh_density);
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
                            meshes[0].generate_from_polygon(active_polygon, mesh_density);
                            
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

                ImGui::Separator();
                
                if (app_mode == PLAY) {
                    if (ImGui::Button("Strike Center")) {
                        if (!meshes.empty() && !meshes[0].vertices.empty() && !all_modes.empty()) {
                            // Strike at mesh centroid
                            double cx = 0, cy = 0;
                            for (const auto& v : meshes[0].vertices) { cx += v.x; cy += v.y; }
                            cx /= meshes[0].vertices.size();
                            cy /= meshes[0].vertices.size();
                            audio.trigger_strike(meshes[0], all_modes[0], cx, cy, 1.0);
                        }
                    }
                }
                
                // ── Physical Parameters ──────────────────────────────────
                static float alpha0_f = (float)audio.alpha0;
                if (ImGui::SliderFloat("Damping (a0)", &alpha0_f, 0.1f, 50.0f, "%.1f")) {
                    audio.alpha0 = (double)alpha0_f;
                    if (!all_modes.empty()) audio.compute_radiation_weights(all_modes[0]);
                }
                static float alpha1_f = (float)audio.alpha1;
                if (ImGui::SliderFloat("Freq Damping (a1)", &alpha1_f, 0.0f, 0.01f, "%.5f")) {
                    audio.alpha1 = (double)alpha1_f;
                    if (!all_modes.empty()) audio.compute_radiation_weights(all_modes[0]);
                }
                static float tension_f = (float)audio.tension;
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
                
                static float v0_f = (float)audio.strike_v0;
                if (ImGui::SliderFloat("Strike Velocity (v0)", &v0_f, 0.1f, 10.0f, "%.1f", ImGuiSliderFlags_Logarithmic)) {
                    audio.strike_v0 = (double)v0_f;
                }
                
                static float delta_f = (float)audio.strike_width_delta;
                if (ImGui::SliderFloat("Mallet Width", &delta_f, 0.001f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic)) {
                    audio.strike_width_delta = (double)delta_f;
                }
                
                static float vol_f = (float)audio.master_volume;
                if (ImGui::SliderFloat("Master Volume", &vol_f, 0.01f, 100.0f, "%.2f", ImGuiSliderFlags_Logarithmic)) {
                    audio.master_volume = (double)vol_f;
                }
                
                ImGui::SliderFloat("Zoom", &zoom, 50.0f, 1000.0f);

                ImGui::Separator();
                ImGui::Text("Listener Direction");
                static float elevation_f = (float)audio.listener_elevation;
                if (ImGui::SliderFloat("Elevation (deg)", &elevation_f, 0.0f, 90.0f, "%.1f")) {
                    audio.listener_elevation = (double)elevation_f;
                    if (!meshes.empty()) audio.compute_xi(meshes[0]);
                    if (!all_modes.empty()) audio.compute_radiation_weights(all_modes[0]);
                }
                static float azimuth_f = (float)audio.listener_azimuth;
                if (ImGui::SliderFloat("Azimuth (deg)", &azimuth_f, 0.0f, 360.0f, "%.1f")) {
                    audio.listener_azimuth = (double)azimuth_f;
                    if (!meshes.empty()) audio.compute_xi(meshes[0]);
                    if (!all_modes.empty()) audio.compute_radiation_weights(all_modes[0]);
                }

                ImGui::Separator();
                ImGui::Text("c = %.1f m/s", audio.wave_speed());
                
                int total_verts = 0;
                for (const auto& m : meshes) total_verts += m.vertices.size();
                ImGui::Text("Meshes: %lu  Total Vertices: %d", meshes.size(), total_verts);

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
                
                ImVec2 center = ImVec2(p0.x + sz.x*0.5f, p0.y + sz.y*0.5f);
                if (sz.x < 50 || sz.y < 50) {
                    center = ImVec2(1280/2, 720/2);
                }

                // ── DRAWING MODE ─────────────────────────────────────────
                if (app_mode == DRAWING) {
                    // Add invisible button to capture clicks in the window area
                    ImGui::InvisibleButton("draw_canvas", sz);
                    
                    if (ImGui::IsItemClicked(0)) {
                        ImVec2 mouse_pos = ImGui::GetMousePos();
                        // Convert screen coords → world coords
                        double wx = (mouse_pos.x - center.x) / zoom;
                        double wy = -(mouse_pos.y - center.y) / zoom;
                        draw_points.push_back({wx, wy});
                    }
                    
                    // Draw existing points and edges
                    for (int i = 0; i < (int)draw_points.size(); ++i) {
                        ImVec2 sp = ImVec2(center.x + draw_points[i].x * zoom,
                                           center.y - draw_points[i].y * zoom);
                        
                        // Draw vertex dot
                        draw_list->AddCircleFilled(sp, 5.0f, IM_COL32(255, 200, 50, 255));
                        
                        // Draw edge to next point
                        if (i > 0) {
                            ImVec2 prev = ImVec2(center.x + draw_points[i-1].x * zoom,
                                                 center.y - draw_points[i-1].y * zoom);
                            draw_list->AddLine(prev, sp, IM_COL32(255, 200, 50, 200), 2.5f);
                        }
                    }
                    
                    // Draw closing edge preview (dotted feel via thinner line)
                    if (draw_points.size() >= 3) {
                        ImVec2 first = ImVec2(center.x + draw_points[0].x * zoom,
                                              center.y - draw_points[0].y * zoom);
                        ImVec2 last = ImVec2(center.x + draw_points.back().x * zoom,
                                             center.y - draw_points.back().y * zoom);
                        draw_list->AddLine(last, first, IM_COL32(255, 200, 50, 100), 1.5f);
                    }
                    
                    // Draw cursor line from last point to mouse
                    if (!draw_points.empty()) {
                        ImVec2 last_sp = ImVec2(center.x + draw_points.back().x * zoom,
                                                center.y - draw_points.back().y * zoom);
                        draw_list->AddLine(last_sp, ImGui::GetMousePos(), IM_COL32(255, 255, 255, 100), 1.0f);
                    }
                    
                    // Help text
                    draw_list->AddText(ImVec2(p0.x + 10, p0.y + 10), IM_COL32(255, 200, 50, 255), 
                                       "Click to add points. Use 'Finish & Build' when done.");
                }
                // ── PLAY MODE ────────────────────────────────────────────
                else {
                    // Add invisible button to capture clicks in the window area so the window doesn't drag
                    ImGui::InvisibleButton("play_canvas", sz);
                    
                    // Interaction: Strike and Drag
                    ImVec2 mouse_pos = ImGui::GetMousePos();
                    double wx = (mouse_pos.x - center.x) / zoom;
                    double wy = -(mouse_pos.y - center.y) / zoom;
                    
                    if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0)) {
                        int closest_node = -1;
                        int hit_mesh_idx = -1;
                        
                        // 1. First check if we actually clicked inside a triangle using Barycentric coordinates
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
                                
                                // Slight margin for floating point on edges
                                if (a >= -1e-6 && b >= -1e-6 && c >= -1e-6) {
                                    // 2. If inside, find the closest of its 3 governing vertices to center the strike
                                    double d1 = (wx - v1.x)*(wx - v1.x) + (wy - v1.y)*(wy - v1.y);
                                    double d2 = (wx - v2.x)*(wx - v2.x) + (wy - v2.y)*(wy - v2.y);
                                    double d3 = (wx - v3.x)*(wx - v3.x) + (wy - v3.y)*(wy - v3.y);
                                    
                                    if (d1 <= d2 && d1 <= d3) closest_node = el.v[0];
                                    else if (d2 <= d1 && d2 <= d3) closest_node = el.v[1];
                                    else closest_node = el.v[2];
                                    
                                    hit_mesh_idx = m_idx;
                                    break;
                                }
                            }
                            if (hit_mesh_idx != -1) break;
                        }
                        
                        if (hit_mesh_idx != -1) {
                            dragged_mesh_idx = hit_mesh_idx; // Begin dragging
                            last_struck_mesh = hit_mesh_idx;
                            // Far-field radiation: no pickup node needed
                            audio.trigger_strike(meshes[hit_mesh_idx], all_modes[hit_mesh_idx], wx, wy, 1.0);
                        }
                    }
                    
                    if (ImGui::IsMouseReleased(0)) {
                        dragged_mesh_idx = -1;
                    }
                    
                    if (ImGui::IsMouseDragging(0) && dragged_mesh_idx != -1 && dragged_mesh_idx < (int)meshes.size()) {
                        ImVec2 delta = ImGui::GetIO().MouseDelta;
                        double dx = delta.x / zoom;
                        double dy = -delta.y / zoom;
                        for (auto& v : meshes[dragged_mesh_idx].vertices) {
                            v.x += dx;
                            v.y += dy;
                        }
                    }
    
                    // Draw Mesh Triangles
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
                        
                        // Draw boundary nodes as bright dots
                        for (int idx : m.boundary_nodes) {
                            if (idx >= 0 && idx < (int)m.vertices.size()) {
                                ImVec2 bp = ImVec2(center.x + m.vertices[idx].x * zoom,
                                                   center.y - m.vertices[idx].y * zoom);
                                draw_list->AddCircleFilled(bp, 3.0f, IM_COL32(255, 100, 100, 200));
                            }
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
                    
                    // Update Data (use physical frequencies)
                    if (freqs.size() != active_modes_data.eigenvalues.size()) {
                        freqs.resize(active_modes_data.eigenvalues.size());
                        amps.resize(active_modes_data.eigenvalues.size());
                        for(size_t i = 0; i < active_modes_data.eigenvalues.size(); ++i) {
                            freqs[i] = (float)audio.eigenvalue_to_freq(active_modes_data.eigenvalues[i]);
                            amps[i] = 0.0f; 
                        }
                    }
                    
                    // Get current amplitudes from Audio Engine (in dB)
                    {
                        std::lock_guard<std::mutex> lock(audio.mutex);
                        if (audio.modes_state.count == (int)active_modes_data.eigenvalues.size()) {
                            for(size_t i = 0; i < active_modes_data.eigenvalues.size(); ++i) {
                                float zr = audio.modes_state.Z_re[i];
                                float zi = audio.modes_state.Z_im[i];
                                float mag = std::sqrt(zr*zr + zi*zi);
                                amps[i] = (mag > 1e-20f) ? 20.0f * std::log10(mag) : -120.0f;
                            }
                        }
                    }
                    
                    if (ImPlot::BeginPlot("Modes", ImVec2(-1, -1))) {
                        ImPlot::SetupAxes("Frequency (Hz)", "Amplitude (dB)");
                        ImPlot::SetupAxisLimits(ImAxis_Y1, -120.0, 0.0, ImPlotCond_Once);
                        ImPlot::PlotBars("Modes", freqs.data(), amps.data(), freqs.size(), 10.0f);
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
