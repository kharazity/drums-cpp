#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <set>
#include <Eigen/Dense>
#include <CDT.h>

struct Vertex {
    double x, y;
};

struct Element {
    int v[3]; // Triangle indices
};

class Mesh {
public:
    std::vector<Vertex> vertices;
    std::vector<Element> elements;
    std::vector<int> boundary_nodes; // Nodes on Dirichlet boundary

    void clear() {
        vertices.clear();
        elements.clear();
        boundary_nodes.clear();
    }

    // Generate Unit Square Mesh
    void generate_square(double size, int n_per_side) {
        clear();
        double half = size / 2.0;
        double dx = size / n_per_side;

        // Nodes
        for (int j = 0; j <= n_per_side; ++j) {
            for (int i = 0; i <= n_per_side; ++i) {
                vertices.push_back({-half + i * dx, -half + j * dx});
                
                // Boundary check
                if (i == 0 || i == n_per_side || j == 0 || j == n_per_side) {
                    boundary_nodes.push_back(vertices.size() - 1);
                }
            }
        }

        // Elements
        for (int j = 0; j < n_per_side; ++j) {
            for (int i = 0; i < n_per_side; ++i) {
                int n1 = j * (n_per_side + 1) + i;
                int n2 = n1 + 1;
                int n3 = (j + 1) * (n_per_side + 1) + i;
                int n4 = n3 + 1;

                // Two triangles per quad
                elements.push_back({n1, n2, n3});
                elements.push_back({n2, n4, n3});
            }
        }
    }

    // ─── Point-in-polygon test (ray casting) ────────────────────────────────
    static bool point_in_polygon(double px, double py, const std::vector<Vertex>& poly) {
        bool inside = false;
        int n = poly.size();
        for (int i = 0, j = n - 1; i < n; j = i++) {
            double xi = poly[i].x, yi = poly[i].y;
            double xj = poly[j].x, yj = poly[j].y;
            if (((yi > py) != (yj > py)) &&
                (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
                inside = !inside;
            }
        }
        return inside;
    }

    // ─── Generate mesh from arbitrary closed polygon (with optional holes) ──
    void generate_from_polygon(const std::vector<Vertex>& boundary_poly,
                               const std::vector<std::vector<Vertex>>& holes = {},
                               int density = 15) {
        clear();
        if (boundary_poly.size() < 3) return;

        int n_outer = (int)boundary_poly.size();

        // 1. Compute bounding box from outer boundary
        double min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
        for (const auto& v : boundary_poly) {
            min_x = std::min(min_x, v.x); max_x = std::max(max_x, v.x);
            min_y = std::min(min_y, v.y); max_y = std::max(max_y, v.y);
        }
        double width = max_x - min_x;
        double height = max_y - min_y;
        double scale = std::max(width, height);
        double step = scale / density;
        double margin = step * 0.3;

        // 2. Build all boundary edges for margin check
        //    (outer boundary edges + all hole edges)
        struct EdgeSeg { Vertex a, b; };
        std::vector<EdgeSeg> all_boundary_edges;
        // Outer edges
        for (int i = 0; i < n_outer; ++i) {
            int j = (i + 1) % n_outer;
            all_boundary_edges.push_back({boundary_poly[i], boundary_poly[j]});
        }
        // Hole edges
        for (const auto& hole : holes) {
            int nh = (int)hole.size();
            for (int i = 0; i < nh; ++i) {
                int j = (i + 1) % nh;
                all_boundary_edges.push_back({hole[i], hole[j]});
            }
        }

        // 3. Add outer boundary vertices
        for (int i = 0; i < n_outer; ++i) {
            vertices.push_back(boundary_poly[i]);
            boundary_nodes.push_back(i);
        }

        // 4. Add hole boundary vertices (tracking per-hole offsets)
        std::vector<int> hole_offsets;  // start index for each hole
        for (const auto& hole : holes) {
            int offset = (int)vertices.size();
            hole_offsets.push_back(offset);
            int nh = (int)hole.size();
            for (int i = 0; i < nh; ++i) {
                vertices.push_back(hole[i]);
                boundary_nodes.push_back(offset + i);
            }
        }

        // 5. Add interior grid points: inside outer, outside all holes,
        //    and not too close to any boundary edge
        for (double y = min_y + step * 0.5; y < max_y; y += step) {
            for (double x = min_x + step * 0.5; x < max_x; x += step) {
                if (!point_in_polygon(x, y, boundary_poly)) continue;

                // Reject if inside any hole
                bool in_hole = false;
                for (const auto& hole : holes) {
                    if (point_in_polygon(x, y, hole)) { in_hole = true; break; }
                }
                if (in_hole) continue;

                // Reject if too close to any boundary/hole edge
                bool too_close = false;
                for (const auto& edge : all_boundary_edges) {
                    double ex = edge.b.x - edge.a.x;
                    double ey = edge.b.y - edge.a.y;
                    double len = std::sqrt(ex*ex + ey*ey);
                    if (len < 1e-10) continue;
                    double t = ((x - edge.a.x)*ex + (y - edge.a.y)*ey) / (len*len);
                    t = std::max(0.0, std::min(1.0, t));
                    double cx = edge.a.x + t * ex;
                    double cy = edge.a.y + t * ey;
                    double dist = std::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy));
                    if (dist < margin) { too_close = true; break; }
                }
                if (!too_close) {
                    vertices.push_back({x, y});
                }
            }
        }

        // 6. Run CDT (Constrained Delaunay Triangulation)
        CDT::Triangulation<double> cdt;

        std::vector<CDT::V2d<double>> cdt_verts;
        for (const auto& v : vertices) {
            cdt_verts.push_back(CDT::V2d<double>::make(v.x, v.y));
        }
        cdt.insertVertices(cdt_verts);

        // Constraint edges: outer boundary loop
        std::vector<CDT::Edge> edges;
        for (int i = 0; i < n_outer; ++i) {
            edges.push_back(CDT::Edge(i, (i + 1) % n_outer));
        }
        // Constraint edges: each hole loop (per-loop wrapping)
        for (size_t h = 0; h < holes.size(); ++h) {
            int offset = hole_offsets[h];
            int nh = (int)holes[h].size();
            for (int i = 0; i < nh; ++i) {
                edges.push_back(CDT::Edge(offset + i, offset + (i + 1) % nh));
            }
        }
        cdt.insertEdges(edges);

        // Erase outer triangles AND hole triangles (depth peeling)
        cdt.eraseOuterTrianglesAndHoles();

        // 7. Extract triangulation result
        // Count total boundary vertices for boundary_set
        int total_boundary_verts = n_outer;
        for (const auto& hole : holes) total_boundary_verts += (int)hole.size();

        vertices.clear();
        boundary_nodes.clear();

        std::set<int> boundary_set;
        for (int i = 0; i < total_boundary_verts; ++i) {
            boundary_set.insert(i);
        }

        for (size_t i = 0; i < cdt.vertices.size(); ++i) {
            vertices.push_back({cdt.vertices[i].x, cdt.vertices[i].y});
            if (boundary_set.count(i)) {
                boundary_nodes.push_back(i);
            }
        }

        for (const auto& tri : cdt.triangles) {
            elements.push_back({(int)tri.vertices[0], (int)tri.vertices[1], (int)tri.vertices[2]});
        }

        std::cout << "Custom mesh: " << vertices.size() << " vertices, " 
                  << elements.size() << " triangles, "
                  << boundary_nodes.size() << " boundary nodes" << std::endl;
    }

    // ─── Generate Ellipse Mesh ──────────────────────────────────────────────
    void generate_ellipse(double a, double b, int density = 15) {
        std::vector<Vertex> boundary;
        
        // Perimeter approximation (Ramanujan) for dynamic point scaling
        double h = std::pow(a - b, 2) / std::pow(a + b, 2);
        double perimeter = M_PI * (a + b) * (1.0 + (3.0 * h) / (10.0 + std::sqrt(4.0 - 3.0 * h)));
        
        // Scale boundary points based on the mesh density
        double scale = 2.0 * std::max(a, b);
        int n_boundary = std::max(16, (int)(perimeter / (scale / density)));

        for (int i = 0; i < n_boundary; ++i) {
            double theta = 2.0 * M_PI * i / n_boundary;
            boundary.push_back({a * std::cos(theta), b * std::sin(theta)});
        }
        generate_from_polygon(boundary, {}, density);
    }

    // ─── Generate Annulus Mesh (ring with hole) ─────────────────────────────
    void generate_annulus(double outer_r, double inner_r, int density = 15) {
        // Enforce a strict minimum gap to prevent CDT from crashing due to overlapping boundaries
        double min_gap = 3.0 / (double)density;
        if (inner_r >= outer_r - min_gap) {
            inner_r = std::max(0.01, outer_r - min_gap);
        }
        if (inner_r >= outer_r) return; // Fallback safety

        // Outer circle boundary
        std::vector<Vertex> outer_boundary;
        int n_outer = std::max(24, (int)(2.0 * M_PI * outer_r * density / (2.0 * outer_r)));
        for (int i = 0; i < n_outer; ++i) {
            double theta = 2.0 * M_PI * i / n_outer;
            outer_boundary.push_back({outer_r * std::cos(theta), outer_r * std::sin(theta)});
        }

        // Inner circle boundary (same winding direction)
        std::vector<Vertex> inner_boundary;
        int n_inner = std::max(16, (int)(2.0 * M_PI * inner_r * density / (2.0 * outer_r)));
        for (int i = 0; i < n_inner; ++i) {
            double theta = 2.0 * M_PI * i / n_inner;
            inner_boundary.push_back({inner_r * std::cos(theta), inner_r * std::sin(theta)});
        }

        generate_from_polygon(outer_boundary, {inner_boundary}, density);
    }

    // ─── Generate Regular N-gon Mesh ────────────────────────────────────────
    void generate_regular_polygon(int n_sides, double side_length, int density = 15) {
        if (n_sides < 3) return;
        
        std::vector<Vertex> boundary;
        double circumradius = side_length / (2.0 * std::sin(M_PI / n_sides));
        
        // Start with a flat bottom edge for aesthetics (offset angle)
        double offset = M_PI / 2.0 - M_PI / n_sides; 

        for (int i = 0; i < n_sides; ++i) {
            double theta = offset + 2.0 * M_PI * i / n_sides;
            boundary.push_back({circumradius * std::cos(theta), circumradius * std::sin(theta)});
        }
        generate_from_polygon(boundary, {}, density);
    }

    // ─── Generate Isospectral Drums (Gordon-Webb-Wolpert) ───────────────────
    static std::vector<Mesh> generate_isospectral(int density = 15) {
        // Base scale for the shapes to fit on screen
        double scale = 0.5;
        
        // Shape 1 vertices (Left Drum)
        std::vector<Vertex> b1 = {
            {0, 2}, {1, 3}, {1, 2}, {3, 2}, {3, 1}, {2, 1}, {2, 0}
        };
        // Shape 2 vertices (Right Drum)
        std::vector<Vertex> b2 = {
            {0, 3}, {1, 3}, {1, 2}, {2, 2}, {3, 1}, {2, 1}, {2, 0}, {0, 2}
        };
        
        // Apply scaling
        for (auto& v : b1) { v.x *= scale; v.y *= scale; }
        for (auto& v : b2) { v.x *= scale; v.y *= scale; }

        // Generate independent meshes
        Mesh m1, m2;
        m1.generate_from_polygon(b1, {}, density);
        m2.generate_from_polygon(b2, {}, density);

        // Center m1 bounding box on X, push left
        // Center m2 bounding box on X, push right
        double offset_x = 1.0; 
        
        // Shift M1 (left)
        for (auto& v : m1.vertices) {
            v.x -= offset_x;
            v.y -= scale * 1.5; // Roughly center Y
        }

        // Shift M2 (right)
        for (auto& v : m2.vertices) {
            v.x += offset_x - scale * 1.5; 
            v.y -= scale * 1.5; 
        }

        std::cout << "Isospectral meshes generated. M1 Vertices: " << m1.vertices.size() 
                  << " | M2 Vertices: " << m2.vertices.size() << std::endl;
                  
        return {m1, m2};
    }
};
