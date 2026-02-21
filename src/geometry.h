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

    // ─── Generate mesh from arbitrary closed polygon ────────────────────────
    void generate_from_polygon(const std::vector<Vertex>& boundary_poly, int density = 15) {
        clear();
        if (boundary_poly.size() < 3) return;

        int n_boundary = boundary_poly.size();

        // 1. Compute bounding box
        double min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
        for (const auto& v : boundary_poly) {
            min_x = std::min(min_x, v.x);
            max_x = std::max(max_x, v.x);
            min_y = std::min(min_y, v.y);
            max_y = std::max(max_y, v.y);
        }

        double width = max_x - min_x;
        double height = max_y - min_y;
        double scale = std::max(width, height);
        double step = scale / density;

        // 2. Add boundary vertices
        for (int i = 0; i < n_boundary; ++i) {
            vertices.push_back(boundary_poly[i]);
            boundary_nodes.push_back(i);
        }

        // 3. Add interior grid points that pass point-in-polygon test
        // Add margin to avoid points too close to boundary
        double margin = step * 0.3;
        for (double y = min_y + step * 0.5; y < max_y; y += step) {
            for (double x = min_x + step * 0.5; x < max_x; x += step) {
                if (point_in_polygon(x, y, boundary_poly)) {
                    // Check distance to all boundary edges to avoid degenerate triangles
                    bool too_close = false;
                    for (int i = 0; i < n_boundary; ++i) {
                        int j = (i + 1) % n_boundary;
                        double ex = boundary_poly[j].x - boundary_poly[i].x;
                        double ey = boundary_poly[j].y - boundary_poly[i].y;
                        double len = std::sqrt(ex*ex + ey*ey);
                        if (len < 1e-10) continue;
                        // Distance from point to line segment
                        double t = ((x - boundary_poly[i].x)*ex + (y - boundary_poly[i].y)*ey) / (len*len);
                        t = std::max(0.0, std::min(1.0, t));
                        double cx = boundary_poly[i].x + t * ex;
                        double cy = boundary_poly[i].y + t * ey;
                        double dist = std::sqrt((x-cx)*(x-cx) + (y-cy)*(y-cy));
                        if (dist < margin) {
                            too_close = true;
                            break;
                        }
                    }
                    if (!too_close) {
                        vertices.push_back({x, y});
                    }
                }
            }
        }

        // 4. Run CDT (Constrained Delaunay Triangulation)
        CDT::Triangulation<double> cdt;

        // Add all vertices
        std::vector<CDT::V2d<double>> cdt_verts;
        for (const auto& v : vertices) {
            cdt_verts.push_back(CDT::V2d<double>::make(v.x, v.y));
        }
        cdt.insertVertices(cdt_verts);

        // Add boundary edges as constraints
        std::vector<CDT::Edge> edges;
        for (int i = 0; i < n_boundary; ++i) {
            edges.push_back(CDT::Edge(i, (i + 1) % n_boundary));
        }
        cdt.insertEdges(edges);

        // Erase outer triangles and super triangle
        cdt.eraseOuterTrianglesAndHoles();

        // 5. Extract triangulation result
        // CDT may have added Steiner points; rebuild our vertex list from CDT's
        vertices.clear();
        boundary_nodes.clear();
        
        // CDT vertices include original + any Steiner points
        std::set<int> boundary_set;
        for (int i = 0; i < n_boundary; ++i) {
            boundary_set.insert(i);
        }

        for (size_t i = 0; i < cdt.vertices.size(); ++i) {
            vertices.push_back({cdt.vertices[i].x, cdt.vertices[i].y});
            if (boundary_set.count(i)) {
                boundary_nodes.push_back(i);
            }
        }

        // Extract triangles
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
        generate_from_polygon(boundary, density);
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
        generate_from_polygon(boundary, density);
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
        m1.generate_from_polygon(b1, density);
        m2.generate_from_polygon(b2, density);

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
