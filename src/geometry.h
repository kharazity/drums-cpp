#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <set>
#include <cstring>
#include <Eigen/Dense>

#include "triangle.h"

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

    // Square mesh: regular 4-gon with given side length and mesh density
    void generate_square(double size, int n_per_side) {
        generate_regular_polygon(4, size, n_per_side);
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
    // Uses Shewchuk's Triangle with quality refinement (min angle ~20°) and optional max area.
    void generate_from_polygon(const std::vector<Vertex>& boundary_poly,
                               const std::vector<std::vector<Vertex>>& holes = {},
                               int density = 15) {
        clear();
        if (boundary_poly.size() < 3) return;

        const int n_outer = (int)boundary_poly.size();

        // Bounding box and scale for optional area constraint
        double min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;
        for (const auto& v : boundary_poly) {
            min_x = std::min(min_x, v.x); max_x = std::max(max_x, v.x);
            min_y = std::min(min_y, v.y); max_y = std::max(max_y, v.y);
        }
        double scale = std::max(max_x - min_x, max_y - min_y);
        if (scale < 1e-10) scale = 1.0;
        double max_area = (scale / (double)density) * (scale / (double)density) * 0.5;

        // Build point list: outer boundary then hole boundaries
        std::vector<TRI_REAL> pointlist;
        std::vector<int> pointmarkerlist;
        for (const auto& v : boundary_poly) {
            pointlist.push_back((TRI_REAL)v.x);
            pointlist.push_back((TRI_REAL)v.y);
            pointmarkerlist.push_back(1);
        }
        std::vector<int> hole_offsets;
        for (const auto& hole : holes) {
            hole_offsets.push_back((int)(pointlist.size() / 2));
            for (const auto& v : hole) {
                pointlist.push_back((TRI_REAL)v.x);
                pointlist.push_back((TRI_REAL)v.y);
                pointmarkerlist.push_back(1);
            }
        }

        // Segment list: outer loop then each hole loop (0-based indices)
        std::vector<int> segmentlist;
        for (int i = 0; i < n_outer; ++i) {
            segmentlist.push_back(i);
            segmentlist.push_back((i + 1) % n_outer);
        }
        for (size_t h = 0; h < holes.size(); ++h) {
            int nh = (int)holes[h].size();
            int offset = hole_offsets[h];
            for (int i = 0; i < nh; ++i) {
                segmentlist.push_back(offset + i);
                segmentlist.push_back(offset + (i + 1) % nh);
            }
        }

        // Hole list: one point inside each hole (centroid)
        std::vector<TRI_REAL> holelist;
        for (const auto& hole : holes) {
            if (hole.empty()) continue;
            double hx = 0, hy = 0;
            for (const auto& v : hole) { hx += v.x; hy += v.y; }
            hx /= (double)hole.size();
            hy /= (double)hole.size();
            holelist.push_back((TRI_REAL)hx);
            holelist.push_back((TRI_REAL)hy);
        }

        // One region: centroid of outer polygon + max area constraint
        double cx = 0, cy = 0;
        for (const auto& v : boundary_poly) { cx += v.x; cy += v.y; }
        cx /= (double)n_outer;
        cy /= (double)n_outer;
        std::vector<TRI_REAL> regionlist;
        regionlist.push_back((TRI_REAL)cx);
        regionlist.push_back((TRI_REAL)cy);
        regionlist.push_back(0);
        regionlist.push_back((TRI_REAL)max_area);

        struct triangulateio in, out;
        memset(&in, 0, sizeof(in));
        memset(&out, 0, sizeof(out));

        in.numberofpoints = (int)(pointlist.size() / 2);
        in.pointlist = pointlist.data();
        in.pointmarkerlist = pointmarkerlist.data();
        in.numberofpointattributes = 0;

        in.numberofsegments = (int)(segmentlist.size() / 2);
        in.segmentlist = segmentlist.data();
        in.segmentmarkerlist = nullptr;

        in.numberofholes = (int)(holelist.size() / 2);
        in.holelist = holelist.empty() ? nullptr : holelist.data();

        in.numberofregions = 1;
        in.regionlist = regionlist.data();

        // p=PSLG, z=zero-based, q20=min angle 20°, a=area constraint from regionlist, Q=quiet
        triangulate(const_cast<char*>("pzq20aQ"), &in, &out, nullptr);

        // Copy output to Mesh
        vertices.reserve(out.numberofpoints);
        for (int i = 0; i < out.numberofpoints; ++i) {
            vertices.push_back({out.pointlist[2 * i], out.pointlist[2 * i + 1]});
        }
        for (int i = 0; i < out.numberofpoints; ++i) {
            if (out.pointmarkerlist[i] != 0) {
                boundary_nodes.push_back(i);
            }
        }
        elements.reserve(out.numberoftriangles);
        for (int i = 0; i < out.numberoftriangles; ++i) {
            elements.push_back({
                out.trianglelist[3 * i],
                out.trianglelist[3 * i + 1],
                out.trianglelist[3 * i + 2]
            });
        }

        // Free Triangle-allocated output (trifree takes int* in the API)
        trifree(reinterpret_cast<int*>(out.pointlist));
        trifree(out.pointmarkerlist);
        trifree(out.trianglelist);

        std::cout << "Triangle mesh: " << vertices.size() << " vertices, "
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
        // Enforce a strict minimum gap to prevent Triangle from failing on overlapping boundaries
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
