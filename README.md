# Drums C++ 

Real-time interactive drum membrane simulator with GUI, built with SDL2, ImGui, and Eigen.

## Mathematical Formulation

### The Wave Equation

We solve the **2D damped wave equation** on an arbitrary membrane domain Ω with fixed (Dirichlet) boundary:

```
ü(x,t) + 2α(f)·u̇(x,t) = c² ∇²u(x,t) + f(x,t)
```

where:
- **u(x,t)** is the transverse membrane displacement
- **c = √(T/ρₛ)** is the wave speed (tension T, surface density ρₛ)
- **α(f) = α₀ + α₁·f** is frequency-dependent damping
- **f(x,t)** is the external strike force

### FEM Modal Decomposition

The domain is triangulated using Constrained Delaunay Triangulation (CDT), and the PDE is discretized into a generalized eigenvalue problem:

```
K·φₙ = λₙ·M·φₙ
```

where K is the stiffness matrix, M is the mass matrix, and (λₙ, φₙ) are eigenvalue/eigenvector pairs representing the natural frequencies and mode shapes of the membrane.

The displacement is expanded as a sum over modes: **u(x,t) = Σ qₙ(t)·φₙ(x)**, which decouples into independent modal ODEs.

### Complex Phasor Synthesis

Each modal ODE is converted to a first-order complex phasor state:

```
Żₙ = λₙ·Zₙ + Γₙ·f(t),  where λₙ = -αₙ + iωₙ
```

This is integrated exactly using an exponential integrator (Zero-Order Hold):

```
Zₙ[k+1] = Eₙ·Zₙ[k] + Gₙ·f[k]
```

where **Eₙ = exp(λₙ·Δt)** is the exact per-sample phasor rotation, and **Gₙ = (Eₙ-1)/λₙ · Γₙ** is the ZOH input gain. The output is **y[k] = Σ 2·pₙ·Re(Zₙ)**, where pₙ is the mode amplitude at the pickup location.

The strike forcing f(t) is a smooth temporal bump function **exp(-1/(1-ξ²))** with duration inversely proportional to strike velocity.

## System Dependencies

CMake will prefer your system-installed SDL2 if available, and only build it from source as a fallback. Installing SDL2 via your package manager is strongly recommended, as your distro's build will already include support for whichever audio backend you're running.

### Ubuntu 22.04+ / Debian 12+ (PipeWire)

PipeWire is the default audio system on modern Ubuntu and Debian. The `libsdl2-dev` package is compiled with PipeWire support out of the box.

```bash
sudo apt-get install -y cmake g++ libeigen3-dev libgl-dev libsdl2-dev
```

### Ubuntu 20.04 / Debian 11 (PulseAudio)

Older releases use PulseAudio. Install the PulseAudio dev library alongside SDL2 so audio initialises correctly.

```bash
sudo apt-get install -y cmake g++ libeigen3-dev libgl-dev libsdl2-dev libpulse-dev
```

### Minimal / Server installs (ALSA only)

If you're on a headless or stripped-down system without PulseAudio or PipeWire, SDL2 will fall back to ALSA directly.

```bash
sudo apt-get install -y cmake g++ libeigen3-dev libgl-dev libsdl2-dev libasound2-dev
```

> **Note:** If you see `Failed to init audio: dsp: no such audio device (Running in silent mode)` at runtime, it means SDL2 was built or installed without support for your audio backend. The fix is to install the appropriate dev package above (matching your running audio system) and do a clean rebuild: `rm -rf build/`.

All other dependencies (ImGui, ImPlot, Spectra, CDT) are fetched automatically by CMake.

## Build & Run

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./drums_cpp
```

## GUI Controls

| Control | Description |
|---|---|
| **Shape** | Dropdown to select geometry: Square, Ellipse, Regular Polygon, Custom, Isospectral |
| **Mesh Density** | Number of interior points for triangulation (5–1,000,000) |
| **Modes** | Number of eigenmodes to compute |
| **Rebuild Mesh** | Regenerate the mesh and solve the eigenvalue problem |
| **Tension** | Membrane tension T (N/m) — controls pitch |
| **Damping (α₀)** | Frequency-independent damping rate (1/s) |
| **Freq Damping (α₁)** | Frequency-proportional damping — higher frequencies decay faster |
| **Strike Velocity** | Mallet velocity — faster = sharper attack, brighter sound |
| **Strike Width** | Spatial extent of the mallet contact — smaller = more overtones |
| **Drum View** | Click to strike the membrane, drag to reposition meshes |
| **Frequency Spectrum** | Live dB-scale plot of modal amplitudes vs frequency |

## Architecture

- **`geometry.h`** — Mesh generation (polygons, ellipses, isospectral drums, custom shapes via CDT)
- **`fem.h`** — FEM stiffness (K) and mass (M) matrix assembly
- **`solver.h`** — Eigenvalue solver using Spectra (shift-invert mode)
- **`audio.h`** — Real-time audio engine with SIMD-optimized Struct-of-Arrays layout
- **`main.cpp`** — SDL2/ImGui application loop, rendering, and interaction

## License

MIT
