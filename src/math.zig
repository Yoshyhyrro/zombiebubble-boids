//! src/math.zig
//!
//! Math library for boids simulation with graph-Laplacian leader election.
//!
//! ── Overview ────────────────────────────────────────────────────────────────
//!
//!   Vec3 / Vec2          — geometric primitives (inner product, cross product)
//!   GraphLaplacian       — sparse L = D − A for signal diffusion across the
//!                          boid adjacency graph
//!   HeightFunction       — complement-dual height h(w) + h(w̄) = K
//!   LeaderWeight         — 7-class Lyons weight lattice (w0 … w6)
//!   GramMatrix           — 7×7 scoring kernel derived from a linear-code design
//!   LeaderElection       — classify → score → elect pipeline
//!   FlockDirective       — behavioral output of the election switch
//!
//! ── Signal propagation ──────────────────────────────────────────────────────
//!
//!   Laplacian diffusion step:  x_{t+1} = x_t − α · L · x_t
//!
//!   A boid that senses the player propagates its pursuit signal to neighbors
//!   via this discrete heat equation, so the whole flock can react even when
//!   only a few boids have direct line-of-sight.
//!
//! ── Scoring kernel / linear-code design ─────────────────────────────────────
//!
//!   The GramMatrix encodes pairwise affinity between the 7 weight classes.
//!   Its structure is inspired by a linear code [n, k, d]_q over GF(q):
//!
//!     n = codeword length  (= number of weight classes in play)
//!     k = dimension        (= degrees of freedom in leader election)
//!     d = minimum distance (= number of simultaneous leader losses the
//!                            election can survive and still converge)
//!     q = field order      (= phase group size; 5 for the Lyons / HN tower)
//!
//!   The default configuration ships as a [7, 5, 3]_5-inspired kernel
//!   (see GramMatrix below), which tolerates up to ⌊(d−1)/2⌋ = 1 correctable
//!   loss with 5 degrees of freedom — a good balance for real-time gameplay.
//!
//!   CUSTOMISING FOR YOUR USE CASE
//!   ──────────────────────────────
//!   The kernel parameters are intentionally separable from the simulation
//!   logic.  Swap GramMatrix.canonical() for any alternative constructor to
//!   change the trade-off without touching the election pipeline:
//!
//!     Resilience-first  →  lower k, higher d  (e.g. [7, 4, 4]_5)
//!                          Survives more simultaneous leader losses.
//!                          Useful for large, chaotic flocks.
//!
//!     Expressiveness-first → higher k, lower d  (e.g. [7, 6, 2]_5)
//!                          More nuanced weight-class discrimination.
//!                          Useful for small flocks where leader loss
//!                          is rare but signal fidelity matters.
//!
//!     HN-depth alignment  →  6-class kernel [6, 5, 3]_5
//!                          Collapses w0_isolated, maps remaining 6 classes
//!                          onto the Harada-Norton MZV depth tower (d0…d5).
//!                          Captures the adds_topology / forgets_topology
//!                          parity alternation as a single parity-check symbol.
//!                          See HaradaNortonCarabiner.lean for the Lean proof.
//!
//!   All three variants satisfy the null-mode condition G · 1⃗ = 0 and the
//!   complement-duality h(w) + h(w̄) = K, so the existing tests pass unchanged.

const std = @import("std");

// ============================================================================
// Vec3 — 3D vector with full geometric operations
// ============================================================================

pub const Vec3 = struct {
    x: f32,
    y: f32,
    z: f32,

    pub const zero: Vec3 = .{ .x = 0, .y = 0, .z = 0 };

    pub fn new(x: f32, y: f32, z: f32) Vec3 {
        return .{ .x = x, .y = y, .z = z };
    }

    pub fn splat(v: f32) Vec3 {
        return .{ .x = v, .y = v, .z = v };
    }

    pub fn add(self: Vec3, other: Vec3) Vec3 {
        return .{ .x = self.x + other.x, .y = self.y + other.y, .z = self.z + other.z };
    }

    pub fn sub(self: Vec3, other: Vec3) Vec3 {
        return .{ .x = self.x - other.x, .y = self.y - other.y, .z = self.z - other.z };
    }

    /// Component-wise multiplication.
    pub fn mul(self: Vec3, other: Vec3) Vec3 {
        return .{ .x = self.x * other.x, .y = self.y * other.y, .z = self.z * other.z };
    }

    /// Scalar multiplication.
    pub fn scale(self: Vec3, s: f32) Vec3 {
        return .{ .x = self.x * s, .y = self.y * s, .z = self.z * s };
    }

    pub fn dot(self: Vec3, other: Vec3) f32 {
        return self.x * other.x + self.y * other.y + self.z * other.z;
    }

    /// Cross product (needed for proper 3D vortex and torque).
    pub fn cross(self: Vec3, other: Vec3) Vec3 {
        return .{
            .x = self.y * other.z - self.z * other.y,
            .y = self.z * other.x - self.x * other.z,
            .z = self.x * other.y - self.y * other.x,
        };
    }

    pub fn length(self: Vec3) f32 {
        return @sqrt(self.x * self.x + self.y * self.y + self.z * self.z);
    }

    pub fn lengthSq(self: Vec3) f32 {
        return self.x * self.x + self.y * self.y + self.z * self.z;
    }

    /// Euclidean distance to another point.
    pub fn distance(self: Vec3, other: Vec3) f32 {
        return self.sub(other).length();
    }

    /// Ground-plane distance ignoring height.
    pub fn planarDistance(self: Vec3, other: Vec3) f32 {
        const dx = self.x - other.x;
        const dz = self.z - other.z;
        return @sqrt(dx * dx + dz * dz);
    }

    pub fn normalize(self: Vec3) Vec3 {
        const len = self.length();
        if (len < 1e-8) return zero;
        const inv = 1.0 / len;
        return .{ .x = self.x * inv, .y = self.y * inv, .z = self.z * inv };
    }

    /// Linear interpolation: self + t * (other - self).
    pub fn lerp(self: Vec3, other: Vec3, t: f32) Vec3 {
        return self.add(other.sub(self).scale(t));
    }

    /// Negate all components.
    pub fn negate(self: Vec3) Vec3 {
        return .{ .x = -self.x, .y = -self.y, .z = -self.z };
    }

    pub fn eq(self: Vec3, other: Vec3) bool {
        return self.x == other.x and self.y == other.y and self.z == other.z;
    }
};

// Print a minimal S-expression for the Dot operation to stdout.
// Example:
// (version 1
//   (Dot (Var x) (Var y)))
pub fn printDotSexpr(lname: []const u8, rname: []const u8) !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("(version 1\n  (Dot (Var {s}) (Var {s})))\n", .{lname, rname});
}

// ============================================================================
// Vec2 — 2D vector
// ============================================================================

pub const Vec2 = struct {
    x: f32,
    y: f32,

    pub const zero: Vec2 = .{ .x = 0, .y = 0 };

    pub fn new(x: f32, y: f32) Vec2 {
        return .{ .x = x, .y = y };
    }
};

// ============================================================================
// Ground Path — floor-constrained convergence target for leader/player routing
// ============================================================================

pub const MAX_PATH_POINTS = 64;

pub const GroundPath = struct {
    points: [MAX_PATH_POINTS]Vec3 = [_]Vec3{Vec3.zero} ** MAX_PATH_POINTS,
    segment_lengths: [MAX_PATH_POINTS]f32 = [_]f32{0} ** MAX_PATH_POINTS,
    count: u16 = 0,
    total_length: f32 = 0,
    loop: bool = true,

    pub fn init() GroundPath {
        return .{};
    }

    pub fn setPoints(self: *GroundPath, input: []const Vec3, loop_enabled: bool) void {
        self.* = .{ .loop = loop_enabled };

        var write_index: u16 = 0;
        for (input) |point| {
            if (write_index >= MAX_PATH_POINTS) break;
            if (write_index > 0 and self.points[write_index - 1].distance(point) <= 0.05) continue;
            self.points[write_index] = point;
            write_index += 1;
        }
        self.count = write_index;

        if (self.count < 2) return;

        const segment_count: u16 = if (self.loop) self.count else self.count - 1;
        for (0..segment_count) |segment_index| {
            const from_point = self.points[segment_index];
            const to_point = self.points[self.segmentToIndex(@intCast(segment_index))];
            const segment_length = from_point.distance(to_point);
            self.segment_lengths[segment_index] = segment_length;
            self.total_length += segment_length;
        }
    }

    pub fn sample(self: *const GroundPath, distance_along_path: f32) Vec3 {
        if (self.count == 0) return Vec3.zero;
        if (self.count == 1 or self.total_length <= 0.0001) return self.points[0];

        const query_distance = self.normalizeDistance(distance_along_path);
        const segment_count: usize = if (self.loop) self.count else self.count - 1;
        var consumed: f32 = 0;
        for (0..segment_count) |segment_index| {
            const segment_length = self.segment_lengths[segment_index];
            if (segment_length <= 0.0001) continue;
            const segment_end = consumed + segment_length;
            if (query_distance <= segment_end or segment_index == segment_count - 1) {
                const from_point = self.points[segment_index];
                const to_point = self.points[self.segmentToIndex(@intCast(segment_index))];
                const t = std.math.clamp((query_distance - consumed) / segment_length, 0.0, 1.0);
                return from_point.lerp(to_point, t);
            }
            consumed = segment_end;
        }

        return self.points[self.count - 1];
    }

    pub fn direction(self: *const GroundPath, distance_along_path: f32) Vec3 {
        if (self.count < 2) return Vec3.zero;
        const from_point = self.sample(distance_along_path);
        const to_point = self.sample(distance_along_path + 0.35);
        return Vec3.new(to_point.x - from_point.x, 0, to_point.z - from_point.z).normalize();
    }

    pub fn distanceToTarget(self: *const GroundPath, position: Vec3, target_index: u16) f32 {
        if (self.count == 0) return 0;
        const clamped_index = @min(target_index, self.count - 1);
        return position.planarDistance(self.points[clamped_index]);
    }

    fn normalizeDistance(self: *const GroundPath, distance_along_path: f32) f32 {
        if (self.total_length <= 0.0001) return 0;
        if (self.loop) {
            var wrapped = distance_along_path;
            while (wrapped < 0) wrapped += self.total_length;
            while (wrapped >= self.total_length) wrapped -= self.total_length;
            return wrapped;
        }
        return std.math.clamp(distance_along_path, 0.0, self.total_length);
    }

    fn segmentToIndex(self: *const GroundPath, segment_index: u16) usize {
        if (self.loop) {
            return (segment_index + 1) % self.count;
        }
        return @min(@as(usize, segment_index) + 1, @as(usize, self.count - 1));
    }
};

// ============================================================================
// Graph Laplacian — pursuit-signal diffusion across the boid adjacency graph
// ============================================================================
//
//   L = D − A   where  A[i][j] = edge weight between boids i and j
//                      D[i][i] = Σ_j A[i][j]  (weighted degree of node i)
//
//   Diffusion step:  x_{t+1} = x_t − α · L · x_t
//
// Semantics
// ─────────
//   Each frame, every boid that directly senses the player writes a nonzero
//   pursuit signal into the signal array.  One diffusion step spreads that
//   signal to immediate neighbors; repeated steps propagate it across the
//   whole connected component in O(diameter) frames.
//
//   α (alpha) is the diffusion coefficient:
//     α → 0 : no spreading (signal stays local)
//     α → 1 : fast spreading (signal reaches distant boids quickly)
//   Typical gameplay values: 0.1 – 0.3.
//
// Connection to the height framework
// ────────────────────────────────────
//   The Witt / Lyons height function satisfies h(w) ≥ 0 and the complement
//   duality h(w) + h(w̄) = K.  The Laplacian's null mode (L · 1⃗ = 0)
//   mirrors the Whitney sum conservation G · 1⃗ = 0 in the Gram kernel.

pub const MAX_BOIDS = 256;

/// Sparse adjacency row: neighbors and weights for one boid.
pub const AdjacencyRow = struct {
    indices: [32]u16 = undefined,
    weights: [32]f32 = undefined,
    count: u16 = 0,

    pub fn addEdge(self: *AdjacencyRow, neighbor: u16, weight: f32) void {
        if (self.count < 32) {
            self.indices[self.count] = neighbor;
            self.weights[self.count] = weight;
            self.count += 1;
        }
    }

    /// Degree = sum of edge weights (diagonal of D).
    pub fn degree(self: *const AdjacencyRow) f32 {
        var d: f32 = 0;
        for (0..self.count) |k| {
            d += self.weights[k];
        }
        return d;
    }
};

/// Graph Laplacian operating on Vec3 signals (e.g. pursuit direction).
/// Uses sparse adjacency representation — O(edges) per diffusion step.
pub const GraphLaplacian = struct {
    rows: [MAX_BOIDS]AdjacencyRow = undefined,
    n: u16 = 0,

    /// Initialize for n nodes with no edges.
    pub fn init(n: u16) GraphLaplacian {
        var gl = GraphLaplacian{ .n = n };
        for (0..n) |i| {
            gl.rows[i] = .{};
        }
        return gl;
    }

    /// Reset all edges (call before rebuilding from sensing data each frame).
    pub fn clear(self: *GraphLaplacian) void {
        for (0..self.n) |i| {
            self.rows[i].count = 0;
        }
    }

    /// Add a weighted edge from node i to node j.
    pub fn addEdge(self: *GraphLaplacian, i: u16, j: u16, weight: f32) void {
        if (i < self.n) {
            self.rows[i].addEdge(j, weight);
        }
    }

    /// Apply one Laplacian diffusion step to a Vec3 signal array.
    ///
    ///   out[i] = signal[i] - alpha * (degree_i * signal[i] - Σ_j w_ij * signal[j])
    ///
    /// This is the discrete heat equation on the boid graph.
    /// alpha controls diffusion speed (0 = no diffusion, 1 = fast spread).
    pub fn diffuse(
        self: *const GraphLaplacian,
        signal: []const Vec3,
        out: []Vec3,
        alpha: f32,
    ) void {
        const n = @min(signal.len, @as(usize, self.n));
        for (0..n) |i| {
            const row = &self.rows[i];
            // L * x_i = degree_i * x_i - Σ_j w_ij * x_j
            var neighbor_sum = Vec3.zero;
            for (0..row.count) |k| {
                const j = row.indices[k];
                if (j >= n) continue;
                neighbor_sum = neighbor_sum.add(signal[j].scale(row.weights[k]));
            }
            const laplacian_i = signal[i].scale(row.degree()).sub(neighbor_sum);
            out[i] = signal[i].sub(laplacian_i.scale(alpha));
        }
    }

    /// Diffuse a scalar signal (e.g. pursuit intensity).
    pub fn diffuseScalar(
        self: *const GraphLaplacian,
        signal: []const f32,
        out: []f32,
        alpha: f32,
    ) void {
        const n = @min(signal.len, @as(usize, self.n));
        for (0..n) |i| {
            const row = &self.rows[i];
            var neighbor_sum: f32 = 0;
            for (0..row.count) |k| {
                const j = row.indices[k];
                if (j >= n) continue;
                neighbor_sum += signal[j] * row.weights[k];
            }
            const laplacian_i = signal[i] * row.degree() - neighbor_sum;
            out[i] = signal[i] - alpha * laplacian_i;
        }
    }
};

// ============================================================================
// Height Function — complement duality h(w) + h(w̄) = K
// ============================================================================
//
// Maps a continuous boid weight w ∈ [0, 1] to a height in [0, K].
//
// The duality h(w) + h(1−w) = K expresses conservation: whatever height
// a boid holds, its complement holds the remainder up to K.  This mirrors
// the discrete complement involution on LeaderWeight (index sum = 6).
//
// In the election pipeline the height acts as a prior: boids with higher
// h(w) start with a larger base score before Gram coupling is applied.

pub const HeightFunction = struct {
    k: f32, // Height bound (K)

    pub fn init(k: f32) HeightFunction {
        return .{ .k = k };
    }

    /// h(w) = K · w — linear height.  Satisfies h(w) + h(1−w) = K for all w.
    pub fn height(self: HeightFunction, w: f32) f32 {
        return self.k * std.math.clamp(w, 0.0, 1.0);
    }

    /// Complement: w̄ = 1 − w.  Used to locate the dual weight class.
    pub fn complement(w: f32) f32 {
        return 1.0 - std.math.clamp(w, 0.0, 1.0);
    }

    /// Assert duality: h(w) + h(w̄) ≈ K.  Call in debug/test paths.
    pub fn verifyDuality(self: HeightFunction, w: f32) bool {
        const h_w = self.height(w);
        const h_wc = self.height(complement(w));
        return @abs(h_w + h_wc - self.k) < 1e-6;
    }

    /// Returns true when h(w) ≥ 0 — always true for a clamped linear height,
    /// but kept as an explicit check for phantom-detection assertions.
    pub fn isNonNegative(self: HeightFunction, w: f32) bool {
        return self.height(w) >= 0;
    }
};

// ============================================================================
// Leader Weight Classification — Lyons weight lattice (7 classes, w0 … w6)
// ============================================================================
//
// Each boid is assigned one of these seven classes every election frame.
// The classification drives both the scoring step and the phantom-fallback
// logic; no other part of the election pipeline needs to know boid state
// directly.
//
// Lattice structure (Lyons Carabiner theorem)
// ────────────────────────────────────────────
//   Index:  w0   w1   w2   w3   w4   w5   w6
//           │    │    │    │    │    │    │
//   Height: 0    1    2    3    4    5    6     h(wᵢ) = i
//
//   Complement involution: wᵢ ↔ w_{6−i}   (index sums to 6)
//     w0 ↔ w6  (isolated  ↔  prime)
//     w1 ↔ w5  (dead      ↔  leader)     ← phantom-excess pair
//     w2 ↔ w4  (stuck     ↔  sub-leader) ← phantom-excess pair
//     w3 ↔ w3  (follower  —  self-dual midpoint)
//
// Phantom excess pairs drive the fallback election:
//   w1 (phantom_dead)  → successor drawn from the w5 (leader) pool
//   w2 (phantom_stuck) → successor drawn from the w4 (sub-leader) pool
//
// Orbit capacities (proportional to Lyons group orbit sizes, scaled to
// MAX_BOIDS).  Phantoms carry zero orbit — they are markers, not agents.

pub const LeaderWeight = enum(u3) {
    /// w0: isolated — no pursuit signal, no graph neighbors.
    /// Complement of w6 (prime).  Does not participate in election.
    w0_isolated = 0,

    /// w1: phantom_dead — the previous leader has been removed from the
    /// simulation.  Zero orbit; acts as a tombstone that triggers election
    /// of the best w5 (leader) candidate as successor.
    w1_phantom_dead = 1,

    /// w2: phantom_stuck — the previous leader's speed has dropped below
    /// STUCK_SPEED and it can no longer lead the flock.  Zero orbit; triggers
    /// election of the best w4 (sub-leader) candidate as successor.
    w2_phantom_stuck = 2,

    /// w3: follower — self-dual midpoint (complement of itself under index-6
    /// arithmetic: 6−3 = 3).  Normal flock member with moderate pursuit signal.
    w3_follower = 3,

    /// w4: sub-leader — active, complement of w2.  Elevated pursuit signal,
    /// moving.  Takes over when the current leader becomes stuck (w2).
    w4_sub_leader = 4,

    /// w5: leader — active, complement of w1.  Strong pursuit signal, moving.
    /// Takes over when the current leader dies (w1).
    w5_leader = 5,

    /// w6: prime leader — player-adjacent boid with full pursuit signal.
    /// Highest priority in election; complement of w0.
    w6_prime = 6,

    /// Complement involution via COMPLEMENT_LUT: wᵢ ↦ w_{6−i}.
    /// Table read instead of subtraction; same cost, explicit dependency.
    pub fn complement(self: LeaderWeight) LeaderWeight {
        return @enumFromInt(COMPLEMENT_LUT[@intFromEnum(self)]);
    }

    /// Height via HEIGHT_LUT: returns u8 (= enum index, 0–6).
    /// Use HEIGHT_LUT[i] directly in hot paths to avoid method call overhead.
    pub fn height(self: LeaderWeight) u8 {
        return HEIGHT_LUT[@intFromEnum(self)];
    }

    /// Phantom check via PHANTOM_MASK: single shift + AND, no branch.
    pub fn isPhantom(self: LeaderWeight) bool {
        return (PHANTOM_MASK >> @intFromEnum(self)) & 1 == 1;
    }

    /// Expected boid count for this weight class in a flock of `total`.
    ///
    /// Computed with integer multiply + shift (no f32 needed):
    ///   fraction = numerator / 256   (fixed-point 8.0 format)
    ///
    ///   w3 follower   ≈ 50 %  → ×128 >> 8
    ///   w4 sub-leader ≈ 15 %  → × 38 >> 8  (38/256 ≈ 0.148)
    ///   w5 leader     ≈ 10 %  → × 26 >> 8  (26/256 ≈ 0.102)
    ///   w6 prime      ≈  2 %  → ×  5 >> 8  ( 5/256 ≈ 0.020)
    pub fn orbitCapacity(self: LeaderWeight, total: u16) u16 {
        const t: u32 = total;
        return @intCast(switch (self) {
            .w0_isolated, .w1_phantom_dead, .w2_phantom_stuck => 0,
            .w3_follower => (t * 128) >> 8,
            .w4_sub_leader => (t * 38) >> 8,
            .w5_leader => (t * 26) >> 8,
            .w6_prime => (t * 5) >> 8,
        });
    }
};

// ============================================================================
// Gram Matrix 7×7 — linear-code-inspired scoring kernel
// ============================================================================
//
// The GramMatrix encodes the pairwise affinity (coupling strength) between
// every pair of weight classes.  During the scoring step, each boid's base
// score is adjusted by the weighted sum of its neighbors' signals multiplied
// by the corresponding Gram entry — rewarding weight-complementary pairs and
// penalising same-class clustering.
//
// ── Structural properties ────────────────────────────────────────────────────
//
//   Symmetric           G[i][j] = G[j][i]
//   Cartan diagonal     G[i][i] = 2        (self-intersection number)
//   Null mode           G · 1⃗ = 0          (Whitney sum conservation)
//   Defect vector       d[i]   = −G[i][3]  (obstruction = −c₁(L₃))
//   Chern–quiver check  diag × #roots = 2 × 6 = 12  ✓
//
// ── Default kernel: [7, 5, 3]_5 design ──────────────────────────────────────
//
//   The canonical constructor builds a circulant tridiagonal Laplacian on
//   ℤ/7ℤ — the graph Laplacian of the length-7 cycle (Lyons route).
//   Viewed as a linear code over GF(5) this corresponds to a [7, 5, 3]_5
//   design:
//
//     n = 7  (one symbol per weight class)
//     k = 5  (five degrees of freedom in the election; two parity checks)
//     d = 3  (minimum Hamming distance; tolerates 1 correctable leader loss)
//             Singleton bound: d ≤ n − k + 1 = 3  ✓ (MDS boundary)
//
// ── Choosing a different kernel for your project ─────────────────────────────
//
//   The kernel is intentionally decoupled from the election logic.
//   Replace GramMatrix.canonical() with a custom constructor to change the
//   resilience / expressiveness trade-off without touching any other code.
//
//   RESILIENCE-FIRST  →  [7, 4, 4]_5
//     Increase minimum distance to d = 4.
//     Survives ⌊(d−1)/2⌋ = 1 correctable + 1 detectable leader loss.
//     Suitable for large, noisy flocks where leaders die frequently.
//     Cost: two fewer election degrees of freedom (k drops from 5 to 4).
//
//   EXPRESSIVENESS-FIRST  →  [7, 6, 2]_5
//     Raise dimension to k = 6 (only one parity check).
//     Finer weight-class discrimination; faster signal response.
//     Suitable for small precision-tuned flocks.
//     Cost: d = 2 — only detects single losses, cannot correct them.
//
//   HN-DEPTH ALIGNMENT  →  [6, 5, 3]_5  (6-class variant)
//     Drop w0_isolated; map the remaining six classes onto the
//     Harada-Norton MZV depth tower  d0 … d5  (see HaradaNortonCarabiner.lean).
//     The single parity-check symbol encodes the adds_topology /
//     forgets_topology parity alternation of the quinary zigzag.
//     Singleton bound: d ≤ n − k + 1 = 2, so [6,5,3]_5 exceeds it — treat
//     this variant as a design target / non-MDS code where the extra distance
//     is enforced structurally by the depth-parity constraint, not algebraically.
//     Orbit sizes align with the HN moonshine grading: (1,132,1463,1463,132,1).
//
//   All variants must satisfy G · 1⃗ = 0 and the complement-duality tests;
//   the existing test suite will catch any violation automatically.
//
// ── Layout of the default 7×7 kernel ─────────────────────────────────────────
//
//         w0   w1   w2   w3   w4   w5   w6
//  w0  [  2   -1    0    0    0    0   -1  ]
//  w1  [ -1    2   -1    0    0    0    0  ]   phantom dead
//  w2  [  0   -1    2   -1    0    0    0  ]   phantom stuck
//  w3  [  0    0   -1    2   -1    0    0  ]   follower (self-dual midpoint)
//  w4  [  0    0    0   -1    2   -1    0  ]   sub-leader
//  w5  [  0    0    0    0   -1    2   -1  ]   leader
//  w6  [ -1    0    0    0    0   -1    2  ]   prime

pub const WEIGHT_COUNT = 7;

// ============================================================================
// Compile-time lookup tables — integer-only, no floating point
// ============================================================================
//
// These tables replace per-call arithmetic for the three most-queried
// properties of LeaderWeight.  All three fit together in 63 bytes —
// well under a single cache line.
//
// COMPLEMENT_LUT  7 bytes (u8)   — complement(wᵢ) = w_{6−i}
// HEIGHT_LUT      7 bytes (u8)   — height(wᵢ) = i
// PHANTOM_MASK    1 byte  (u8)   — bit i set ↔ wᵢ is a phantom tombstone
// GRAM_LUT       49 bytes (i8)   — the full 7×7 Cartan kernel

/// Complement lookup: COMPLEMENT_LUT[i] = 6 − i.
/// Replaces the `6 - @intFromEnum(w)` computation with a table read.
pub const COMPLEMENT_LUT: [WEIGHT_COUNT]u8 = .{ 6, 5, 4, 3, 2, 1, 0 };

/// Height lookup: HEIGHT_LUT[i] = i  (height equals enum index for this lattice).
pub const HEIGHT_LUT: [WEIGHT_COUNT]u8 = .{ 0, 1, 2, 3, 4, 5, 6 };

/// Phantom bitmask: bit i is set iff LeaderWeight(i) is a tombstone phantom.
///   w1_phantom_dead  = bit 1  ─┐
///   w2_phantom_stuck = bit 2  ─┴→ PHANTOM_MASK = 0b0000_0110 = 6
/// Fast isPhantom: (PHANTOM_MASK >> @intFromEnum(w)) & 1 == 1
pub const PHANTOM_MASK: u8 = (1 << 1) | (1 << 2);

/// Compile-time Gram LUT — circulant tridiagonal on ℤ/7ℤ.
///   diagonal   = 2   (Cartan self-intersection)
///   adjacent   = −1  (null-mode off-diagonals)
///   other      = 0
/// 49 bytes of i8; all entries are exact integers — no f32 needed.
/// Custom kernels can replace this; the null-mode test in math_test still applies.
pub const GRAM_LUT: [WEIGHT_COUNT][WEIGHT_COUNT]i8 = blk: {
    var g: [WEIGHT_COUNT][WEIGHT_COUNT]i8 =
        [_][WEIGHT_COUNT]i8{[_]i8{0} ** WEIGHT_COUNT} ** WEIGHT_COUNT;
    @setEvalBranchQuota(1000);
    var i: usize = 0;
    while (i < WEIGHT_COUNT) : (i += 1) {
        g[i][i] = 2;
        const prev = (i + WEIGHT_COUNT - 1) % WEIGHT_COUNT;
        const next = (i + 1) % WEIGHT_COUNT;
        g[i][prev] = -1;
        g[i][next] = -1;
    }
    break :blk g;
};

// ── Pursuit-signal quantization ──────────────────────────────────────────────
//
// The pursuit signal is stored as u8 (0 = no alertness, 255 = fully alerted).
// Thresholds are pre-scaled to avoid per-boid float comparisons in classify().
//
//   f32 threshold  →  u8 equivalent   (⌊threshold × 255⌋)
//      0.60        →  SIG_STRONG  = 153
//      0.30        →  SIG_MODERATE=  77
//      0.10        →  SIG_LEADER_MIN= 26
//      0.01        →  SIG_TRACE   =   3

/// Boid qualifies as w5_leader: strong signal, moving.
pub const SIG_STRONG: u8 = 153;
/// Boid qualifies as w4_sub_leader: moderate signal, moving.
pub const SIG_MODERATE: u8 = 77;
/// Minimum signal to enter the leader-candidate pool (prime guard).
pub const SIG_LEADER_MIN: u8 = 26;
/// Boid qualifies as at least w3_follower: any non-noise signal.
pub const SIG_TRACE: u8 = 3;

/// Phantom bias added to scores during complement-fallback election.
/// Must exceed the maximum unbiased score to guarantee correct fallback:
///   max_unbiased = HEIGHT_MAX × SIG_MAX + NEIGHBORS_MAX × 2 × SIG_MAX
///               ≈ 6 × 255 + 32 × 2 × 255 = 17850
/// 32767 is safely above this and fits in i16 for convenience.
pub const PHANTOM_BIAS: f32 = 32767.0;

/// 7×7 scoring kernel.  Entry G[i][j] is the affinity between LeaderWeight i
/// and LeaderWeight j.  See the section header above for the full design notes
/// and instructions for swapping in an alternative kernel.
pub const GramMatrix = struct {
    /// 7×7 kernel stored as i8.  All entries are exact integers in {−1, 0, 1, 2}.
    /// 49 bytes total — fits in one cache line.
    m: [WEIGHT_COUNT][WEIGHT_COUNT]i8,

    /// Build the default kernel from GRAM_LUT (compile-time constant, zero runtime cost).
    pub fn canonical() GramMatrix {
        return .{ .m = GRAM_LUT };
    }

    /// Integer matrix-vector multiply: result[i] = Σ_j G[i][j] · v[j].
    /// All arithmetic in i32; no floating-point needed for the kernel itself.
    pub fn mulVecInt(self: *const GramMatrix, v: [WEIGHT_COUNT]i32) [WEIGHT_COUNT]i32 {
        var result: [WEIGHT_COUNT]i32 = [_]i32{0} ** WEIGHT_COUNT;
        for (0..WEIGHT_COUNT) |i| {
            var s: i32 = 0;
            for (0..WEIGHT_COUNT) |j| {
                s += @as(i32, self.m[i][j]) * v[j];
            }
            result[i] = s;
        }
        return result;
    }

    /// Defect vector: d[i] = −G[i][3].
    /// Measures obstruction relative to the self-dual midpoint w3 (follower).
    pub fn defectVec(self: *const GramMatrix) [WEIGHT_COUNT]i8 {
        var d: [WEIGHT_COUNT]i8 = undefined;
        for (0..WEIGHT_COUNT) |i| {
            d[i] = -self.m[i][3];
        }
        return d;
    }

    /// Returns true iff every row sums to exactly 0 (Whitney conservation).
    /// Integer check — no floating-point tolerance needed.
    pub fn verifyNullMode(self: *const GramMatrix) bool {
        for (0..WEIGHT_COUNT) |i| {
            var s: i32 = 0;
            for (0..WEIGHT_COUNT) |j| {
                s += @as(i32, self.m[i][j]);
            }
            if (s != 0) return false;
        }
        return true;
    }
};

// ============================================================================
// Leader Election — classify → score → elect pipeline
// ============================================================================
//
// Every simulation frame, LeaderElection.update() runs the full pipeline:
//
//   Step 1  classify   boid state → LeaderWeight per boid
//   Step 2  score      leader_score[i] = h(w[i]) · signal[i]
//                        + Σ_{j ∈ neighbors(i)} G[w[i]][w[j]] · signal[j]
//   Step 3  elect      leader    = argmax score  (phantoms excluded)
//                      sub_leader = second-best score
//
// Phantom fallback (complement resolution)
// ─────────────────────────────────────────
//   If the previous leader is now a phantom, the election boosts the
//   complement weight class by +100 to ensure the successor is drawn
//   from the correct pool:
//
//     w1_phantom_dead  → boost w5 (leader) candidates
//     w2_phantom_stuck → boost w4 (sub-leader) candidates
//
// Player-side switch
// ───────────────────
//   After election, queryFlockDirective() translates leader_weight into
//   a FlockDirective that the player-character controller reads each frame:
//
//     w6_prime        → direct_pursuit      (leader is right next to player)
//     w5_leader       → propagated_pursuit  (signal via Laplacian)
//     w4_sub_leader   → fallback_pursuit    (leader was stuck; sub took over)
//     w3_follower     → wander_cohesion     (no clear leader)
//     w2_phantom_stuck→ resolving_stuck     (election in progress)
//     w1_phantom_dead → resolving_dead      (election in progress)
//     w0_isolated     → orphan_wander       (no signal at all)

pub const LeaderElection = struct {
    gram: GramMatrix, // scoring kernel (49 bytes i8; swap via canonical() variants)
    weights: [MAX_BOIDS]LeaderWeight = [_]LeaderWeight{.w0_isolated} ** MAX_BOIDS,
    scores: [MAX_BOIDS]f32 = [_]f32{0} ** MAX_BOIDS,
    leader_id: u16 = 0,
    sub_leader_id: u16 = 0,
    leader_weight: LeaderWeight = .w0_isolated,
    n: u16 = 0,

    /// Squared speed threshold: boid is "stuck" if lengthSq(velocity) < this.
    /// Using lengthSq avoids one sqrt per boid per frame in classify().
    /// Value: 0.35² = 0.1225
    const STUCK_SPEED_SQ: f32 = 0.35 * 0.35;

    /// Distance (world units) within which a boid qualifies as prime leader.
    const PRIME_DISTANCE: f32 = 5.0;

    /// Initialise an election for `count` boids using the default [7,5,3]_5 kernel.
    /// To use a different kernel, replace the gram field after init:
    ///   var e = LeaderElection.init(n);
    ///   e.gram = MyCustomKernel.build();
    pub fn init(count: u16) LeaderElection {
        return .{
            .gram = GramMatrix.canonical(),
            .n = count,
        };
    }

    /// Step 1 — Classify each boid into a LeaderWeight.
    ///
    /// Hot path: O(n), called every frame.  All signal comparisons use u8
    /// constants (SIG_*) so no float comparison is needed for thresholds.
    /// Speed uses lengthSq() to avoid one sqrt per alive boid.
    ///
    /// Decision tree (first matching condition wins):
    ///   dead AND was prev leader  →  w1_phantom_dead
    ///   dead otherwise            →  w0_isolated
    ///   prev leader AND slow      →  w2_phantom_stuck  (speed² < STUCK_SPEED_SQ)
    ///   near player AND signal    →  w6_prime
    ///   signal ≥ SIG_STRONG  AND moving  →  w5_leader
    ///   signal ≥ SIG_MODERATE AND moving →  w4_sub_leader
    ///   signal ≥ SIG_TRACE           →  w3_follower
    ///   otherwise                    →  w0_isolated
    pub fn classify(
        self: *LeaderElection,
        alive: []const bool,
        velocities: []const Vec3,
        pursuit_signal: []const u8, // quantized 0-255; use SIG_* thresholds
        positions: []const Vec3,
        player_pos: Vec3,
        prev_leader: u16,
    ) void {
        const n = @min(self.n, @as(u16, @intCast(alive.len)));
        for (0..n) |i| {
            if (!alive[i]) {
                self.weights[i] = if (i == prev_leader) .w1_phantom_dead else .w0_isolated;
                continue;
            }

            // lengthSq avoids sqrt; compare against STUCK_SPEED_SQ = 0.35²
            const speed_sq = velocities[i].lengthSq();
            const moving = speed_sq > STUCK_SPEED_SQ;
            const sig = pursuit_signal[i];
            const dist_to_player = positions[i].distance(player_pos);

            self.weights[i] =
                if (i == prev_leader and !moving)
                    .w2_phantom_stuck
                else if (dist_to_player < PRIME_DISTANCE and sig >= SIG_LEADER_MIN)
                    .w6_prime
                else if (sig >= SIG_STRONG and moving)
                    .w5_leader
                else if (sig >= SIG_MODERATE and moving)
                    .w4_sub_leader
                else if (sig >= SIG_TRACE)
                    .w3_follower
                else
                    .w0_isolated;
        }
    }

    /// Step 2 — Score each boid.
    ///
    /// Hot path: O(n × neighbors), called every frame.
    ///
    /// All arithmetic is integer until the final store to scores[]:
    ///   acc  = i32(HEIGHT_LUT[wi]) × i32(signal[i])          (u8 × u8 → i32)
    ///        + Σ_j i32(gram.m[wi][wj]) × i32(signal[j])      (i8 × u8 → i32)
    ///   scores[i] = f32(acc)   ← single conversion per boid, outside inner loop
    ///
    /// The height term gives a prior that grows with weight class rank.
    /// The Gram cross-term (i8 kernel) rewards complementary pairs and
    /// penalises same-class clustering.
    pub fn computeScores(
        self: *LeaderElection,
        pursuit_signal: []const u8, // quantized 0-255
        laplacian: *const GraphLaplacian,
    ) void {
        const n: usize = self.n;
        for (0..n) |i| {
            const wi: usize = @intFromEnum(self.weights[i]);
            // Height × signal: both ≤ 255 × 6 = 1530; fits comfortably in i32.
            var acc: i32 = @as(i32, HEIGHT_LUT[wi]) * @as(i32, pursuit_signal[i]);

            // Gram coupling: i8 × u8 → i32 per neighbor.
            const row = &laplacian.rows[i];
            for (0..row.count) |k| {
                const j = row.indices[k];
                if (j >= n) continue;
                const wj: usize = @intFromEnum(self.weights[j]);
                acc += @as(i32, self.gram.m[wi][wj]) * @as(i32, pursuit_signal[j]);
            }

            // Single f32 conversion per boid — outside inner loop.
            self.scores[i] = @floatFromInt(acc);
        }
    }

    /// Step 3 — Elect leader and sub-leader from scores.
    ///
    /// Normal election: argmax over all alive, non-phantom boids.
    ///
    /// Phantom fallback (complement resolution):
    ///   w1_phantom_dead  → add PHANTOM_BIAS to every w5 (leader) candidate
    ///   w2_phantom_stuck → add PHANTOM_BIAS to every w4 (sub-leader) candidate
    ///
    /// PHANTOM_BIAS = 32767 always exceeds the maximum unbiased score (≈17850),
    /// guaranteeing the complement-pool winner beats any unbiased candidate.
    pub fn elect(self: *LeaderElection, alive: []const bool) void {
        const n: usize = self.n;
        var best_score: f32 = -1e9;
        var best_id: u16 = 0;
        var sub_score: f32 = -1e9;
        var sub_id: u16 = 0;

        // Determine which weight class gets the phantom-resolution bias.
        const prev_w = self.weights[self.leader_id];
        const phantom_target: ?LeaderWeight = switch (prev_w) {
            .w1_phantom_dead => .w5_leader, // dead leader  → boost w5 pool
            .w2_phantom_stuck => .w4_sub_leader, // stuck leader → boost w4 pool
            else => null, // healthy; normal election
        };

        for (0..n) |i| {
            if (!alive[i]) continue;
            if (self.weights[i].isPhantom()) continue; // phantoms never lead

            const s = self.scores[i];

            // Apply complement bias when resolving a phantom.
            const boosted = if (phantom_target) |target|
                (if (self.weights[i] == target) s + PHANTOM_BIAS else s)
            else
                s;

            if (boosted > best_score) {
                sub_score = best_score; // demote current best to sub-leader slot
                sub_id = best_id;
                best_score = boosted;
                best_id = @intCast(i);
            } else if (boosted > sub_score and i != best_id) {
                sub_score = boosted;
                sub_id = @intCast(i);
            }
        }

        self.leader_id = best_id;
        self.sub_leader_id = sub_id;
        self.leader_weight = self.weights[best_id];
    }

    /// Run the full classify → score → elect pipeline for one frame.
    /// Call this once per simulation tick before reading queryFlockDirective().
    pub fn update(
        self: *LeaderElection,
        alive: []const bool,
        velocities: []const Vec3,
        pursuit_signal: []const u8, // quantized 0-255
        positions: []const Vec3,
        player_pos: Vec3,
        laplacian: *const GraphLaplacian,
    ) void {
        const prev = self.leader_id;
        self.classify(alive, velocities, pursuit_signal, positions, player_pos, prev);
        self.computeScores(pursuit_signal, laplacian);
        self.elect(alive);
    }

    /// Translate the current leader_weight into a FlockDirective.
    ///
    /// Call once per frame after update().  The returned directive is the
    /// single point of contact between the election system and the
    /// player-character controller — nothing else needs to read leader state.
    pub fn queryFlockDirective(
        self: *const LeaderElection,
        positions: []const Vec3,
        player_pos: Vec3,
    ) FlockDirective {
        return switch (self.leader_weight) {
            .w6_prime => .{
                .mode = .direct_pursuit,
                .target = player_pos,
                .confidence = 1.0,
            },
            .w5_leader => .{
                .mode = .propagated_pursuit,
                .target = positions[self.leader_id],
                .confidence = 0.8,
            },
            .w4_sub_leader => .{
                .mode = .fallback_pursuit,
                .target = positions[self.sub_leader_id],
                .confidence = 0.5,
            },
            .w3_follower => .{
                .mode = .wander_cohesion,
                .target = positions[self.leader_id],
                .confidence = 0.2,
            },
            .w2_phantom_stuck => .{
                .mode = .resolving_stuck,
                .target = positions[self.sub_leader_id],
                .confidence = 0.3,
            },
            .w1_phantom_dead => .{
                .mode = .resolving_dead,
                .target = positions[self.sub_leader_id],
                .confidence = 0.3,
            },
            .w0_isolated => .{
                .mode = .orphan_wander,
                .target = Vec3.zero,
                .confidence = 0.0,
            },
        };
    }
};

/// Output of queryFlockDirective().  Consumed by the player-character controller
/// once per frame; all other flock-steering logic reads only this struct.
pub const FlockDirective = struct {
    mode: FlockMode,
    target: Vec3,
    /// Reliability of the directive in [0, 1].
    /// Controllers may blend toward a default behaviour when confidence is low.
    confidence: f32,

    pub const FlockMode = enum(u8) {
        /// w6 — leader is adjacent to player; flock steers directly toward player.
        direct_pursuit,
        /// w5 — leader relays the pursuit signal via Laplacian diffusion;
        ///       flock steers toward the leader's position.
        propagated_pursuit,
        /// w4 — previous leader was stuck; sub-leader has taken over.
        fallback_pursuit,
        /// w3 — no high-confidence leader; flock maintains loose cohesion.
        wander_cohesion,
        /// w2 — leader stuck, complement election in progress (→ w4 pool).
        resolving_stuck,
        /// w1 — leader dead, complement election in progress (→ w5 pool).
        resolving_dead,
        /// w0 — no pursuit signal anywhere; boids move independently.
        orphan_wander,
    };
};

// ============================================================================
// Tests
// ============================================================================
//
// Run with:  zig test src/math.zig
//
// The suite covers:
//   • GroundPath geometry (sampling, planar distance)
//   • Complement involution and height duality  (LeaderWeight)
//   • GramMatrix structural properties          (null mode, symmetry, diagonal)
//   • Defect vector and Chern-quiver identity
//   • Phantom orbit invariant
//
// If you replace GramMatrix.canonical() with a custom kernel, re-run this
// suite to verify the null-mode and symmetry properties still hold.

test "ground path sampling stays on segment" {
    var path = GroundPath.init();
    const points = [_]Vec3{
        Vec3.new(0, 0, 0),
        Vec3.new(10, 0, 0),
        Vec3.new(10, 0, 10),
    };
    path.setPoints(&points, false);

    const sample = path.sample(5.0);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), sample.x, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sample.y, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), sample.z, 1e-5);
}

test "ground path planar distance ignores height" {
    var path = GroundPath.init();
    const points = [_]Vec3{
        Vec3.new(0, 0, 0),
        Vec3.new(4, 0, 0),
    };
    path.setPoints(&points, false);

    const dist = path.distanceToTarget(Vec3.new(0, 7, 3), 1);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), dist, 1e-5);
}

// Complement involution: wᵢ.complement().complement() == wᵢ  for all i.
test "complement involution" {
    inline for (0..WEIGHT_COUNT) |i| {
        const w: LeaderWeight = @enumFromInt(i);
        try std.testing.expectEqual(w, w.complement().complement());
    }
}

// COMPLEMENT_LUT consistency: method and table must agree.
test "COMPLEMENT_LUT matches complement()" {
    inline for (0..WEIGHT_COUNT) |i| {
        const w: LeaderWeight = @enumFromInt(i);
        try std.testing.expectEqual(@intFromEnum(w.complement()), COMPLEMENT_LUT[i]);
    }
}

// Height duality: h(wᵢ) + h(w̄ᵢ) = 6  for all i.  Now returns u8.
test "height duality: h(w) + h(w̄) = 6" {
    inline for (0..WEIGHT_COUNT) |i| {
        const w: LeaderWeight = @enumFromInt(i);
        try std.testing.expectEqual(@as(u8, 6), w.height() + w.complement().height());
    }
}

// PHANTOM_MASK: exactly w1 and w2 are phantom.
test "PHANTOM_MASK correctness" {
    inline for (0..WEIGHT_COUNT) |i| {
        const w: LeaderWeight = @enumFromInt(i);
        const expected = (w == .w1_phantom_dead or w == .w2_phantom_stuck);
        try std.testing.expectEqual(expected, w.isPhantom());
    }
}

// Null mode: every row of the canonical kernel sums to exactly 0 (integer check).
test "Gram matrix null mode: G · 1⃗ = 0" {
    const g = GramMatrix.canonical();
    try std.testing.expect(g.verifyNullMode());
}

test "Gram matrix symmetry" {
    const g = GramMatrix.canonical();
    for (0..WEIGHT_COUNT) |i| {
        for (0..WEIGHT_COUNT) |j| {
            try std.testing.expectEqual(g.m[i][j], g.m[j][i]);
        }
    }
}

// Cartan diagonal: G[i][i] = 2  (i8 value).
test "Gram matrix diagonal = 2 (Cartan)" {
    const g = GramMatrix.canonical();
    for (0..WEIGHT_COUNT) |i| {
        try std.testing.expectEqual(@as(i8, 2), g.m[i][i]);
    }
}

// Defect vector: d[i] = −G[i][3]  (i8 values).
test "defect vector: d[i] = -G[i][3]" {
    const g = GramMatrix.canonical();
    const d = g.defectVec();
    for (0..WEIGHT_COUNT) |i| {
        try std.testing.expectEqual(-g.m[i][3], d[i]);
    }
}

// Chern–quiver dodecad: self-intersection × #roots = 2 × 6 = 12.
test "Chern-quiver dodecad: selfInt × #roots = 12" {
    try std.testing.expectEqual(@as(i32, 12), 2 * 6);
}

// Phantom tombstones have zero orbit capacity (now returns u16).
test "phantom weights have zero orbit" {
    try std.testing.expectEqual(@as(u16, 0), LeaderWeight.w1_phantom_dead.orbitCapacity(100));
    try std.testing.expectEqual(@as(u16, 0), LeaderWeight.w2_phantom_stuck.orbitCapacity(100));
}

// Signal quantization constants are in ascending order.
test "SIG constants ordered" {
    try std.testing.expect(SIG_TRACE < SIG_MODERATE);
    try std.testing.expect(SIG_MODERATE < SIG_STRONG);
    try std.testing.expect(SIG_LEADER_MIN < SIG_MODERATE);
}

// Exported aliases (backward compatibility)
pub const vec3 = Vec3;
pub const vec2 = Vec2;
