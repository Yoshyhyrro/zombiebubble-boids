//! src/ffi_python.zig
//!
//! Minimal C ABI for Python ctypes.  Free tier: Vec3 math + 3-force boids
//! (cohesion / alignment / separation).
//!
//! NOT exposed here: leader election, force fields, Godot GDExtension bindings.
//!
//! Build with:  zig build pip-lib
//! Output:      zig-out/lib/libzombiebubble.so   (Linux)
//!              zig-out/lib/zombiebubble.dll      (Windows)
//!              zig-out/lib/libzombiebubble.dylib (macOS)

const std = @import("std");
const math = @import("math");
const Vec3 = math.vec3;

const gpa = std.heap.page_allocator;

// ── Internal swarm state ──────────────────────────────────────────────────────

const Swarm = struct {
    count: u32,
    pos: []Vec3, //    positions   [count]
    vel: []Vec3, //    velocities  [count]
    scratch: []Vec3, // force accumulator (reused each step)
    buf: []Vec3, //    backing allocation (pos ++ vel ++ scratch)
    // flocking weights
    cohesion_w: f32,
    alignment_w: f32,
    separation_w: f32,
    max_speed: f32,
    cohesion_radius: f32,
    separation_radius: f32,
};

// ── Lifecycle ─────────────────────────────────────────────────────────────────

export fn zb_swarm_create(count: u32) ?*Swarm {
    if (count == 0 or count > 4096) return null;
    const s = gpa.create(Swarm) catch return null;
    const buf = gpa.alloc(Vec3, count * 3) catch {
        gpa.destroy(s);
        return null;
    };
    s.* = .{
        .count = count,
        .pos = buf[0..count],
        .vel = buf[count .. count * 2],
        .scratch = buf[count * 2 .. count * 3],
        .buf = buf,
        .cohesion_w = 1.0,
        .alignment_w = 0.6,
        .separation_w = 1.5,
        .max_speed = 3.0,
        .cohesion_radius = 12.0,
        .separation_radius = 2.0,
    };
    // Scatter initial positions using a golden-angle (sunflower) spiral so
    // that boids start well-separated but still within cohesion_radius of
    // each other.  No PRNG needed; the pattern is deterministic and even.
    //
    // golden_angle ≈ 137.5° — adjacent points never share a spoke.
    // r_scale = sqrt(i+1) * spacing  →  uniform area density.
    const golden_angle: f32 = 2.3999631; // 2π / φ²  radians
    const spacing: f32 = 2.2; // metres between "rings"
    for (0..count) |i| {
        const fi: f32 = @floatFromInt(i);
        const r = @sqrt(fi + 1.0) * spacing;
        const theta = fi * golden_angle;
        s.pos[i] = .{ .x = @cos(theta) * r, .y = 0.0, .z = @sin(theta) * r };
        s.vel[i] = Vec3.zero;
        s.scratch[i] = Vec3.zero;
    }
    return s;
}

export fn zb_swarm_destroy(s: ?*Swarm) void {
    const sw = s orelse return;
    gpa.free(sw.buf);
    gpa.destroy(sw);
}

// ── Config setters ────────────────────────────────────────────────────────────

export fn zb_swarm_set_cohesion_weight(s: ?*Swarm, w: f32) void {
    if (s) |sw| sw.cohesion_w = w;
}
export fn zb_swarm_set_alignment_weight(s: ?*Swarm, w: f32) void {
    if (s) |sw| sw.alignment_w = w;
}
export fn zb_swarm_set_separation_weight(s: ?*Swarm, w: f32) void {
    if (s) |sw| sw.separation_w = w;
}
export fn zb_swarm_set_max_speed(s: ?*Swarm, v: f32) void {
    if (s) |sw| sw.max_speed = v;
}
export fn zb_swarm_set_cohesion_radius(s: ?*Swarm, r: f32) void {
    if (s) |sw| sw.cohesion_radius = r;
}
export fn zb_swarm_set_separation_radius(s: ?*Swarm, r: f32) void {
    if (s) |sw| sw.separation_radius = r;
}

// ── Simulation step ───────────────────────────────────────────────────────────

export fn zb_swarm_step(s: ?*Swarm, dt: f32) void {
    const sw = s orelse return;
    const n = sw.count;

    // Clear force accumulator.
    for (sw.scratch) |*f| f.* = Vec3.zero;

    // Compute flocking forces (O(n²) naive; sufficient for free-tier demos).
    for (0..n) |i| {
        var avg_pos = Vec3.zero;
        var avg_vel = Vec3.zero;
        var sep = Vec3.zero;
        var neighbors: u32 = 0;

        for (0..n) |j| {
            if (i == j) continue;
            const diff = sw.pos[i].sub(sw.pos[j]);
            const dist = diff.length();
            if (dist < sw.cohesion_radius and dist > 0.001) {
                avg_pos = avg_pos.add(sw.pos[j]);
                avg_vel = avg_vel.add(sw.vel[j]);
                neighbors += 1;
                if (dist < sw.separation_radius) {
                    // Push away; force grows as distance shrinks.
                    sep = sep.add(diff.normalize().scale(sw.separation_radius / (dist + 0.001)));
                }
            }
        }

        if (neighbors > 0) {
            const nc: f32 = @floatFromInt(neighbors);
            const cohesion = avg_pos.scale(1.0 / nc).sub(sw.pos[i]).normalize().scale(sw.cohesion_w);
            const alignment = avg_vel.scale(1.0 / nc).normalize().scale(sw.alignment_w);
            const separation = sep.normalize().scale(sw.separation_w);
            sw.scratch[i] = cohesion.add(alignment).add(separation);
        }
    }

    // Integrate: v += force*dt, clamp speed, p += v*dt.
    for (0..n) |i| {
        sw.vel[i] = sw.vel[i].add(sw.scratch[i].scale(dt));
        const speed = sw.vel[i].length();
        if (speed > sw.max_speed) sw.vel[i] = sw.vel[i].scale(sw.max_speed / speed);
        sw.pos[i] = sw.pos[i].add(sw.vel[i].scale(dt));
    }
}

// ── Read-back ─────────────────────────────────────────────────────────────────
//
// Flat f32 layout: [x0, y0, z0,  x1, y1, z1, ...]
// `n` is the total number of floats in `out` (must be >= count * 3).

export fn zb_swarm_get_positions(s: ?*const Swarm, out: [*]f32, n: u32) void {
    const sw = s orelse return;
    const count = @min(n / 3, sw.count);
    for (0..count) |i| {
        out[i * 3 + 0] = sw.pos[i].x;
        out[i * 3 + 1] = sw.pos[i].y;
        out[i * 3 + 2] = sw.pos[i].z;
    }
}

export fn zb_swarm_get_velocities(s: ?*const Swarm, out: [*]f32, n: u32) void {
    const sw = s orelse return;
    const count = @min(n / 3, sw.count);
    for (0..count) |i| {
        out[i * 3 + 0] = sw.vel[i].x;
        out[i * 3 + 1] = sw.vel[i].y;
        out[i * 3 + 2] = sw.vel[i].z;
    }
}

export fn zb_swarm_count(s: ?*const Swarm) u32 {
    return if (s) |sw| sw.count else 0;
}

// ── Vec3 math exports ─────────────────────────────────────────────────────────

export fn zb_vec3_dot(ax: f32, ay: f32, az: f32, bx: f32, by: f32, bz: f32) f32 {
    return (Vec3{ .x = ax, .y = ay, .z = az }).dot(.{ .x = bx, .y = by, .z = bz });
}

export fn zb_vec3_length(x: f32, y: f32, z: f32) f32 {
    return (Vec3{ .x = x, .y = y, .z = z }).length();
}

export fn zb_vec3_add(
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
    ox: *f32,
    oy: *f32,
    oz: *f32,
) void {
    const r = (Vec3{ .x = ax, .y = ay, .z = az }).add(.{ .x = bx, .y = by, .z = bz });
    ox.* = r.x;
    oy.* = r.y;
    oz.* = r.z;
}

export fn zb_vec3_sub(
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
    ox: *f32,
    oy: *f32,
    oz: *f32,
) void {
    const r = (Vec3{ .x = ax, .y = ay, .z = az }).sub(.{ .x = bx, .y = by, .z = bz });
    ox.* = r.x;
    oy.* = r.y;
    oz.* = r.z;
}

export fn zb_vec3_scale(
    vx: f32,
    vy: f32,
    vz: f32,
    scalar: f32,
    ox: *f32,
    oy: *f32,
    oz: *f32,
) void {
    const r = (Vec3{ .x = vx, .y = vy, .z = vz }).scale(scalar);
    ox.* = r.x;
    oy.* = r.y;
    oz.* = r.z;
}

export fn zb_vec3_cross(
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
    ox: *f32,
    oy: *f32,
    oz: *f32,
) void {
    const r = (Vec3{ .x = ax, .y = ay, .z = az }).cross(.{ .x = bx, .y = by, .z = bz });
    ox.* = r.x;
    oy.* = r.y;
    oz.* = r.z;
}

export fn zb_vec3_lerp(
    ax: f32,
    ay: f32,
    az: f32,
    bx: f32,
    by: f32,
    bz: f32,
    t: f32,
    ox: *f32,
    oy: *f32,
    oz: *f32,
) void {
    const r = (Vec3{ .x = ax, .y = ay, .z = az }).lerp(.{ .x = bx, .y = by, .z = bz }, t);
    ox.* = r.x;
    oy.* = r.y;
    oz.* = r.z;
}

export fn zb_vec3_normalize(
    vx: f32,
    vy: f32,
    vz: f32,
    ox: *f32,
    oy: *f32,
    oz: *f32,
) void {
    const r = (Vec3{ .x = vx, .y = vy, .z = vz }).normalize();
    ox.* = r.x;
    oy.* = r.y;
    oz.* = r.z;
}
