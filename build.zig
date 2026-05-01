const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const math_mod = b.createModule(.{
        .root_source_file = b.path("src/math.zig"),
    });

    // Build shared library for Python ctypes
    const pip_lib = b.addLibrary(.{
        .name = "zombiebubble",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/ffi_python.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "math", .module = math_mod },
            },
        }),
        .linkage = .dynamic,
    });
    const pip_install = b.addInstallArtifact(pip_lib, .{});
    const pip_step = b.step("pip-lib", "Build shared library for Python ctypes");
    pip_step.dependOn(&pip_install.step);
}
