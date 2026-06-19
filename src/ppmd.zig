const std = @import("std");
const ppmd_shkarin = @import("ppmd_shkarin.zig");

pub const PpmModel = struct {
    inner: ppmd_shkarin.Model,
    tree: [512]f32,
    vocab: [256]bool,

    pub fn init(allocator: std.mem.Allocator, vocab: [256]bool) !PpmModel {
        // C++ uses order 25, memory 14000. 
        // 2048MB RAM heap size ensures no cutoffs/resets occur for 1MB input.
        const memory_mb = 2048;
        const inner = try ppmd_shkarin.Model.init(allocator, 25, memory_mb, 1, 0);
        return PpmModel{
            .inner = inner,
            .tree = .{1.0} ** 512,
            .vocab = vocab,
        };
    }

    pub fn deinit(self: *PpmModel, allocator: std.mem.Allocator) void {
        _ = allocator;
        self.inner.deinit();
    }

    pub fn update(self: *PpmModel, history: []const u8, pos: usize, b: u8) void {
        _ = history;
        _ = pos;
        self.inner.ppmdUpdateByte(b);
    }

    pub fn predict(self: *PpmModel, history: []const u8, pos: usize) void {
        _ = history;
        _ = pos;
        self.inner.ppmdPrepareByte();

        // Populate tree leaves from inner.sqp
        var probs_sum: f32 = 0.0;
        for (0..256) |b| {
            if (self.vocab[b]) {
                const p = @as(f32, @floatFromInt(self.inner.sqp[b]));
                const p_clamped = if (p < 1.0) 1.0 else p;
                self.tree[256 + b] = p_clamped;
                probs_sum += p_clamped;
            } else {
                self.tree[256 + b] = 0.0;
            }
        }

        // Normalize
        for (0..256) |b| {
            self.tree[256 + b] /= probs_sum;
        }

        var idx: usize = 255;
        while (idx >= 1) : (idx -= 1) {
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1];
        }
    }

    pub fn predict_bit(self: *PpmModel, bc: u32) f32 {
        const sum = self.tree[bc];
        if (sum <= 0.000001) return 0.5;
        const p1 = self.tree[bc * 2 + 1] / sum;
        if (p1 < 0.0001) return 0.0001;
        if (p1 > 0.9999) return 0.9999;
        return p1;
    }
};
