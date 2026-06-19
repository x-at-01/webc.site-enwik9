const std = @import("std");

pub const DirectModel = struct {
    predictions: []f32,
    counts: []u8,
    limit: u32,
    delta: f32,
    divisor: f32,
    size: usize,
    byte_context: u64 = 0,

    pub fn init(allocator: std.mem.Allocator, size: usize, limit: u32, delta: f32) !DirectModel {
        const num_elems = size * 256;
        const predictions = try allocator.alloc(f32, num_elems);
        @memset(predictions, 0.5);
        const counts = try allocator.alloc(u8, num_elems);
        @memset(counts, 0);

        return DirectModel{
            .predictions = predictions,
            .counts = counts,
            .limit = limit,
            .delta = delta,
            .divisor = 1.0 / (@as(f32, @floatFromInt(limit)) + delta),
            .size = size,
        };
    }

    pub fn deinit(self: *DirectModel, allocator: std.mem.Allocator) void {
        allocator.free(self.predictions);
        allocator.free(self.counts);
    }

    pub inline fn byteUpdate(self: *DirectModel, byte_context: u64) void {
        self.byte_context = byte_context % self.size;
    }

    pub inline fn predict(self: *const DirectModel, bit_context: u32) f32 {
        const idx = self.byte_context * 256 + bit_context;
        return self.predictions[idx];
    }

    pub inline fn perceive(self: *DirectModel, bit_context: u32, bit: u1) void {
        const idx = self.byte_context * 256 + bit_context;
        var div = self.divisor;
        if (self.counts[idx] < self.limit) {
            self.counts[idx] += 1;
            div = 1.0 / (@as(f32, @floatFromInt(self.counts[idx])) + self.delta);
        }
        self.predictions[idx] += (@as(f32, @floatFromInt(bit)) - self.predictions[idx]) * div;
    }
};
