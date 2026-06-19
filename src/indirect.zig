const std = @import("std");
const tables = @import("tables.zig");
const nonstationary_table = tables.nonstationary_table;

pub const IndirectModel = struct {
    predictions: [256]f32,
    map_offset: usize,
    divisor: f32,
    map_index: usize = 0,

    pub fn init(map_offset: usize, delta: f32) IndirectModel {
        var self = IndirectModel{
            .predictions = undefined,
            .map_offset = map_offset,
            .divisor = 1.0 / delta,
        };
        @memset(&self.predictions, 0.5);
        return self;
    }

    pub inline fn byteUpdate(self: *IndirectModel, byte_context: u64, map_len: usize) void {
        self.map_index = (257 *% byte_context +% self.map_offset) % (map_len - 257);
    }

    pub inline fn predict(self: *const IndirectModel, map: []const u8, bit_context: u32) f32 {
        const state = map[self.map_index + bit_context];
        return self.predictions[state];
    }

    pub inline fn perceive(self: *IndirectModel, map: []u8, bit_context: u32, bit: u1) void {
        const idx = self.map_index + bit_context;
        const state = map[idx];
        self.predictions[state] += (@as(f32, @floatFromInt(bit)) - self.predictions[state]) * self.divisor;
        map[idx] = nonstationary_table[state][bit];
    }
};
