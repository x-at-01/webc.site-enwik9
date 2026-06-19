const std = @import("std");

pub const MatchModel = struct {
    map: []u32,
    predictions: [256]f32,
    counts: [256]u32,
    limit: u32,
    delta: f32,
    divisor: f32,
    cur_match: usize = 0,
    cur_byte: u8 = 0,
    bit_pos: u8 = 128,
    match_length: u8 = 0,

    pub fn init(allocator: std.mem.Allocator, map_size: usize, limit: u32, delta: f32) !MatchModel {
        const map = try allocator.alloc(u32, map_size);
        @memset(map, 0);
        var self = MatchModel{
            .map = map,
            .predictions = undefined,
            .counts = undefined,
            .limit = limit,
            .delta = delta,
            .divisor = 1.0 / (@as(f32, @floatFromInt(limit)) + delta),
        };
        for (0..256) |i| {
            self.predictions[i] = 0.5 + @as(f32, @floatFromInt(i)) / 512.0;
        }
        @memset(&self.counts, 0);
        return self;
    }

    pub fn deinit(self: *MatchModel, allocator: std.mem.Allocator) void {
        allocator.free(self.map);
    }

    pub inline fn predict(self: *const MatchModel) f32 {
        const expected_bit = (self.cur_byte & self.bit_pos) != 0;
        if (expected_bit) {
            return self.predictions[self.match_length];
        } else {
            return 1.0 - self.predictions[self.match_length];
        }
    }

    pub inline fn perceive(self: *MatchModel, byte_context: u64, bit_context: u32, bit: u1, history_pos: usize) void {
        const expected_bit = (self.cur_byte & self.bit_pos) != 0;
        const match: f32 = if (bit == @intFromBool(expected_bit)) 1.0 else 0.0;
        self.bit_pos >>= 1;

        const ml = self.match_length;
        var div = self.divisor;
        if (self.counts[ml] < self.limit) {
            self.counts[ml] += 1;
            div = 1.0 / (@as(f32, @floatFromInt(self.counts[ml])) + self.delta);
        }
        self.predictions[ml] += (match - self.predictions[ml]) * div;

        if (match == 1.0) {
            if (self.match_length < 255) {
                self.match_length += 1;
            }
        } else {
            self.match_length = 0;
        }

        if (bit_context >= 128) {
            self.map[byte_context % self.map.len] = @intCast(history_pos);
        }
    }

    pub inline fn byteUpdate(self: *MatchModel, byte_context: u64, history: []const u8) void {
        if (self.match_length < 8) {
            self.cur_match = self.map[byte_context % self.map.len];
        } else {
            self.cur_match += 1;
        }
        if (history.len > 0) {
            self.cur_match %= history.len;
            self.cur_byte = history[self.cur_match];
        } else {
            self.cur_byte = 0;
        }
        self.bit_pos = 128;
    }
};
