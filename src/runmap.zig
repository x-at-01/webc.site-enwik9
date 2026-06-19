const tables = @import("tables.zig");
const run_map_table = tables.run_map_table;

pub const RunMapModel = struct {
    predictions: [256]f32,
    map_offset: usize,
    divisor: f32,
    map_index: usize = 0,

    pub fn init(map_offset: usize, delta: f32) RunMapModel {
        var self = RunMapModel{
            .predictions = undefined,
            .map_offset = map_offset,
            .divisor = 1.0 / delta,
        };
        for (0..256) |i| {
            if (i < 128) {
                self.predictions[i] = @as(f32, @floatFromInt(128 - i)) / 256.0;
            } else {
                self.predictions[i] = @as(f32, @floatFromInt(i)) / 256.0;
            }
        }
        return self;
    }

    pub inline fn byteUpdate(self: *RunMapModel, byte_context: u64, map_len: usize) void {
        self.map_index = (257 *% byte_context +% self.map_offset) % (map_len - 257);
    }

    pub inline fn predict(self: *const RunMapModel, map: []const u8, bit_context: u32) f32 {
        const state = map[self.map_index + bit_context];
        return self.predictions[state];
    }

    pub inline fn perceive(self: *RunMapModel, map: []u8, bit_context: u32, bit: u1) void {
        const idx = self.map_index + bit_context;
        const state = map[idx];
        self.predictions[state] += (@as(f32, @floatFromInt(bit)) - self.predictions[state]) * self.divisor;
        map[idx] = run_map_table[@as(usize, state) * 2 + bit];
    }
};
