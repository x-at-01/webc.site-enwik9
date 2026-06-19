const std = @import("std");

pub const Mixer = struct {
    keys: []u64,
    weight_table: []f32,
    learning_rate: f32,
    num_inputs: usize,
    table_size: usize,
    current_idx: usize = 0,

    pub fn init(allocator: std.mem.Allocator, table_size: usize, num_inputs: usize, learning_rate: f32) !Mixer {
        std.debug.assert((table_size & (table_size - 1)) == 0);
        const keys = try allocator.alloc(u64, table_size);
        @memset(keys, 0xffffffffffffffff);
        const weight_table = try allocator.alloc(f32, table_size * num_inputs);
        @memset(weight_table, 0.0);
        return Mixer{
            .keys = keys,
            .weight_table = weight_table,
            .num_inputs = num_inputs,
            .table_size = table_size,
            .learning_rate = learning_rate,
        };
    }

    pub fn deinit(self: *Mixer, allocator: std.mem.Allocator) void {
        allocator.free(self.keys);
        allocator.free(self.weight_table);
    }

    pub inline fn selectContext(self: *Mixer, context: u64) void {
        const mask = self.table_size - 1;
        var idx = @as(usize, @intCast(context & mask));
        var i: usize = 0;
        while (i < 128) : (i += 1) {
            const key = self.keys[idx];
            if (key == context) {
                self.current_idx = idx;
                return;
            }
            if (key == 0xffffffffffffffff) {
                self.keys[idx] = context;
                self.current_idx = idx;
                return;
            }
            idx = (idx + 1) & mask;
        }
        self.current_idx = @as(usize, @intCast(context & mask));
    }

    pub inline fn mix(self: *const Mixer, inputs: []const f32) f32 {
        const start = self.current_idx * self.num_inputs;
        const weights = self.weight_table[start .. start + self.num_inputs];
        var sum: f32 = 0.0;
        for (inputs, weights) |input, weight| {
            sum += input * weight;
        }
        return sum;
    }

    pub inline fn perceive(self: *Mixer, inputs: []const f32, mixed_logit: f32, bit: u1, decay: f32) void {
        const start = self.current_idx * self.num_inputs;
        const weights = self.weight_table[start .. start + self.num_inputs];
        const pred_prob = 1.0 / (1.0 + @exp(-mixed_logit));
        const error_val = @as(f32, @floatFromInt(bit)) - pred_prob;
        const update = self.learning_rate * error_val * decay;
        for (weights, inputs) |*weight, input| {
            weight.* += update * input;
        }
    }
};
