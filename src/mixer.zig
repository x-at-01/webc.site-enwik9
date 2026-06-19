const std = @import("std");

pub const ContextData = struct {
    weights: []f32,
    extra_weights: []f32,

    pub fn init(allocator: std.mem.Allocator, num_inputs: usize, extra_input_size: usize) !ContextData {
        const weights = try allocator.alloc(f32, num_inputs);
        @memset(weights, 0.0);
        const extra_weights = try allocator.alloc(f32, extra_input_size);
        @memset(extra_weights, 0.0);
        return .{
            .weights = weights,
            .extra_weights = extra_weights,
        };
    }

    pub fn deinit(self: *ContextData, allocator: std.mem.Allocator) void {
        allocator.free(self.weights);
        allocator.free(self.extra_weights);
    }
};

pub const Mixer = struct {
    allocator: std.mem.Allocator,
    learning_rate: f32,
    num_inputs: usize,
    extra_input_size: usize,
    steps: usize = 0,
    p: f32 = 0.5,
    context_map: std.AutoHashMap(u32, ContextData),
    context_base: ContextData,
    current_data: *ContextData = undefined,

    pub fn init(allocator: std.mem.Allocator, num_inputs: usize, extra_input_size: usize, learning_rate: f32) !Mixer {
        var self = Mixer{
            .allocator = allocator,
            .learning_rate = learning_rate,
            .num_inputs = num_inputs,
            .extra_input_size = extra_input_size,
            .context_map = std.AutoHashMap(u32, ContextData).init(allocator),
            .context_base = try ContextData.init(allocator, num_inputs, extra_input_size),
        };
        self.current_data = &self.context_base;
        return self;
    }

    pub fn deinit(self: *Mixer, allocator: std.mem.Allocator) void {
        _ = allocator;
        var it = self.context_map.valueIterator();
        while (it.next()) |data| {
            data.deinit(self.allocator);
        }
        self.context_map.deinit();
        self.context_base.deinit(self.allocator);
    }

    pub inline fn selectContext(self: *Mixer, context: u64) void {
        const key = @as(u32, @intCast(context & 0xffffffff));
        const limit = 10000;
        
        const gpre = self.context_map.getEntry(key);
        if (gpre) |entry| {
            self.current_data = entry.value_ptr;
        } else {
            if (self.context_map.count() >= limit) {
                self.current_data = &self.context_base;
            } else {
                const new_data = ContextData.init(self.allocator, self.num_inputs, self.extra_input_size) catch unreachable;
                self.context_map.put(key, new_data) catch unreachable;
                self.current_data = self.context_map.getPtr(key).?;
            }
        }
    }

    pub inline fn mix(self: *Mixer, inputs: []const f32, extra_inputs: []const f32) f32 {
        const data = self.current_data;
        var sum: f32 = 0.0;
        for (inputs, data.weights) |input, w| {
            sum += input * w;
        }
        var extra_sum: f32 = 0.0;
        for (extra_inputs, data.extra_weights) |ex_in, ex_w| {
            extra_sum += ex_in * ex_w;
        }
        self.p = sum + extra_sum;
        return self.p;
    }

    pub inline fn perceive(self: *Mixer, inputs: []const f32, extra_inputs: []const f32, bit: u1, steps: usize) void {
        var decay: f32 = 0.2;
        if (steps < 25000000) {
            decay = 0.3;
            if (steps < 5000000) {
                decay = 0.7;
                if (steps < 1000000) {
                    decay = 1.0;
                }
            }
        }

        // Sigmoid Logistic function: 1 / (1 + exp(-p))
        const pred_prob = 1.0 / (1.0 + @exp(-self.p));
        var update = self.learning_rate * (pred_prob - @as(f32, @floatFromInt(bit)));
        
        if (@abs(update) < 0.000000000005 and self.extra_input_size > 0) {
            return;
        }

        update = decay * update;
        const data = self.current_data;
        for (data.weights, inputs) |*w, input| {
            w.* -= update * input;
        }
        for (data.extra_weights, extra_inputs) |*ex_w, ex_in| {
            ex_w.* -= update * ex_in;
        }
    }
};
