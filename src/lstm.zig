const std = @import("std");

pub inline fn logistic(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

pub const NeuronLayer = struct {
    weights: []f32, // num_cells * input_size
    state: []f32, // horizon * num_cells
    update: []f32, // num_cells * input_size
    m: []f32, // num_cells * input_size
    v: []f32, // num_cells * input_size
    transpose: []f32, // num_cells * transpose_cols
    norm: []f32, // horizon * num_cells
    error_grad: []f32, // num_cells
    gamma: []f32, // num_cells
    gamma_u: []f32, // num_cells
    gamma_m: []f32, // num_cells
    gamma_v: []f32, // num_cells
    beta: []f32, // num_cells
    beta_u: []f32, // num_cells
    beta_m: []f32, // num_cells
    beta_v: []f32, // num_cells
    ivar: []f32, // horizon

    pub fn init(allocator: std.mem.Allocator, input_size: usize, num_cells: usize, horizon: usize, offset: usize) !NeuronLayer {
        const transpose_cols = input_size - offset;

        const weights = try allocator.alloc(f32, num_cells * input_size);
        const state = try allocator.alloc(f32, horizon * num_cells);
        const update = try allocator.alloc(f32, num_cells * input_size);
        const m = try allocator.alloc(f32, num_cells * input_size);
        const v = try allocator.alloc(f32, num_cells * input_size);
        const transpose = try allocator.alloc(f32, num_cells * transpose_cols);
        const norm = try allocator.alloc(f32, horizon * num_cells);
        const error_grad = try allocator.alloc(f32, num_cells);
        const gamma = try allocator.alloc(f32, num_cells);
        const gamma_u = try allocator.alloc(f32, num_cells);
        const gamma_m = try allocator.alloc(f32, num_cells);
        const gamma_v = try allocator.alloc(f32, num_cells);
        const beta = try allocator.alloc(f32, num_cells);
        const beta_u = try allocator.alloc(f32, num_cells);
        const beta_m = try allocator.alloc(f32, num_cells);
        const beta_v = try allocator.alloc(f32, num_cells);
        const ivar = try allocator.alloc(f32, horizon);

        @memset(weights, 0.0);
        @memset(state, 0.0);
        @memset(update, 0.0);
        @memset(m, 0.0);
        @memset(v, 0.0);
        @memset(transpose, 0.0);
        @memset(norm, 0.0);
        @memset(error_grad, 0.0);
        @memset(gamma, 1.0);
        @memset(gamma_u, 0.0);
        @memset(gamma_m, 0.0);
        @memset(gamma_v, 0.0);
        @memset(beta, 0.0);
        @memset(beta_u, 0.0);
        @memset(beta_m, 0.0);
        @memset(beta_v, 0.0);
        @memset(ivar, 0.0);

        return NeuronLayer{
            .weights = weights,
            .state = state,
            .update = update,
            .m = m,
            .v = v,
            .transpose = transpose,
            .norm = norm,
            .error_grad = error_grad,
            .gamma = gamma,
            .gamma_u = gamma_u,
            .gamma_m = gamma_m,
            .gamma_v = gamma_v,
            .beta = beta,
            .beta_u = beta_u,
            .beta_m = beta_m,
            .beta_v = beta_v,
            .ivar = ivar,
        };
    }

    pub fn deinit(self: *NeuronLayer, allocator: std.mem.Allocator) void {
        allocator.free(self.weights);
        allocator.free(self.state);
        allocator.free(self.update);
        allocator.free(self.m);
        allocator.free(self.v);
        allocator.free(self.transpose);
        allocator.free(self.norm);
        allocator.free(self.error_grad);
        allocator.free(self.gamma);
        allocator.free(self.gamma_u);
        allocator.free(self.gamma_m);
        allocator.free(self.gamma_v);
        allocator.free(self.beta);
        allocator.free(self.beta_u);
        allocator.free(self.beta_m);
        allocator.free(self.beta_v);
        allocator.free(self.ivar);
    }
};

pub const LstmLayer = struct {
    state: []f32,
    state_error: []f32,
    stored_error: []f32,
    tanh_state: []f32,
    input_gate_state: []f32,
    last_state: []f32,

    gradient_clip: f32,
    learning_rate: f32,
    num_cells: usize,
    epoch: usize = 0,
    horizon: usize,
    input_size: usize,
    output_size: usize,
    update_steps: usize = 0,

    forget_gate: NeuronLayer,
    input_node: NeuronLayer,
    output_gate: NeuronLayer,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, auxiliary_input_size: usize, output_size: usize, num_cells: usize, horizon: usize, gradient_clip: f32, learning_rate: f32, random: std.Random) !LstmLayer {
        const state = try allocator.alloc(f32, num_cells);
        @memset(state, 0.0);
        const state_error = try allocator.alloc(f32, num_cells);
        @memset(state_error, 0.0);
        const stored_error = try allocator.alloc(f32, num_cells);
        @memset(stored_error, 0.0);
        const tanh_state = try allocator.alloc(f32, horizon * num_cells);
        @memset(tanh_state, 0.0);
        const input_gate_state = try allocator.alloc(f32, horizon * num_cells);
        @memset(input_gate_state, 0.0);
        const last_state = try allocator.alloc(f32, horizon * num_cells);
        @memset(last_state, 0.0);

        const offset = output_size + auxiliary_input_size;
        var forget_gate = try NeuronLayer.init(allocator, input_size, num_cells, horizon, offset);
        var input_node = try NeuronLayer.init(allocator, input_size, num_cells, horizon, offset);
        var output_gate = try NeuronLayer.init(allocator, input_size, num_cells, horizon, offset);

        const val = @sqrt(6.0 / @as(f32, @floatFromInt(auxiliary_input_size + output_size)));
        const low = -val;
        const range = 2.0 * val;

        for (0..num_cells) |i| {
            for (0..input_size) |j| {
                forget_gate.weights[i * input_size + j] = low + random.float(f32) * range;
                input_node.weights[i * input_size + j] = low + random.float(f32) * range;
                output_gate.weights[i * input_size + j] = low + random.float(f32) * range;
            }
            forget_gate.weights[i * input_size + input_size - 1] = 1.0;
        }

        return LstmLayer{
            .state = state,
            .state_error = state_error,
            .stored_error = stored_error,
            .tanh_state = tanh_state,
            .input_gate_state = input_gate_state,
            .last_state = last_state,
            .gradient_clip = gradient_clip,
            .learning_rate = learning_rate,
            .num_cells = num_cells,
            .horizon = horizon,
            .input_size = auxiliary_input_size,
            .output_size = output_size,
            .forget_gate = forget_gate,
            .input_node = input_node,
            .output_gate = output_gate,
        };
    }

    pub fn deinit(self: *LstmLayer, allocator: std.mem.Allocator) void {
        allocator.free(self.state);
        allocator.free(self.state_error);
        allocator.free(self.stored_error);
        allocator.free(self.tanh_state);
        allocator.free(self.input_gate_state);
        allocator.free(self.last_state);
        self.forget_gate.deinit(allocator);
        self.input_node.deinit(allocator);
        self.output_gate.deinit(allocator);
    }

    fn forwardPassLayer(self: *LstmLayer, neurons: *NeuronLayer, input: []const f32, input_symbol: usize) void {
        const offset = self.output_size + self.input_size;
        const row_size = offset + self.num_cells + 1;
        for (0..self.num_cells) |i| {
            var f = neurons.weights[i * row_size + input_symbol];
            for (input, 0..) |inp_val, j| {
                f += inp_val * neurons.weights[i * row_size + self.output_size + j];
            }
            neurons.norm[self.epoch * self.num_cells + i] = f;
        }

        var sum_sq: f32 = 0.0;
        const norm_epoch = neurons.norm[self.epoch * self.num_cells .. (self.epoch + 1) * self.num_cells];
        for (norm_epoch) |val| {
            sum_sq += val * val;
        }
        const ivar_val = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(self.num_cells)) + 1e-5);
        neurons.ivar[self.epoch] = ivar_val;

        for (0..self.num_cells) |i| {
            const idx = self.epoch * self.num_cells + i;
            neurons.norm[idx] *= ivar_val;
            neurons.state[idx] = neurons.norm[idx] * neurons.gamma[i] + neurons.beta[i];
        }
    }

    pub fn forwardPass(self: *LstmLayer, input: []const f32, input_symbol: usize, hidden: []f32, hidden_start: usize) void {
        const epoch_offset = self.epoch * self.num_cells;
        @memcpy(self.last_state[epoch_offset .. epoch_offset + self.num_cells], self.state);

        self.forwardPassLayer(&self.forget_gate, input, input_symbol);
        self.forwardPassLayer(&self.input_node, input, input_symbol);
        self.forwardPassLayer(&self.output_gate, input, input_symbol);

        for (0..self.num_cells) |i| {
            const idx = epoch_offset + i;
            self.forget_gate.state[idx] = logistic(self.forget_gate.state[idx]);
            self.input_node.state[idx] = @as(f32, @floatCast(std.math.tanh(self.input_node.state[idx])));
            self.output_gate.state[idx] = logistic(self.output_gate.state[idx]);

            self.input_gate_state[idx] = 1.0 - self.forget_gate.state[idx];
            self.state[i] = self.state[i] * self.forget_gate.state[idx] + self.input_node.state[idx] * self.input_gate_state[idx];
            self.tanh_state[idx] = @as(f32, @floatCast(std.math.tanh(self.state[i])));

            hidden[hidden_start + i] = self.output_gate.state[idx] * self.tanh_state[idx];
        }

        self.epoch += 1;
        if (self.epoch == self.horizon) {
            self.epoch = 0;
        }
    }

    fn clipGradients(self: *LstmLayer, arr: []f32) void {
        for (arr) |*val| {
            if (val.* < -self.gradient_clip) {
                val.* = -self.gradient_clip;
            } else if (val.* > self.gradient_clip) {
                val.* = self.gradient_clip;
            }
        }
    }

    fn backwardPassLayer(self: *LstmLayer, neurons: *NeuronLayer, input: []const f32, epoch: usize, input_symbol: usize, hidden_error: []f32) void {
        const offset = self.output_size + self.input_size;
        const row_size = offset + self.num_cells + 1;
        const epoch_offset = epoch * self.num_cells;

        if (epoch == self.horizon - 1) {
            @memset(neurons.gamma_u, 0.0);
            @memset(neurons.beta_u, 0.0);
            for (0..self.num_cells) |i| {
                @memset(neurons.update[i * row_size .. (i + 1) * row_size], 0.0);
                for (0..self.num_cells + 1) |j| {
                    neurons.transpose[j * self.num_cells + i] = neurons.weights[i * row_size + j + offset];
                }
            }
        }

        for (0..self.num_cells) |i| {
            const idx = epoch_offset + i;
            neurons.beta_u[i] += neurons.error_grad[i];
            neurons.gamma_u[i] += neurons.error_grad[i] * neurons.norm[idx];

            neurons.error_grad[i] *= neurons.gamma[i] * neurons.ivar[epoch];
        }

        var sum_error_norm: f32 = 0.0;
        for (0..self.num_cells) |i| {
            const idx = epoch_offset + i;
            sum_error_norm += neurons.error_grad[i] * neurons.norm[idx];
        }
        const mean_error_norm = sum_error_norm / @as(f32, @floatFromInt(self.num_cells));

        for (0..self.num_cells) |i| {
            const idx = epoch_offset + i;
            neurons.error_grad[i] -= mean_error_norm * neurons.norm[idx];
        }

        if (epoch > 0) {
            for (0..self.num_cells) |i| {
                var f: f32 = 0.0;
                for (0..self.num_cells) |j| {
                    f += neurons.error_grad[j] * neurons.transpose[i * self.num_cells + j];
                }
                hidden_error[i] += f;
            }
        }

        for (0..self.num_cells) |i| {
            const w_row = i * row_size;
            for (input, 0..) |inp_val, j| {
                neurons.update[w_row + self.output_size + j] += neurons.error_grad[i] * inp_val;
            }
            neurons.update[w_row + input_symbol] += neurons.error_grad[i];
        }

        if (epoch == 0) {
            for (0..self.num_cells) |i| {
                const w_row = i * row_size;
                self.adam(neurons.update[w_row .. w_row + row_size], neurons.m[w_row .. w_row + row_size], neurons.v[w_row .. w_row + row_size], neurons.weights[w_row .. w_row + row_size]);
            }
            self.adam(neurons.gamma_u, neurons.gamma_m, neurons.gamma_v, neurons.gamma);
            self.adam(neurons.beta_u, neurons.beta_m, neurons.beta_v, neurons.beta);
        }
    }

    fn adam(self: *LstmLayer, g: []f32, m: []f32, v: []f32, w: []f32) void {
        const beta1 = 0.025;
        const beta2 = 0.9999;
        const eps = 1e-6;
        const update_limit = 3000.0;

        const t = @as(f32, @floatFromInt(self.update_steps));
        const limit_t = if (t < update_limit) t else update_limit;

        const alpha = self.learning_rate * 0.1 / @sqrt(5e-5 * limit_t + 1.0);

        const correction1 = 1.0 - std.math.pow(f32, beta1, limit_t);
        const correction2 = 1.0 - std.math.pow(f32, beta2, limit_t);

        for (g, m, v, w) |g_val, *m_val, *v_val, *w_val| {
            m_val.* = m_val.* * beta1 + (1.0 - beta1) * g_val;
            v_val.* = v_val.* * beta2 + (1.0 - beta2) * g_val * g_val;
            w_val.* -= alpha * ((m_val.* / correction1) / (@sqrt(v_val.* / correction2) + eps));
        }
    }

    pub fn backwardPass(self: *LstmLayer, input: []const f32, epoch: usize, layer: usize, input_symbol: usize, hidden_error: []f32) void {
        _ = layer;
        const epoch_offset = epoch * self.num_cells;

        if (epoch == self.horizon - 1) {
            @memcpy(self.stored_error, hidden_error);
            @memset(self.state_error, 0.0);
        } else {
            for (0..self.num_cells) |i| {
                self.stored_error[i] += hidden_error[i];
            }
        }

        for (0..self.num_cells) |i| {
            const idx = epoch_offset + i;

            self.output_gate.error_grad[i] = self.tanh_state[idx] * self.stored_error[i] * self.output_gate.state[idx] * (1.0 - self.output_gate.state[idx]);

            self.state_error[i] += self.stored_error[i] * self.output_gate.state[idx] * (1.0 - (self.tanh_state[idx] * self.tanh_state[idx]));

            self.input_node.error_grad[i] = self.state_error[i] * self.input_gate_state[idx] * (1.0 - (self.input_node.state[idx] * self.input_node.state[idx]));

            const prev_state_val = self.last_state[epoch_offset + i];

            self.forget_gate.error_grad[i] = (prev_state_val - self.input_node.state[idx]) * self.state_error[i] * self.forget_gate.state[idx] * self.input_gate_state[idx];
        }

        @memset(hidden_error, 0.0);
        if (epoch > 0) {
            for (0..self.num_cells) |i| {
                const idx = epoch_offset + i;
                self.state_error[i] *= self.forget_gate.state[idx];
            }
            @memset(self.stored_error, 0.0);
        } else {
            if (self.update_steps < 3000) {
                self.update_steps += 1;
            }
        }

        self.backwardPassLayer(&self.forget_gate, input, epoch, input_symbol, hidden_error);
        self.backwardPassLayer(&self.input_node, input, epoch, input_symbol, hidden_error);
        self.backwardPassLayer(&self.output_gate, input, epoch, input_symbol, hidden_error);

        self.clipGradients(self.state_error);
        self.clipGradients(self.stored_error);
        self.clipGradients(hidden_error);
    }
};

pub const Lstm = struct {
    layers: LstmLayer,
    input_history: []u8,
    hidden: []f32,
    hidden_error: []f32,
    output_layer: []f32, // horizon * vocab_size * (num_cells + 1)
    output: []f32, // horizon * vocab_size
    layer_input: []f32, // horizon * (input_size + num_cells + 1)

    learning_rate: f32,
    num_cells: usize,
    epoch: usize = 0,
    horizon: usize,
    input_size: usize,
    output_size: usize,
    last_input: i32 = -1,

    pub fn init(allocator: std.mem.Allocator, input_size: usize, output_size: usize, num_cells: usize, horizon: usize, learning_rate: f32, gradient_clip: f32, random: std.Random) !Lstm {
        const layers = try LstmLayer.init(allocator, input_size + 1 + num_cells + output_size, input_size, output_size, num_cells, horizon, gradient_clip, learning_rate, random);

        const input_history = try allocator.alloc(u8, horizon);
        @memset(input_history, 0);

        const hidden = try allocator.alloc(f32, num_cells + 1);
        @memset(hidden, 0.0);
        hidden[num_cells] = 1.0;

        const hidden_error = try allocator.alloc(f32, num_cells);
        @memset(hidden_error, 0.0);

        const output_layer = try allocator.alloc(f32, horizon * output_size * (num_cells + 1));
        @memset(output_layer, 0.0);

        const output = try allocator.alloc(f32, horizon * output_size);
        for (0..horizon) |h| {
            const out_slice = output[h * output_size .. (h + 1) * output_size];
            @memset(out_slice, 1.0 / @as(f32, @floatFromInt(output_size)));
        }

        const layer_input_row_size = input_size + num_cells + 1;
        const layer_input = try allocator.alloc(f32, horizon * layer_input_row_size);
        @memset(layer_input, 0.0);
        for (0..horizon) |h| {
            layer_input[h * layer_input_row_size + layer_input_row_size - 1] = 1.0;
        }

        return Lstm{
            .layers = layers,
            .input_history = input_history,
            .hidden = hidden,
            .hidden_error = hidden_error,
            .output_layer = output_layer,
            .output = output,
            .layer_input = layer_input,
            .learning_rate = learning_rate,
            .num_cells = num_cells,
            .horizon = horizon,
            .input_size = input_size,
            .output_size = output_size,
        };
    }

    pub fn deinit(self: *Lstm, allocator: std.mem.Allocator) void {
        self.layers.deinit(allocator);
        allocator.free(self.input_history);
        allocator.free(self.hidden);
        allocator.free(self.hidden_error);
        allocator.free(self.output_layer);
        allocator.free(self.output);
        allocator.free(self.layer_input);
    }

    pub fn setInput(self: *Lstm, inputs: []const f32) void {
        const row_size = self.input_size + self.num_cells + 1;
        const start = self.epoch * row_size;
        @memcpy(self.layer_input[start .. start + self.input_size], inputs[0..self.input_size]);
    }

    pub fn predict(self: *Lstm, input_symbol: usize) []const f32 {
        const row_size = self.input_size + self.num_cells + 1;
        const epoch_offset = self.epoch * row_size;

        @memcpy(self.layer_input[epoch_offset + self.input_size .. epoch_offset + self.input_size + self.num_cells], self.hidden[0..self.num_cells]);

        const input_slice = self.layer_input[epoch_offset .. epoch_offset + row_size];
        self.layers.forwardPass(input_slice, input_symbol, self.hidden, 0);

        const output_epoch = self.epoch * self.output_size;
        const output_layer_epoch = self.epoch * self.output_size * (self.num_cells + 1);

        var sum_exp: f32 = 0.0;
        for (0..self.output_size) |i| {
            var sum: f32 = 0.0;
            const w_row = i * (self.num_cells + 1);
            for (self.hidden, 0..) |h_val, j| {
                sum += h_val * self.output_layer[output_layer_epoch + w_row + j];
            }
            const out_val = @exp(sum);
            self.output[output_epoch + i] = out_val;
            sum_exp += out_val;
        }

        const out_slice = self.output[output_epoch .. output_epoch + self.output_size];
        for (out_slice) |*val| {
            val.* /= sum_exp;
        }

        const prev_epoch = self.epoch;
        self.epoch += 1;
        if (self.epoch == self.horizon) {
            self.epoch = 0;
        }
        self.last_input = @intCast(input_symbol);

        return self.output[prev_epoch * self.output_size .. (prev_epoch + 1) * self.output_size];
    }

    pub fn perceive(self: *Lstm, input_symbol: usize) []const f32 {
        const last_epoch = if (self.epoch > 0) self.epoch - 1 else self.horizon - 1;
        const old_input = self.input_history[last_epoch];
        self.input_history[last_epoch] = @intCast(input_symbol);

        if (self.epoch == 0) {
            @memset(self.hidden_error, 0.0);
            var epoch: usize = self.horizon;
            while (epoch > 0) {
                epoch -= 1;

                const output_epoch = epoch * self.output_size;
                const output_layer_epoch = epoch * self.output_size * (self.num_cells + 1);
                const target_symbol = self.input_history[epoch];

                for (0..self.output_size) |i| {
                    const error_val = if (i == target_symbol) self.output[output_epoch + i] - 1.0 else self.output[output_epoch + i];
                    const w_row = i * (self.num_cells + 1);
                    for (0..self.num_cells) |j| {
                        self.hidden_error[j] += self.output_layer[output_layer_epoch + w_row + j] * error_val;
                    }
                }

                const prev_epoch = if (epoch > 0) epoch - 1 else self.horizon - 1;
                const prev_input_symbol = if (epoch == 0) old_input else self.input_history[prev_epoch];

                const row_size = self.input_size + self.num_cells + 1;
                const input_slice = self.layer_input[epoch * row_size .. (epoch + 1) * row_size];
                self.layers.backwardPass(input_slice, epoch, 0, prev_input_symbol, self.hidden_error);
            }
        }

        const output_epoch = last_epoch * self.output_size;
        const output_layer_last_epoch = last_epoch * self.output_size * (self.num_cells + 1);
        const output_layer_epoch = self.epoch * self.output_size * (self.num_cells + 1);

        for (0..self.output_size) |i| {
            const error_val = if (i == input_symbol) self.output[output_epoch + i] - 1.0 else self.output[output_epoch + i];
            const w_row = i * (self.num_cells + 1);
            for (self.hidden, 0..) |h_val, j| {
                const idx_last = output_layer_last_epoch + w_row + j;
                const idx_curr = output_layer_epoch + w_row + j;
                self.output_layer[idx_curr] = self.output_layer[idx_last] - self.learning_rate * error_val * h_val;
            }
        }

        return self.predict(input_symbol);
    }
};
