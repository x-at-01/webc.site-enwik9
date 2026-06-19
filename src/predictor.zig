const std = @import("std");
const lstm_mod = @import("lstm.zig");
const Lstm = lstm_mod.Lstm;
const fxcm = @import("fxcm.zig");
const Fxcm = fxcm.Fxcm;

const tables = @import("tables.zig");
const nonstationary_table = tables.nonstationary_table;
const wrt_2b = tables.wrt_2b;
const wrt_3b = tables.wrt_3b;
const wrt_4b = tables.wrt_4b;
const wrt_5b = tables.wrt_5b;
const run_map_table = tables.run_map_table;

const indirect = @import("indirect.zig");
const IndirectModel = indirect.IndirectModel;

const runmap = @import("runmap.zig");
const RunMapModel = runmap.RunMapModel;

const match = @import("match.zig");
const MatchModel = match.MatchModel;

const bracket = @import("bracket.zig");
const BracketModel = bracket.BracketModel;

const ppmd = @import("ppmd.zig");
const PpmModel = ppmd.PpmModel;

const sse = @import("sse.zig");
const ShelwienSSE = sse.ShelwienSSE;

const mixer = @import("mixer.zig");
const Mixer = mixer.Mixer;

const direct = @import("direct.zig");
const DirectModel = direct.DirectModel;

pub inline fn logit(p: f32) f32 {
    const min_p = 0.0001;
    const max_p = 0.9999;
    const cp = if (p < min_p) min_p else if (p > max_p) max_p else p;
    return @log(cp / (1.0 - cp));
}

pub inline fn logistic(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

inline fn hashBytes(bytes: []const u8) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (bytes) |b| {
        h = (h ^ b) *% 0x100000001b3;
    }
    return h;
}

const order_deltas = [_]f32{
    100.0, 150.0, 200.0, 250.0, 300.0, 300.0, 300.0, 350.0,
    350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0, 350.0,
    350.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0,
};

const sparse_deltas = [_]f32{
    200.0, 200.0, 200.0, 200.0, 250.0, 250.0, 250.0, 250.0, 250.0, 250.0,
};

const run_deltas = [_]f32{ 100.0, 150.0, 200.0, 250.0 };
const double_ind_deltas = [_]f32{400.0} ** 4;

// Comptime configuration meta for array models
const model_fields = .{
    .{ .name = "orders", .offset = 0, .has_map_bc = true },
    .{ .name = "sparse_models", .offset = 24, .has_map_bc = true },
    .{ .name = "run_models", .offset = 34, .has_map_bc = true },
    .{ .name = "match_models", .offset = 38, .has_map_bc = false },
    .{ .name = "double_indirect_models", .offset = 49, .has_map_bc = true },
};

pub const Predictor = struct {
    vocab: [256]bool,
    is_possible: [512]bool,
    shared_map: []u8,
    history: []u8,
    history_pos: usize = 0,
    bit_context: u32 = 1,
    steps: usize = 0,
    bpos: u3 = 0,

    // Double Indirect Hash vectors
    hashes_ind1: []u8,
    hashes_ind2: []u32,
    hashes_ind3: []u32,
    hashes_ind5: []u32,

    ind1: u64 = 0,
    ind2: u64 = 0,
    ind3: u64 = 0,
    ind5: u64 = 0,
    context1_ind: u64 = 0,
    context1_ind2: u64 = 0,
    context1_ind3: u64 = 0,
    context1_ind5: u64 = 0,

    // Word and stream context states
    words: [8]u64 = .{0} ** 8,
    recent_bytes: [8]u8 = .{0} ** 8,
    b2stream: u64 = 0,
    b3stream: u64 = 0,
    b4stream: u64 = 0,
    stream2bR: u64 = 0,
    stream3bR: u64 = 0,
    o2bState: u8 = 0,
    n2bState: u8 = 0,
    o3bState: u8 = 0,
    n3bState: u8 = 0,
    mx5: u64 = 0,
    mx6: u64 = 0,
    mx7: u64 = 0,
    mx8: u64 = 0,
    mx9: u64 = 0,
    mx9cxt: u64 = 0,
    mx10: u64 = 0,
    mx10cxt: u64 = 0,
    mx11: u64 = 0,
    mx11cxt: u64 = 0,
    mx12: u64 = 0,
    mx12cxt: u64 = 0,
    mx13: u64 = 0,
    mx13cxt: u64 = 0,
    mx14: u64 = 0,
    mx15: u64 = 0,
    mx16: u64 = 0,
    mx17: u64 = 0,
    mx18: u64 = 0,
    mx18cxt: u64 = 0,
    words_state: u64 = 0,
    wordscxt: u64 = 0,
    line_break: u64 = 0,
    longest_match: u64 = 0,
    auxiliary_context: u64 = 0,
    b2streamcxt: u64 = 0,
    b3streamcxt: u64 = 0,

    // Base models grouped into arrays
    orders: [24]IndirectModel,
    sparse_models: [10]IndirectModel,
    run_models: [4]RunMapModel,
    match_models: [11]MatchModel,
    double_indirect_models: [4]IndirectModel,
    direct_bracket_model: DirectModel,

    // Shelwien SSE model
    sse: ShelwienSSE,

    // PPM Model
    ppm: PpmModel,

    // Bracket Model
    bracket: BracketModel,

    // 23 First-stage mixers
    mixers: [23]Mixer,

    // Layer 1 Mixer
    mixer_l1: Mixer,

    // FXCM Model
    fxcm: Fxcm,

    // Cache predictions and logits (increased size to 57 + 431 = 488 for direct_bracket_model + fxcm)
    model_predictions: [488]f32 = undefined,
    inputs: [488]f32 = undefined,
    first_stage_logits: [23]f32 = undefined,
    first_stage_outputs: [511]f32 = undefined,
    final_l1_logit: f32 = 0.0,
    p_l1: f32 = 0.0,

    lstm: ?Lstm = null,
    byte_map: [256]u8 = undefined,
    byte_mixer_inputs: []f32 = undefined,
    byte_mixer_probs: [256]f32 = .{0.0} ** 256,
    byte_mixer_tree: [512]f32 = .{1.0} ** 512,

    pub fn init(allocator: std.mem.Allocator, vocab: [256]bool) !*Predictor {
        const shared_map = try allocator.alloc(u8, 256 * 1000000); // 256 MB
        @memset(shared_map, 0);

        const history = try allocator.alloc(u8, 1000000 + 100);
        @memset(history, 0);

        const hashes_ind1 = try allocator.alloc(u8, 0x1000000);
        @memset(hashes_ind1, 0);
        const hashes_ind2 = try allocator.alloc(u32, 0x1000000);
        @memset(hashes_ind2, 0);
        const hashes_ind3 = try allocator.alloc(u32, 0x2000000);
        @memset(hashes_ind3, 0);
        const hashes_ind5 = try allocator.alloc(u32, 0x100);
        @memset(hashes_ind5, 0);

        var prng = std.Random.DefaultPrng.init(12345);
        const random = prng.random();

        const randOffset = struct {
            fn get(r: std.Random, max: usize) usize {
                return r.intRangeLessThan(usize, 0, max - 257);
            }
        }.get;

        const limit = shared_map.len;

        var is_possible = [_]bool{false} ** 512;
        for (0..256) |i| {
            is_possible[256 + i] = vocab[i];
        }
        var ip: usize = 255;
        while (ip >= 1) : (ip -= 1) {
            is_possible[ip] = is_possible[2 * ip] or is_possible[2 * ip + 1];
        }

        var vocab_size: usize = 0;
        for (vocab) |v| {
            if (v) vocab_size += 1;
        }

        var byte_map = [_]u8{0} ** 256;
        var offset: u8 = 0;
        for (0..256) |i| {
            byte_map[i] = offset;
            if (vocab[i]) {
                offset += 1;
            }
        }

        const byte_mixer_inputs = try allocator.alloc(f32, vocab_size);
        @memset(byte_mixer_inputs, 0.0);

        const lstm_model = try Lstm.init(allocator, vocab_size, vocab_size, 200, 128, 0.03, 10.0, random);

        const self = try allocator.create(Predictor);
        errdefer allocator.destroy(self);

        self.* = Predictor{
            .vocab = vocab,
            .is_possible = is_possible,
            .shared_map = shared_map,
            .history = history,
            .history_pos = 0,
            .bit_context = 1,
            .steps = 0,
            .bpos = 0,
            .hashes_ind1 = hashes_ind1,
            .hashes_ind2 = hashes_ind2,
            .hashes_ind3 = hashes_ind3,
            .hashes_ind5 = hashes_ind5,
            .ind1 = 0,
            .ind2 = 0,
            .ind3 = 0,
            .ind5 = 0,
            .context1_ind = 0,
            .context1_ind2 = 0,
            .context1_ind3 = 0,
            .context1_ind5 = 0,
            .words = .{0} ** 8,
            .recent_bytes = .{0} ** 8,
            .b2stream = 0,
            .b3stream = 0,
            .b4stream = 0,
            .stream2bR = 0,
            .stream3bR = 0,
            .o2bState = 0,
            .n2bState = 0,
            .o3bState = 0,
            .n3bState = 0,
            .mx5 = 0,
            .mx6 = 0,
            .mx7 = 0,
            .mx8 = 0,
            .mx9 = 0,
            .mx9cxt = 0,
            .mx10 = 0,
            .mx10cxt = 0,
            .mx11 = 0,
            .mx11cxt = 0,
            .mx12 = 0,
            .mx12cxt = 0,
            .mx13 = 0,
            .mx13cxt = 0,
            .mx14 = 0,
            .mx15 = 0,
            .mx16 = 0,
            .mx17 = 0,
            .mx18 = 0,
            .mx18cxt = 0,
            .words_state = 0,
            .wordscxt = 0,
            .line_break = 0,
            .longest_match = 0,
            .auxiliary_context = 0,
            .b2streamcxt = 0,
            .b3streamcxt = 0,

            .orders = undefined,
            .sparse_models = undefined,
            .run_models = undefined,
            .match_models = undefined,
            .double_indirect_models = undefined,
            .direct_bracket_model = try DirectModel.init(allocator, 51400, 30, 0.0),

            .sse = undefined,
            .ppm = try PpmModel.init(allocator, vocab),
            .bracket = try BracketModel.init(allocator, vocab),
            .fxcm = try Fxcm.init(allocator),
            .mixers = undefined,
            .mixer_l1 = undefined,
            .lstm = lstm_model,
            .byte_map = byte_map,
            .byte_mixer_inputs = byte_mixer_inputs,
            .byte_mixer_probs = [_]f32{0.0} ** 256,
            .byte_mixer_tree = [_]f32{1.0} ** 512,
            .final_l1_logit = 0.0,
            .p_l1 = 0.0,
        };

        // Initialize array models using comptime configs
        const init_configs = .{
            .{ .name = "orders", .Type = IndirectModel, .deltas = &order_deltas },
            .{ .name = "sparse_models", .Type = IndirectModel, .deltas = &sparse_deltas },
            .{ .name = "run_models", .Type = RunMapModel, .deltas = &run_deltas },
            .{ .name = "double_indirect_models", .Type = IndirectModel, .deltas = &double_ind_deltas },
        };
        inline for (init_configs) |cfg| {
            const array = &@field(self, cfg.name);
            inline for (array, 0..) |*model, i| {
                model.* = cfg.Type.init(randOffset(random, limit), cfg.deltas[i]);
            }
        }

        // Initialize match models
        inline for (&self.match_models) |*model| {
            model.* = try MatchModel.init(allocator, 2000000, 200, 0.5);
        }

        // Initialize mixers: mixing 488 input models + i extra inputs
        const learning_rates = [_]f32{
            0.005, 0.0005, 0.005, 0.0005, 0.005, 0.001, 0.002, 0.0007, 0.0005,
            0.002, 0.0005, 0.001, 0.001,  0.005, 0.001, 0.001, 0.005,  0.001,
            0.001, 0.005,  0.005, 0.005,  0.005,
        };
        for (&self.mixers, 0..) |*mixer_ptr, i| {
            mixer_ptr.* = try Mixer.init(allocator, 131072, 488 + i, learning_rates[i]);
        }

        // Layer 1 Mixer: mixes 23 mixers + 488 base models = 511 inputs
        self.mixer_l1 = try Mixer.init(allocator, 1, 511, 0.0005);
        try self.sse.init(allocator);

        return self;
    }

    pub fn deinit(self: *Predictor, allocator: std.mem.Allocator) void {
        allocator.free(self.shared_map);
        allocator.free(self.history);
        allocator.free(self.hashes_ind1);
        allocator.free(self.hashes_ind2);
        allocator.free(self.hashes_ind3);
        allocator.free(self.hashes_ind5);

        inline for (&self.match_models) |*model| {
            model.deinit(allocator);
        }
        self.direct_bracket_model.deinit(allocator);

        self.sse.deinit(allocator);
        self.ppm.deinit(allocator);
        self.bracket.deinit();
        self.fxcm.deinit();
        for (&self.mixers) |*m| {
            m.deinit(allocator);
        }
        self.mixer_l1.deinit(allocator);
        if (self.lstm) |*l| {
            l.deinit(allocator);
        }
        allocator.free(self.byte_mixer_inputs);
        allocator.destroy(self);
    }

    inline fn getLstmprLstmex(self: *const Predictor) struct { pr: i32, ex: i32 } {
        const bc = self.bit_context;
        const L = 31 - @clz(bc);
        const step = @as(u32, 256) >> @intCast(L);
        const bot = (bc - (@as(u32, 1) << @intCast(L))) * step;
        const top = bot + step - 1;
        var lstmex: i32 = @intCast(bot);
        var max_prob_val = self.byte_mixer_probs[bot];
        var i_idx = bot + 1;
        while (i_idx <= top) : (i_idx += 1) {
            if (self.byte_mixer_probs[i_idx] > max_prob_val) {
                max_prob_val = self.byte_mixer_probs[i_idx];
                lstmex = @intCast(i_idx);
            }
        }

        const lstmpr_float = self.predict_lstm_bit(bc);
        if (std.math.isNan(lstmpr_float)) {
            std.debug.print("PANIC: lstmpr_float is NaN! bc={d}, tree[1]=0x{x}, tree[3]=0x{x}\n", .{
                bc,
                @as(u32, @bitCast(self.byte_mixer_tree[1])),
                @as(u32, @bitCast(self.byte_mixer_tree[3])),
            });
            std.process.exit(1);
        }
        const lstmpr: i32 = @intFromFloat(1.0 + 4094.0 * lstmpr_float);

        return .{ .pr = lstmpr, .ex = lstmex };
    }

    inline fn getByte(self: *const Predictor, back: usize) u8 {
        if (self.history_pos < back) return 0;
        return self.history[self.history_pos - back];
    }

    inline fn hashHistory(self: *const Predictor, offset: usize, len: usize) u64 {
        if (self.history_pos < offset + len) return 0;
        const start = self.history_pos - offset - len;
        const end = self.history_pos - offset;
        return hashBytes(self.history[start..end]);
    }

    pub fn predict_lstm_bit(self: *const Predictor, bc: u32) f32 {
        const sum = self.byte_mixer_tree[bc];
        if (sum <= 0.000001) return 0.5;
        const p1 = self.byte_mixer_tree[bc * 2 + 1] / sum;
        if (p1 < 0.0001) return 0.0001;
        if (p1 > 0.9999) return 0.9999;
        return p1;
    }

    pub fn predict(self: *Predictor) f32 {
        const map = self.shared_map;
        const bc = self.bit_context;

        // 1. Gather predictions from base models using comptime config map
        inline for (model_fields) |field| {
            const array = &@field(self, field.name);
            inline for (array, 0..) |*model, i| {
                if (field.has_map_bc) {
                    self.model_predictions[field.offset + i] = model.predict(map, bc);
                } else {
                    self.model_predictions[field.offset + i] = model.predict();
                }
            }
        }
        self.model_predictions[53] = self.ppm.predict_bit(bc);
        self.model_predictions[54] = self.bracket.predict();
        self.model_predictions[55] = self.predict_lstm_bit(bc);
        self.model_predictions[56] = self.direct_bracket_model.predict(bc);

        // Feed FXCM predictions
        const lstm = self.getLstmprLstmex();
        _ = self.fxcm.predict(lstm.pr, lstm.ex);
        @memcpy(self.model_predictions[57..488], &fxcm.model_predictions);

        // 2. Convert to logit domain
        for (&self.inputs, self.model_predictions) |*input, p| {
            input.* = logit(p);
        }

        // 3. Compute Mixer contexts and mix
        const bc_u64: u64 = self.bit_context;

        // Compute mxx context dynamically based on bpos
        var mxx: u64 = 0;
        if (self.bpos == 0) {
            mxx = (self.stream2bR & 63) * 8 + (self.b3stream & 7);
        } else if (self.bpos > 3) {
            const bc_shifted = (bc_u64 << @as(u6, @intCast(8 - @as(u32, self.bpos)))) & 255;
            const c_val = wrt_2b[bc_shifted];
            mxx = ((self.b2stream << 2) & 63) + @as(u64, c_val) * 8 + (self.b3stream & 7);
        } else {
            mxx = (self.stream2bR & 63) * 8 + (self.b3stream & 7);
        }

        // Proxy for mx19cxt
        const mx19cxt = 0x10000 + (self.b2stream & 0xffff);

        // Proxy for auxiliary_context
        const p_order3 = self.model_predictions[2];
        const p_ppm = self.model_predictions[53];
        const auxiliary_average = (p_order3 + p_ppm) / 2.0;
        self.auxiliary_context = @as(u64, @intFromFloat(auxiliary_average * 15.0));

        // Context selections matching C++ structures exactly
        self.mixers[0].selectContext(self.mx9);
        self.mixers[1].selectContext(self.mx10);
        self.mixers[2].selectContext(self.mx11);
        self.mixers[3].selectContext(self.mx12);
        self.mixers[4].selectContext(self.mx13);
        self.mixers[5].selectContext(mxx);
        self.mixers[6].selectContext(self.recent_bytes[2]);
        self.mixers[7].selectContext(self.line_break);
        self.mixers[8].selectContext(self.longest_match);
        self.mixers[9].selectContext(mx19cxt);
        self.mixers[10].selectContext(self.auxiliary_context);
        self.mixers[11].selectContext(self.mx18);
        self.mixers[12].selectContext(self.mx7);
        self.mixers[13].selectContext(self.wordscxt);
        self.mixers[14].selectContext(self.b2streamcxt);
        self.mixers[15].selectContext(self.mx5);
        self.mixers[16].selectContext(self.mx6);
        self.mixers[17].selectContext(self.b3streamcxt);
        self.mixers[18].selectContext(self.mx8);
        self.mixers[19].selectContext(self.mx17);
        self.mixers[20].selectContext(self.mx16);
        self.mixers[21].selectContext(self.mx14);
        self.mixers[22].selectContext(self.mx15);

        // Mix at stage 1 (mixing 488 input models + i extra inputs)
        for (&self.first_stage_logits, &self.mixers, 0..) |*logit_val, m, i| {
            var mixer_inputs: [510]f32 = undefined;
            @memcpy(mixer_inputs[0..488], &self.inputs);
            @memcpy(mixer_inputs[488 .. 488 + i], self.first_stage_logits[0..i]);
            logit_val.* = m.mix(mixer_inputs[0 .. 488 + i]);
        }

        // Layer 1 inputs setup: 23 first stage logits + 488 base model inputs = 511 inputs
        @memcpy(self.first_stage_outputs[0..23], &self.first_stage_logits);
        @memcpy(self.first_stage_outputs[23..511], &self.inputs);

        // Layer 1 Mix
        self.mixer_l1.selectContext(0);
        self.final_l1_logit = self.mixer_l1.mix(&self.first_stage_outputs);
        self.p_l1 = logistic(self.final_l1_logit);
        if (std.math.isNan(self.p_l1)) {
            return std.math.nan(f32);
        }

        return self.sse.predict(self.p_l1);
    }

    pub fn perceive(self: *Predictor, bit: u1) void {
        const decay = getDecay(self.steps);

        // 1. Train Shelwien SSE model
        self.sse.perceive(bit);

        // 2. Train Layer 1 mixer
        self.mixer_l1.perceive(&self.first_stage_outputs, self.final_l1_logit, bit, decay);

        // 3. Train first-stage mixers
        for (&self.mixers, self.first_stage_logits, 0..) |*m, mixed_logit, i| {
            var mixer_inputs: [510]f32 = undefined;
            @memcpy(mixer_inputs[0..488], &self.inputs);
            @memcpy(mixer_inputs[488 .. 488 + i], self.first_stage_logits[0..i]);
            m.perceive(mixer_inputs[0 .. 488 + i], mixed_logit, bit, decay);
        }

        // 5. Train base models using comptime config map
        const map = self.shared_map;
        const bc = self.bit_context;

        inline for (model_fields) |field| {
            if (field.has_map_bc) {
                inline for (&@field(self, field.name)) |*model| {
                    model.perceive(map, bc, bit);
                }
            }
        }
        self.direct_bracket_model.perceive(bc, bit);

        // Sparse word-based contexts for match models
        const s_0 = self.words[0];
        const s_1 = self.words[1];
        const s_1_3 = self.words[1] +% 256 *% self.words[3];
        const s_1_2_3 = self.words[1] +% 256 *% self.words[2] +% 899 *% self.words[3];
        const s_7_2 = self.words[7] +% 256 *% self.words[2];
        const c1 = self.getByte(1);
        const c2 = self.getByte(2);
        const c3 = self.getByte(3);
        const o4 = (@as(u64, c1) << 24) | (@as(u64, c2) << 16) | (@as(u64, c3) << 8) | self.getByte(4);

        // Hash contexts for match6 to match10
        const h_0_8 = self.hashHistory(0, 8);
        const h_1_8 = self.hashHistory(1, 8);
        const h_7_4 = self.hashHistory(7, 4);
        const h_11_3 = self.hashHistory(11, 3);
        const h_13_2 = self.hashHistory(13, 2);

        const match_contexts = .{
            s_0,
            s_1,
            s_1_3,
            s_1_2_3,
            s_7_2,
            o4,
            h_0_8,
            h_1_8,
            h_7_4,
            h_11_3,
            h_13_2,
        };
        inline for (&self.match_models, 0..) |*model, i| {
            model.perceive(match_contexts[i], bc, bit, self.history_pos);
        }

        self.bracket.perceive(bit);

        const lstm = self.getLstmprLstmex();
        self.fxcm.perceive(bit, lstm.pr, lstm.ex);

        // 6. Update state variables
        self.steps += 1;
        self.bpos = (self.bpos +% 1) & 7;
        self.bit_context = self.bit_context * 2 + bit;

        // Byte Boundary check
        if (self.bit_context >= 256) {
            self.byteUpdate();
        }

        // Bit-level context updates (matching C++ long_bit_context_ behavior)
        const bc_u64 = @as(u64, self.bit_context);
        self.wordscxt = (self.words_state & 0x7F) *% 256 +% bc_u64;
        self.mx6 = (self.stream2bR & 0xff) *% 256 +% bc_u64;
        self.mx9 = (self.mx9cxt) *% 256 +% bc_u64;
        self.mx10 = (self.mx10cxt) *% 256 +% bc_u64;
        self.mx11 = (self.mx11cxt) *% 256 +% bc_u64;
        self.mx12 = bc_u64;
        self.mx13 = bc_u64;
        self.mx16 = @as(u64, self.recent_bytes[1]) *% 256 +% bc_u64;
        self.mx17 = (self.b3stream & 0x3f) *% 256 +% bc_u64;
    }

    fn byteUpdate(self: *Predictor) void {
        const byte_val: u8 = @intCast(self.bit_context - 256);
        self.bit_context = 1;

        // Save decoded byte in history
        self.history[self.history_pos] = byte_val;
        self.history_pos += 1;

        self.longest_match = 0;

        if (byte_val == '\n') {
            self.line_break = 0;
        } else if (self.line_break < 99) {
            self.line_break += 1;
        }

        // Word parser
        const c = byte_val;
        if (c == 'R' or c == 'P' or c == 93) {
            self.b3stream = (self.b3stream & 0xfffffff8) +% 3;
        } else if (c == '=') {
            self.b3stream = (self.b3stream & 0xfffffff8) +% 4;
        }
        self.n2bState = wrt_2b[c];
        self.b2stream = self.b2stream *% 4 +% self.n2bState;
        self.n3bState = wrt_3b[c];
        self.b3stream = self.b3stream *% 8 +% self.n3bState;
        if (self.o3bState != self.n3bState) {
            self.stream3bR = (self.stream3bR << 3) +% self.n3bState;
            self.o3bState = self.n3bState;
        }
        if (c == 10 or c == ')') {
            self.b3stream <<= 6;
        }
        if (c == 'Q') {
            self.b3stream = self.b3stream *% 8 +% wrt_3b[c];
        }
        self.b2streamcxt = self.b2stream & 0x3ff;
        self.b3streamcxt = self.b3stream & 0x1ff;

        if (self.o2bState != self.n2bState) {
            self.stream2bR = (self.stream2bR << 2) +% self.n2bState;
            self.o2bState = self.n2bState;
        }
        self.b4stream = self.b4stream *% 16 +% wrt_4b[c];
        self.mx18cxt = self.mx18cxt *% 16 +% wrt_5b[c];
        self.mx18 = self.mx18cxt & 0xff;

        self.words_state = self.words_state *% 2;

        if ((c >= 'a' and c <= 'z') or c >= 0x80) {
            self.words[7] = self.words[7] *% 15952 +% c;
            if (self.recent_bytes[0] != 12) {
                self.words_state += 1;
            }
        } else {
            self.words[7] = 0;
        }

        if ((c >= 'a' and c <= 'z') or (c >= '0' and c <= '9') or c == 8 or c == 6 or c >= 0x80) {
            self.words[0] = (self.words[0] *% 15952 +% c) & 0xfffffff;
            self.words[1] = self.words[1] *% 8416 +% c;
        } else {
            var i: usize = 6;
            while (i >= 2) : (i -= 1) {
                self.words[i] = self.words[i - 1];
            }
            self.words[1] = 0;
        }

        if (c == 10) {
            self.words_state = 0xfffc;
        } else if (c == '.') {
            self.words_state |= 0xffc;
        } else if (c == ',') {
            self.words_state |= 0xffc;
        }

        self.mx5 = self.b2stream & 0xffff;
        self.mx7 = self.b4stream & 0xff;
        self.mx8 = (self.mx8 *% 4 +% (self.b3stream & 0x3f)) & 0x3FFF;
        self.mx9cxt = (self.mx9cxt *% 16 +% c) & 0xff;

        self.mx10cxt = c;
        self.mx11cxt = c;
        self.mx12cxt = 0;
        self.mx13cxt = 0;

        self.mx14 = @as(u64, c) *% 256 +% self.recent_bytes[0];
        self.mx15 = @as(u64, self.recent_bytes[0]) *% 256 +% self.recent_bytes[1];

        // Double Indirect Hash Updates
        const ind1_next = ((self.ind1 *% 256) +% c) & 0xff;
        self.hashes_ind1[self.context1_ind] = @intCast(ind1_next);
        self.context1_ind = (self.context1_ind *% 256 +% c) & 0xffffff;
        self.ind1 = self.hashes_ind1[self.context1_ind];

        const ind2_next = ((self.ind2 *% 256) +% c) & 0xffffffff;
        self.hashes_ind2[self.context1_ind2] = @intCast(ind2_next);
        self.context1_ind2 = (self.context1_ind2 *% 64 +% c) & 0xffffff;
        self.ind2 = self.hashes_ind2[self.context1_ind2];

        const ind3_next = ((self.ind3 *% 32) +% c) & 0x1ffffff;
        self.hashes_ind3[self.context1_ind3] = @intCast(ind3_next);
        self.context1_ind3 = (self.context1_ind3 *% 32 +% c) & 0x1ffffff;
        self.ind3 = self.hashes_ind3[self.context1_ind3];

        const ind5_next = ((self.ind5 *% 64) +% c) & 0x3fffffff;
        self.hashes_ind5[self.context1_ind5] = @intCast(ind5_next);
        self.context1_ind5 = c;
        self.ind5 = self.hashes_ind5[self.context1_ind5];

        self.wordscxt = (self.words_state & 0x7F) *% 256 +% self.bit_context;

        // Shift recent_bytes (matching C++ context order)
        var r: usize = 7;
        while (r >= 1) : (r -= 1) {
            self.recent_bytes[r] = self.recent_bytes[r - 1];
        }
        self.recent_bytes[0] = byte_val;

        // Context updating for order-K models
        var orders_ctx: [24]u64 = undefined;
        const c1: u64 = self.recent_bytes[0];
        const c2: u64 = self.recent_bytes[1];
        const c3: u64 = self.recent_bytes[2];
        const c4: u64 = self.recent_bytes[3];

        orders_ctx[0] = c1;
        orders_ctx[1] = (c1 << 8) | c2;
        orders_ctx[2] = (c1 << 16) | (c2 << 8) | c3;
        orders_ctx[3] = (c1 << 24) | (c2 << 16) | (c3 << 8) | c4;

        inline for (4..24) |i| {
            orders_ctx[i] = (orders_ctx[i - 1] *% 133) +% self.getByte(i + 1);
        }

        const map_len = self.shared_map.len;

        // Sparse models contexts updated dynamically
        const s_0 = self.words[0];
        const s_0_1 = self.words[0] +% 256 *% self.words[1];
        const s_1 = self.words[1];
        const s_1_2 = self.words[1] +% 256 *% self.words[2];
        const s_1_3 = self.words[1] +% 256 *% self.words[3];
        const s_2_3 = self.words[2] +% 256 *% self.words[3];
        const s_3_4 = self.words[3] +% 256 *% self.words[4];
        const s_1_2_4 = self.words[1] +% 256 *% self.words[2] +% 899 *% self.words[4];
        const s_2_3_4 = self.words[2] +% 256 *% self.words[3] +% 899 *% self.words[4];
        const s_2 = self.words[2];
        const s_1_2_3 = self.words[1] +% 256 *% self.words[2] +% 899 *% self.words[3];
        const s_7_2 = self.words[7] +% 256 *% self.words[2];

        const sparse_ctx = .{
            s_0,
            s_0_1,
            s_1,
            s_1_2,
            s_1_3,
            s_2_3,
            s_3_4,
            s_1_2_4,
            s_2_3_4,
            s_2,
        };

        const run_ctx = .{ s_1, s_2, s_1_3, s_1_2_3 };

        const history_slice = self.history[0..self.history_pos];

        // Hash contexts for match6 to match10
        const h_0_8 = self.hashHistory(0, 8);
        const h_1_8 = self.hashHistory(1, 8);
        const h_7_4 = self.hashHistory(7, 4);
        const h_11_3 = self.hashHistory(11, 3);
        const h_13_2 = self.hashHistory(13, 2);

        const match_update_ctx = .{
            s_0,
            s_1,
            s_1_3,
            s_1_2_3,
            s_7_2,
            orders_ctx[3], // o4
            h_0_8,
            h_1_8,
            h_7_4,
            h_11_3,
            h_13_2,
        };

        const double_ind_ctx = .{ self.ind1, self.ind2, self.ind3, self.ind5 };

        // Unified byte update loop utilizing comptime update config
        const update_configs = .{
            .{ .name = "orders", .ctxs = orders_ctx, .use_map = true },
            .{ .name = "sparse_models", .ctxs = sparse_ctx, .use_map = true },
            .{ .name = "run_models", .ctxs = run_ctx, .use_map = true },
            .{ .name = "match_models", .ctxs = match_update_ctx, .use_map = false },
            .{ .name = "double_indirect_models", .ctxs = double_ind_ctx, .use_map = true },
        };
        inline for (update_configs) |cfg| {
            const array = &@field(self, cfg.name);
            inline for (array, 0..) |*model, i| {
                if (cfg.use_map) {
                    model.byteUpdate(cfg.ctxs[i], map_len);
                } else {
                    model.byteUpdate(cfg.ctxs[i], history_slice);
                }
            }
        }

        self.direct_bracket_model.byteUpdate(self.bracket.getBracketContext());

        inline for (&self.match_models) |*model| {
            self.longest_match = @max(self.longest_match, @as(u64, model.match_length) / 32);
        }

        // Update PPM Model
        self.ppm.update(self.history[0..self.history_pos], self.history_pos - 1, byte_val);
        self.ppm.predict(self.history[0..self.history_pos], self.history_pos);

        self.bracket.byteUpdate(byte_val);

        if (self.lstm) |*lstm| {
            var offset_idx: usize = 0;
            for (0..256) |j| {
                if (self.vocab[j]) {
                    self.byte_mixer_inputs[offset_idx] = self.ppm.tree[256 + j] * 2.0;
                    offset_idx += 1;
                }
            }
            lstm.setInput(self.byte_mixer_inputs);
            const output = lstm.perceive(self.byte_map[byte_val]);
            offset_idx = 0;
            for (0..256) |j| {
                if (self.vocab[j]) {
                    self.byte_mixer_probs[j] = output[offset_idx];
                    offset_idx += 1;
                } else {
                    self.byte_mixer_probs[j] = 0.0;
                }
            }
            for (0..256) |j| {
                self.byte_mixer_tree[256 + j] = self.byte_mixer_probs[j];
            }
            var idx: usize = 255;
            while (idx >= 1) : (idx -= 1) {
                self.byte_mixer_tree[idx] = self.byte_mixer_tree[2 * idx] + self.byte_mixer_tree[2 * idx + 1];
            }
        }
    }
};

inline fn getDecay(steps: usize) f32 {
    if (steps < 1000000) return 1.0;
    if (steps < 5000000) return 0.7;
    return 0.3;
}
