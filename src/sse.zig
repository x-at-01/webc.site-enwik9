const std = @import("std");

pub const SseModel = struct {
    table: []f32,
    num_bins: usize,
    table_size: usize,
    learning_rate: f32,

    pub fn init(allocator: std.mem.Allocator, table_size: usize, num_bins: usize, learning_rate: f32) !SseModel {
        const table = try allocator.alloc(f32, table_size * num_bins);
        for (0..table_size) |t| {
            for (0..num_bins) |b| {
                table[t * num_bins + b] = @as(f32, @floatFromInt(b)) / @as(f32, @floatFromInt(num_bins - 1));
            }
        }
        return SseModel{
            .table = table,
            .num_bins = num_bins,
            .table_size = table_size,
            .learning_rate = learning_rate,
        };
    }

    pub fn deinit(self: *SseModel, allocator: std.mem.Allocator) void {
        allocator.free(self.table);
    }

    pub inline fn predict(self: *const SseModel, context: u64, p: f32) f32 {
        const ctx_idx = (context % self.table_size) * self.num_bins;
        const p_scaled = p * @as(f32, @floatFromInt(self.num_bins - 1));
        const idx = @min(self.num_bins - 2, @as(usize, @intFromFloat(p_scaled)));
        const frac = p_scaled - @as(f32, @floatFromInt(idx));
        const p0 = self.table[ctx_idx + idx];
        const p1 = self.table[ctx_idx + idx + 1];
        return p0 * (1.0 - frac) + p1 * frac;
    }

    pub inline fn perceive(self: *SseModel, context: u64, p: f32, bit: u1) void {
        const ctx_idx = (context % self.table_size) * self.num_bins;
        const p_scaled = p * @as(f32, @floatFromInt(self.num_bins - 1));
        const idx = @min(self.num_bins - 2, @as(usize, @intFromFloat(p_scaled)));
        const frac = p_scaled - @as(f32, @floatFromInt(idx));
        const target = @as(f32, @floatFromInt(bit));
        const err0 = target - self.table[ctx_idx + idx];
        const err1 = target - self.table[ctx_idx + idx + 1];
        self.table[ctx_idx + idx] += err0 * self.learning_rate * (1.0 - frac);
        self.table[ctx_idx + idx + 1] += err1 * self.learning_rate * frac;
    }
};

const M_mx1mask0 = [256]u8{ 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 31, 32, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 39, 40, 40, 41, 41, 42, 42, 43, 43, 44, 44, 45, 45, 46, 46, 47, 47, 47, 47, 48, 48, 48, 48, 49, 49, 49, 49, 50, 50, 50, 50, 51, 51, 51, 51, 52, 52, 52, 52, 53, 53, 53, 53, 54, 54, 54, 54, 55, 55, 55, 55, 56, 56, 56, 56, 57, 57, 57, 57, 58, 58, 58, 58, 59, 59, 59, 59, 60, 60, 60, 60, 61, 61, 61, 61, 62, 62, 62, 62, 63, 63, 63, 63, 63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 65, 65, 65, 66, 66, 66, 66, 66, 66, 66, 66, 67, 67, 67, 67, 67, 67, 67, 67, 68, 68, 68, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 69, 69, 69, 70, 70, 70, 70, 70, 70, 70, 70, 71, 71, 71, 71, 71, 71, 71, 71, 72, 72, 72, 72, 72, 72, 72, 72, 73, 73, 73, 73, 73, 73, 73, 73, 74, 74, 74, 74, 74, 74, 74, 74, 75, 75, 75, 75, 75, 75, 75, 75, 76, 76, 76, 76, 76, 76, 76, 76, 77, 77, 77, 77, 77, 77, 77, 77, 78, 78, 78, 78, 78, 78, 78, 78 };
const M_sm7mask0 = [256]u8{ 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254 };

fn st(p: f64) f64 {
    return @log((1.0 - p) / p) * 1.4426950408889634;
}

fn sq(p: f64) f64 {
    return 1.0 / (1.0 + @exp(p / 1.4426950408889634));
}

fn st_i(p: u32) u32 {
    const SCALE = 32768.0;
    const hSCALE = 16384.0;
    const st_coef = (hSCALE - 1.0) / (@log(SCALE - 1.0) * 1.4426950408889634);
    const p_f = @as(f64, @floatFromInt(p));
    const val = st(p_f / SCALE) * st_coef + hSCALE;
    return @intFromFloat(val);
}

fn sq_i(p: u32) u32 {
    const SCALE = 32768.0;
    const hSCALE = 16384.0;
    const st_coef = (hSCALE - 1.0) / (@log(SCALE - 1.0) * 1.4426950408889634);
    const sq_coef = 1.0 / st_coef;
    const p_f = @as(f64, @floatFromInt(p));
    const val = sq((p_f - hSCALE) * sq_coef) * SCALE;
    return @intFromFloat(val);
}

inline fn Extrap(p1: i32, C: i32) i32 {
    const hSCALE = 16384;
    const mSCALE = 32767;
    var res = @as(i32, @intCast((@as(i64, p1 - hSCALE) * C) >> 13)) + hSCALE;
    if (res < 1) res = 1;
    if (res > mSCALE) res = mSCALE;
    return res;
}

pub const SSEi_updstr = struct {
    P: i32 = 0,
    sw: i32 = 0,
    sse_idx: usize = 0,
    ctx_idx: usize = 0,
};

pub const SSEi = struct {
    P: [7]u16,

    pub fn init(self: *SSEi, Wi: i32) void {
        const SCALE = 32768;
        const SCw = @divTrunc(SCALE - Wi, 6);
        const INC = @divTrunc(Wi, 2) + 8192;
        var p1: i32 = INC;
        for (0..7) |i| {
            self.P[i] = @intCast(p1);
            p1 += SCw;
        }
    }

    pub inline fn SSE_Pred(self: []SSEi, ctx_idx: usize, iP: i32, X: *SSEi_updstr) i32 {
        const SCALE = 32768;
        const mSCALE = SCALE - 1;
        const sse = &self[ctx_idx];

        const safe_iP = if (iP < 0) 0 else if (iP >= SCALE) mSCALE else iP;

        const sseFreq = @as(usize, @intCast(@divTrunc(6 * safe_iP, SCALE)));
        X.sw = (6 * safe_iP) & mSCALE;
        X.sse_idx = sseFreq;
        X.ctx_idx = ctx_idx;

        const f = @divTrunc((SCALE - X.sw) * @as(i32, sse.P[sseFreq]) + X.sw * @as(i32, sse.P[sseFreq + 1]), SCALE) - 8192;
        var res = f;
        if (res <= 0) res = 1;
        if (res >= SCALE) res = mSCALE;
        X.P = res;
        return res;
    }

    pub inline fn SSE_Update(self: []SSEi, bit: u1, wr0: i32, X: *SSEi_updstr) void {
        const SCALE = 32768;
        const mSCALE = SCALE - 1;
        X.P = @divTrunc(X.P * (SCALE - wr0), SCALE);
        if (bit == 0) X.P += wr0;

        const sse = &self[X.ctx_idx];
        const dC = @as(i32, sse.P[X.sse_idx]) - @as(i32, sse.P[X.sse_idx + 1]);
        const sw_dC = @divTrunc(X.sw * dC + mSCALE, SCALE);
        sse.P[X.sse_idx] = @intCast(X.P + sw_dC + 8192);
        sse.P[X.sse_idx + 1] = @intCast(X.P - (dC - sw_dC) + 8192);
    }
};

pub const ShelwienMixer = struct {
    w: i32 = 0,

    pub fn init(self: *ShelwienMixer, w0: i32) void {
        self.w = w0 + 16384;
    }

    inline fn rdiv64(x: i64, a: i64, d: u6) i32 {
        if (x >= 0) {
            return @intCast((x + a) >> d);
        } else {
            return @intCast((x - a) >> d);
        }
    }

    pub inline fn Mixup(self: *ShelwienMixer, s1: i32, s0: i32) i32 {
        const SCALE = 32768;
        const diff = @as(i64, self.w - 16384) * @as(i64, s0 - s1);
        const x_val = s1 + rdiv64(diff, 16384, 15);
        var res = x_val;
        if (res > 0) {
            if (res < SCALE) {
                // do nothing
            } else {
                res = SCALE - 1;
            }
        } else {
            res = 1;
        }
        return res;
    }

    pub inline fn Update(self: *ShelwienMixer, y: i32, p0: i32, p1: i32, wq: i32, pm: i32) void {
        const SCALE = 32768;
        const py = SCALE - (y << 15);
        const e = py - pm;
        const diff1 = @as(i64, e) * @as(i64, p0 - p1);
        var d = rdiv64(diff1, 16384, 15);
        const diff2 = @as(i64, d) * @as(i64, wq);
        d = rdiv64(diff2, 16384, 15);
        self.w = @intCast(@as(i64, self.w) + d);
    }
};

pub const ShelwienSSE = struct {
    t_st: [32768]u16 = undefined,
    t_sq: [32768]u16 = undefined,

    su6: SSEi_updstr = .{},
    su7: SSEi_updstr = .{},
    sm6x: usize = 0,
    mix1: usize = 0,
    sm7x: usize = 0,
    mix2: usize = 0,

    mix1_s0: i32 = 0,
    mix1_s1: i32 = 0,
    mix1_p: i32 = 0,
    mix2_s0: i32 = 0,
    mix2_s1: i32 = 0,
    mix2_p: i32 = 0,

    M_j: u32 = 1,
    M_pc: u8 = 0,
    M_ffl: u8 = 0,

    s6: []SSEi,
    s7: []SSEi,
    x1: []ShelwienMixer,
    x2: []ShelwienMixer,

    pub fn init(self: *ShelwienSSE, allocator: std.mem.Allocator) !void {
        const M_sm6x_Volume = 3 * 128 * 256 * 256;
        const M_sm7x_Volume = 3 * 32 * 256 * 255;
        const M_mix1_Volume = 4 * 256 * 8 * 79;
        const M_mix2_Volume = 3 * 2 * 256 * 256;

        self.s6 = try allocator.alloc(SSEi, M_sm6x_Volume);
        self.s7 = try allocator.alloc(SSEi, M_sm7x_Volume);
        self.x1 = try allocator.alloc(ShelwienMixer, M_mix1_Volume);
        self.x2 = try allocator.alloc(ShelwienMixer, M_mix2_Volume);

        self.su6 = SSEi_updstr{};
        self.su7 = SSEi_updstr{};
        self.sm6x = 0;
        self.mix1 = 0;
        self.sm7x = 0;
        self.mix2 = 0;
        self.mix1_s0 = 0;
        self.mix1_s1 = 0;
        self.mix1_p = 0;
        self.mix2_s0 = 0;
        self.mix2_s1 = 0;
        self.mix2_p = 0;
        self.M_j = 1;
        self.M_pc = 0;
        self.M_ffl = 0;

        self.initSTSQ();

        for (self.s6) |*s| s.init(0);
        for (self.s7) |*s| s.init(8192);
        for (self.x1) |*x| x.init(7648);
        for (self.x2) |*x| x.init(2560);
    }

    pub fn deinit(self: *ShelwienSSE, allocator: std.mem.Allocator) void {
        allocator.free(self.s6);
        allocator.free(self.s7);
        allocator.free(self.x1);
        allocator.free(self.x2);
    }

    fn initSTSQ(self: *ShelwienSSE) void {
        const SCALE = 32768;
        for (1..SCALE) |i| {
            self.t_sq[i] = @intCast(sq_i(@intCast(i)));
        }
        var x: usize = 0;
        self.t_st[x] = 0;
        for (1..SCALE) |i| {
            const s = @as(u16, @intCast(st_i(@intCast(i))));
            self.t_st[i] = s;
            if (s != self.t_st[x]) {
                const y = i - 1;
                self.t_sq[self.t_st[x]] = @intCast((x + y + 1) / 2);
                x = i;
            }
        }
    }

    pub fn estimate(self: *ShelwienSSE, p: i32) i32 {
        const M_f0C = 10240;
        const M_f1C = 7935;
        const M_f2C = 9592;
        const M_sm6C1 = 8092;
        const M_f3C = 8200;
        const M_f4C = 7677;
        const M_sm7C1 = 8202;

        const prq = @as(usize, @intCast(p >> 11));
        const j = self.M_j;
        const pc = self.M_pc;
        const ffl = self.M_ffl;

        // sm7x
        var sm7x_val: usize = 0;
        const prq_gt0_7: usize = @intFromBool(prq > 0);
        const prq_gt14_7: usize = @intFromBool(prq > 14);
        sm7x_val = sm7x_val * 3 + (prq_gt0_7 + prq_gt14_7);
        sm7x_val = (sm7x_val << 5) + (ffl & 31);
        sm7x_val = (sm7x_val << 8) + pc;
        sm7x_val = (sm7x_val * 255) + M_sm7mask0[j];
        self.sm7x = sm7x_val;

        // mix2
        var mix2_val: usize = 0;
        const prq_gt0_m2: usize = @intFromBool(prq > 0);
        const prq_gt14_m2: usize = @intFromBool(prq > 14);
        mix2_val = mix2_val * 3 + (prq_gt0_m2 + prq_gt14_m2);
        mix2_val = (mix2_val << 1) + (ffl & 1);
        mix2_val = (mix2_val << 8) + pc;
        mix2_val = (mix2_val * 256) + j;
        self.mix2 = mix2_val;

        // sm6x
        var sm6x_val: usize = 0;
        const prq_gt0_6: usize = @intFromBool(prq > 0);
        const prq_gt14_6: usize = @intFromBool(prq > 14);
        sm6x_val = sm6x_val * 3 + (prq_gt0_6 + prq_gt14_6);
        sm6x_val = (sm6x_val << 7) + (ffl & 127);
        sm6x_val = (sm6x_val << 8) + pc;
        sm6x_val = (sm6x_val * 256) + j;
        self.sm6x = sm6x_val;

        // mix1
        var mix1_val: usize = 0;
        const prq_gt0_m1: usize = @intFromBool(prq > 0);
        const prq_gt7_m1: usize = @intFromBool(prq > 7);
        const prq_gt14_m1: usize = @intFromBool(prq > 14);
        mix1_val = mix1_val * 4 + (prq_gt0_m1 + prq_gt7_m1 + prq_gt14_m1);
        mix1_val = (mix1_val << 8) + ffl;
        mix1_val = (mix1_val << 3) + ((pc >> 5) & 7);
        mix1_val = (mix1_val * 79) + M_mx1mask0[j];
        self.mix1 = mix1_val;

        const p0 = p;

        const t_st_p0 = self.t_st[@as(usize, @intCast(p0))];
        const ext_p0_f0 = Extrap(t_st_p0, M_f0C);
        const t_sq_ext = self.t_sq[@as(usize, @intCast(ext_p0_f0))];
        const p1 = SSEi.SSE_Pred(self.s6, self.sm6x, t_sq_ext, &self.su6);

        const s0 = Extrap(t_st_p0, M_f1C);
        const t_st_p1 = self.t_st[@as(usize, @intCast(p1))];
        const s1 = Extrap(t_st_p1, M_f2C);

        self.mix1_s0 = s0;
        self.mix1_s1 = s1;

        const s2 = self.x1[self.mix1].Mixup(self.mix1_s0, self.mix1_s1);
        const ext_s2 = Extrap(s2, M_sm6C1);
        self.mix1_p = @intCast(self.t_sq[@as(usize, @intCast(ext_s2))]);

        const ext_p0_f3 = Extrap(t_st_p0, M_f3C);
        const t_sq_ext_f3 = self.t_sq[@as(usize, @intCast(ext_p0_f3))];
        const p2 = SSEi.SSE_Pred(self.s7, self.sm7x, t_sq_ext_f3, &self.su7);

        const t_st_p2 = self.t_st[@as(usize, @intCast(p2))];
        const s4 = Extrap(t_st_p2, M_f4C);

        self.mix2_s0 = s2;
        self.mix2_s1 = s4;

        const s5 = self.x2[self.mix2].Mixup(self.mix2_s0, self.mix2_s1);
        const ext_s5 = Extrap(s5, M_sm7C1);
        self.mix2_p = @intCast(self.t_sq[@as(usize, @intCast(ext_s5))]);

        return self.mix2_p;
    }

    pub fn predict(self: *ShelwienSSE, input: f32) f32 {
        const safe_input = if (input < 0.0) 0.0 else if (input > 1.0) 1.0 else input;
        const discrete = 1 + @as(i32, @intFromFloat((1.0 - safe_input) * 32766.0));
        const est = self.estimate(discrete);
        const est_f = @as(f32, @floatFromInt(est)) - 1.0;
        return 1.0 - (est_f / 32766.0);
    }

    pub fn perceive(self: *ShelwienSSE, bit: u1) void {
        const M_sm6wrB = 106;
        const M_x1wr = 6202;
        const M_sm7wrB = 127;
        const M_x2wr = 8320;

        const bit_i: i32 = bit;

        SSEi.SSE_Update(self.s6, bit, M_sm6wrB, &self.su6);
        self.x1[self.mix1].Update(bit_i, self.mix1_s0, self.mix1_s1, M_x1wr, self.mix1_p);
        SSEi.SSE_Update(self.s7, bit, M_sm7wrB, &self.su7);
        self.x2[self.mix2].Update(bit_i, self.mix2_s0, self.mix2_s1, M_x2wr, self.mix2_p);

        self.M_j += self.M_j + bit;

        if (self.M_j >= 256) {
            const temp = @as(u32, self.M_ffl) * 2 + @intFromBool(self.M_pc >= 0x40);
            self.M_ffl = @intCast(temp & 255);
            self.M_pc = @intCast(self.M_j & 255);
            self.M_j = 1;
        }
    }
};
