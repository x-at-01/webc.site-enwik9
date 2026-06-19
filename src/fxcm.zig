const std = @import("std");
const math = std.math;
const tables = @import("tables.zig");
const wrt_2b = tables.wrt_2b;
const wrt_3b = tables.wrt_3b;
const wrt_4b = tables.wrt_4b;
const wrt_5b = tables.wrt_5b;
const dictionary = @import("dictionary.zig");

pub var prediction_index: usize = 0;
pub var model_predictions: [431]f32 = [_]f32{0.5} ** 431;

pub fn addPrediction(x_val: i32) void {
    if (prediction_index < 431) {
        model_predictions[prediction_index] = @as(f32, @floatFromInt(x_val)) * (1.0 / 4095.0);
        if (x.blpos == 0) {
            std.debug.print("[DEBUG addPrediction] index={d}, val={d}, float={d:.4}\n", .{prediction_index, x_val, model_predictions[prediction_index]});
        }
        prediction_index += 1;
    }
}

pub fn resetPredictions() void {
    prediction_index = 0;
}

const strt_table = init_strt: {
    @setEvalBranchQuota(100000);
    var arr = [_]i16{0} ** 4096;
    var i: usize = 0;
    while (i < 4096) : (i += 1) {
        var p = i;
        if (p == 0) p = 1;
        const f = @as(f64, @floatFromInt(p)) / 4096.0;
        const d = math.log(f64, 2.718281828459045, f / (1.0 - f)) * 256.0;
        var di = @as(i32, @intFromFloat(math.round(d)));
        if (di > 2047) di = 2047;
        if (di < -2047) di = -2047;
        arr[i] = @intCast(di);
    }
    break :init_strt arr;
};

pub inline fn stretch(p: u32) i32 {
    return strt_table[p];
}

const sqt_table = init_sqt: {
    @setEvalBranchQuota(100000);
    var arr = [_]u16{0} ** 4095;
    var d: i32 = -2047;
    while (d <= 2047) : (d += 1) {
        const exp_val = math.exp(-@as(f64, @floatFromInt(d)) / 256.0);
        const p_val = 1.0 / (1.0 + exp_val) * 4096.0;
        var pi = @as(u32, @intFromFloat(math.round(p_val)));
        if (pi > 4095) pi = 4095;
        if (pi < 1) pi = 1;
        arr[@intCast(d + 2047)] = @intCast(pi);
    }
    break :init_sqt arr;
};

pub inline fn squash(d: i32) i32 {
    if (d < -2047) return 1;
    if (d > 2047) return 4095;
    return sqt_table[@intCast(d + 2047)];
}

pub inline fn sc(p: i32) i32 {
    if (p > 0) return p >> 7;
    return (p + 127) >> 7;
}

pub inline fn clp(z: i32) i32 {
    if (z < -2047) return -2047;
    if (z > 2047) return 2047;
    return z;
}

pub inline fn cc_shift_bpos(cc: i32, bp: i32) i32 {
    const shift = @as(u5, @intCast(7 - bp));
    return (cc << 1) ^ (@as(i32, 256) >> shift);
}

pub const Inputs = struct {
    n: [560]i16 = [_]i16{0} ** 560,
    ncount: usize = 0,
    pub fn add(self: *Inputs, p: i32) void {
        self.n[self.ncount] = @intCast(p);
        self.ncount += 1;
        addPrediction(squash(p));
    }
};

pub const Inputs2 = struct {
    n: [32]i16 = [_]i16{0} ** 32,
    ncount: usize = 0,
    pub fn add(self: *Inputs2, p: i32) void {
        self.n[self.ncount] = @intCast(p);
        self.ncount += 1;
        addPrediction(squash(p));
    }
};

pub const BlockData = struct {
    y: u1 = 0,
    c0: i32 = 1,
    c4: u32 = 0,
    bpos: i32 = 0,
    blpos: u32 = 0,
    bposshift: u5 = 0,
    c0shift_bpos: i32 = 0,

    mxInputs1: Inputs = .{},
    mxInputs2: Inputs2 = .{},

    pub fn init(self: *BlockData) void {
        self.y = 0;
        self.c0 = 1;
        self.c4 = 0;
        self.bpos = 0;
        self.blpos = 0;
        self.bposshift = 0;
        self.c0shift_bpos = 0;
        self.mxInputs1.ncount = 512;
        self.mxInputs2.ncount = 16;
    }
};

pub var x: BlockData = .{};

pub var ilog: [256]u8 = [_]u8{0} ** 256;
pub fn initIlog() void {
    var val: u32 = 14155776;
    var i: usize = 2;
    while (i < 257) : (i += 1) {
        val += 774541002 / @as(u32, @intCast(i * 2 - 1));
        ilog[i - 1] = @intCast(val >> 24);
    }
}

pub const StateTable = struct {
    mdc: i32 = 0,
    b: [6]i32 = [_]i32{0} ** 6,
    ns: [1024]u8 = [_]u8{0} ** 1024,
    t: [64][64][2]u8 = [_][64][2]u8{[_][2]u8{.{ 0, 0 }} ** 64} ** 64,

    pub fn num_states(self: *const StateTable, xx: i32, yy: i32) i32 {
        if (xx < yy) return self.num_states(yy, xx);
        if (xx < 0 or yy < 0 or xx >= 64 or yy >= 64 or yy >= 5 or xx >= self.b[@intCast(yy)]) return 0;
        return 1 + @as(i32, @intFromBool(yy > 0 and xx + yy < self.b[5]));
    }

    pub fn discount(self: *const StateTable, xx: *i32) void {
        var yy: i32 = 0;
        if (xx.* > 2) {
            var i: i32 = 1;
            while (i < self.mdc) : (i += 1) {
                if (xx.* >= i) yy += 1;
            }
            xx.* = yy;
        }
    }

    pub fn next_state(self: *const StateTable, xx: *i32, yy: *i32, bit: i32) void {
        if (xx.* < yy.*) {
            var y_tmp = xx.*;
            var x_tmp = yy.*;
            self.next_state(&x_tmp, &y_tmp, 1 - bit);
            xx.* = y_tmp;
            yy.* = x_tmp;
        } else {
            if (bit != 0) {
                yy.* += 1;
                var x_val = xx.*;
                self.discount(&x_val);
                xx.* = x_val;
            } else {
                xx.* += 1;
                var y_val = yy.*;
                self.discount(&y_val);
                yy.* = y_val;
            }
            while (self.t[@intCast(xx.*)][@intCast(yy.*)][1] == 0) {
                if (yy.* < 2) {
                    xx.* -= 1;
                } else {
                    xx.* = @divTrunc(xx.* * (yy.* - 1) + @divTrunc(yy.*, 2), yy.*);
                    yy.* -= 1;
                }
            }
        }
    }

    pub fn generate(self: *StateTable) void {
        @memset(&self.ns, 0);
        @memset(std.mem.asBytes(&self.t), 0);
        var state: i32 = 0;
        var i: i32 = 0;
        while (i < 256) : (i += 1) {
            var yy: i32 = 0;
            while (yy <= i) : (yy += 1) {
                const xx = i - yy;
                const n = self.num_states(xx, yy);
                if (n != 0) {
                    self.t[@intCast(xx)][@intCast(yy)][0] = @intCast(state);
                    self.t[@intCast(xx)][@intCast(yy)][1] = @intCast(n);
                    state += n;
                }
            }
        }

        state = 0;
        var xi: i32 = 0;
        while (xi < 64) : (xi += 1) {
            var yy: i32 = 0;
            while (yy <= xi) : (yy += 1) {
                const xx = xi - yy;
                var k: i32 = 0;
                const limit = self.t[@intCast(xx)][@intCast(yy)][1];
                while (k < limit) : (k += 1) {
                    var x0 = xx;
                    var y0 = yy;
                    var x1 = xx;
                    var y1 = yy;
                    self.next_state(&x0, &y0, 0);
                    self.next_state(&x1, &y1, 1);
                    const ns0 = self.t[@intCast(x0)][@intCast(y0)][0];
                    const ns1 = self.t[@intCast(x1)][@intCast(y1)][0] + if (self.t[@intCast(x1)][@intCast(y1)][1] > 1) @as(u8, 1) else @as(u8, 0);
                    self.ns[@intCast(state * 4)] = ns0;
                    self.ns[@intCast(state * 4 + 1)] = ns1;
                    self.ns[@intCast(state * 4 + 2)] = @intCast(xx);
                    self.ns[@intCast(state * 4 + 3)] = @intCast(yy);

                    if (state > 0xff or self.t[@intCast(xx)][@intCast(yy)][1] == 0 or self.t[@intCast(x0)][@intCast(y0)][1] == 0 or self.t[@intCast(x1)][@intCast(y1)][1] == 0) return;
                    state += 1;
                    if (state > 0xff) return;
                }
            }
        }
    }

    pub fn init_table(self: *StateTable, s0: i32, s1: i32, s2: i32, s3: i32, s4: i32, s5: i32, s6: i32, table: []u8) void {
        self.b[0] = s0;
        self.b[1] = s1;
        self.b[2] = s2;
        self.b[3] = s3;
        self.b[4] = s4;
        self.b[5] = s5;
        self.mdc = s6;
        self.generate();
        @memcpy(table[0..1024], self.ns[0..1024]);
    }
};

pub var STA1: [256][4]u8 = undefined;
pub var STA2: [256][4]u8 = undefined;
pub var STA4: [256][4]u8 = undefined;
pub var STA5: [256][4]u8 = undefined;
pub var STA6: [256][4]u8 = undefined;
pub var STA7: [256][4]u8 = undefined;

pub const codeword2sym = init_codeword2sym: {
    var arr = [_]i32{0} ** 256;
    var c: usize = 128;
    while (c < 256) : (c += 1) {
        arr[c] = @intCast(c - 128);
    }
    break :init_codeword2sym arr;
};

pub const dict1size: i32 = 80;
pub const dict2size: i32 = 32;
pub const dict12size: i32 = 80 * 32;

pub fn decodeCodeWord(cw: u32) i32 {
    var i: i32 = 0;
    var c: usize = cw & 255;
    if (codeword2sym[c] < dict1size) {
        i = codeword2sym[c];
        return i;
    }

    i = dict1size * (codeword2sym[c] - dict1size);
    c = (cw >> 8) & 255;

    if (codeword2sym[c] < dict1size) {
        i += codeword2sym[c];
        return i + dict1size;
    }

    i = (i - dict12size) * dict2size;
    i += dict1size * (codeword2sym[c] - dict1size);

    c = (cw >> 16) & 255;
    i += codeword2sym[c];
    return i + 80 * 49;
}

pub var so: []const u8 = "";
pub var lastCW: i32 = 0;

pub fn decodeWord(c: u32) void {
    const j = decodeCodeWord(c);
    lastCW = j;
    if (j >= 0 and j < dictionary.word_count) {
        so = dictionary.getWord(@intCast(j));
    } else {
        so = "";
    }
}

pub fn dot_product(t_slice: []const i16, w: []const i16, n: usize) i32 {
    var sum: i32 = 0;
    var i: usize = 0;
    while (i < n) : (i += 2) {
        const p0 = @as(i32, t_slice[i]) * w[i];
        const p1 = @as(i32, t_slice[i + 1]) * w[i + 1];
        sum = sum +% ((p0 +% p1) >> 8);
    }
    return sum;
}

pub fn train(t_slice: []const i16, w: []i16, n: usize, e: i16) void {
    if (e == 0) return;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var t2_val = @as(i32, t_slice[i]) * 2;
        if (t2_val > 32767) {
            t2_val = 32767;
        } else if (t2_val < -32768) {
            t2_val = -32768;
        }
        const m = (t2_val * e) >> 16;
        var m2 = m + 1;
        if (m2 > 32767) {
            m2 = 32767;
        } else if (m2 < -32768) {
            m2 = -32768;
        }
        const shift_m2 = m2 >> 1;
        var new_w = shift_m2 + w[i];
        if (new_w > 32767) {
            new_w = 32767;
        } else if (new_w < -32768) {
            new_w = -32768;
        }
        w[i] = @intCast(new_w);
    }
}

pub const Mixer1 = struct {
    N: usize = 0,
    M: usize = 0,
    tx: []i16 = &.{},
    wx: []i16 = &.{},
    wx_ptr: []i16 = &.{},
    cxt: usize = 0,
    pr: i32 = 2048,
    shift1: i32 = 0,
    elim: i32 = 0,
    uperr: i32 = 0,
    err: i32 = 0,

    pub fn deinit(self: *Mixer1, allocator: std.mem.Allocator) void {
        allocator.free(self.wx_ptr);
    }

    pub fn init_mixer(self: *Mixer1, m: usize, s: u32, e: u32, ue: u32) void {
        self.M = m;
        self.cxt = 0;
        self.shift1 = @intCast(s);
        self.elim = @intCast(e);
        self.uperr = @intCast(ue);
        self.err = 0;
        self.pr = 2048;
    }

    pub fn setTxWx(self: *Mixer1, allocator: std.mem.Allocator, n: usize, mn: []i16) !void {
        self.N = n;
        const size_val = self.N * self.M + 32;
        self.wx_ptr = try allocator.alloc(i16, size_val);
        @memset(self.wx_ptr, 129);
        self.wx = self.wx_ptr;
        self.tx = mn;
    }

    pub fn update(self: *Mixer1, y: u1) void {
        self.err = @divTrunc(((@as(i32, y) << 12) - self.pr) * self.uperr, 4);
        if (self.err > 32767) self.err = 32767;
        if (self.err < -32768) self.err = -32768;
        if (self.err >= -self.elim and self.err <= self.elim) self.err = 0;
        train(self.tx[0..self.N], self.wx[self.cxt * self.N .. (self.cxt + 1) * self.N], self.N, @intCast(self.err));
    }

    pub fn p(self: *Mixer1) i32 {
        const dp = dot_product(self.tx[0..self.N], self.wx[self.cxt * self.N .. (self.cxt + 1) * self.N], self.N);
        const term = @divTrunc(dp * self.shift1, 2048);
        self.pr = squash(term);
        return self.pr;
    }

    pub fn p1(self: *Mixer1) i32 {
        const dp = dot_product(self.tx[0..self.N], self.wx[self.cxt * self.N .. (self.cxt + 1) * self.N], self.N);
        var term = @divTrunc(dp * self.shift1, 2048);
        if (term < -2047) {
            term = -2047;
        } else if (term > 2047) {
            term = 2047;
        }
        self.pr = squash(term);
        return term;
    }
};

pub const StateMap = struct {
    t: []u32 = &.{},
    cxt: usize = 0,
    pr: u32 = 2048,
    nn: []const u8 = &.{},

    pub fn deinit(self: *StateMap, allocator: std.mem.Allocator) void {
        allocator.free(self.t);
    }

    pub fn next_val(self: *const StateMap, i: usize, y: usize) u8 {
        return self.nn[y + i * 4];
    }

    pub fn init_map(self: *StateMap, allocator: std.mem.Allocator, n: usize, nn1: []const u8) !void {
        self.nn = nn1;
        self.t = try allocator.alloc(u32, n);
        self.cxt = 0;
        self.pr = 2048;
        for (self.t, 0..) |*val, i| {
            const n0 = @as(u32, self.next_val(i, 2)) * 3 + 1;
            const n1 = @as(u32, self.next_val(i, 3)) * 3 + 1;
            val.* = ((n1 << 20) / (n0 + n1)) << 12;
        }
    }

    pub fn update(self: *StateMap, y: u1) void {
        const p0 = self.t[self.cxt];
        const pr1 = p0 >> 13;
        self.t[self.cxt] = p0 +% (@as(u32, y) << 19) -% pr1;
    }

    pub fn set_cxt(self: *StateMap, c: usize, y: u1) void {
        self.update(y);
        self.cxt = c;
        self.pr = self.t[c] >> 20;
    }
};

pub const dt_table = init_dt: {
    @setEvalBranchQuota(100000);
    var arr = [_]i32{0} ** 1024;
    var o: i32 = 2;
    var i: usize = 0;
    while (i < 1024) : (i += 1) {
        arr[i] = @divTrunc(4096, o);
        o += 1;
    }
    arr[1023] = 1;
    break :init_dt arr;
};

pub const StateMap1 = struct {
    t: []u32 = &.{},
    cxt: usize = 0,
    pr: u32 = 2048,
    mask: usize = 0,
    limit: u32 = 0,

    pub fn deinit(self: *StateMap1, allocator: std.mem.Allocator) void {
        allocator.free(self.t);
    }

    pub fn init_map(self: *StateMap1, allocator: std.mem.Allocator, n: usize, lim: u32) !void {
        self.t = try allocator.alloc(u32, n);
        self.cxt = 0;
        self.pr = 2048;
        self.mask = n - 1;
        self.limit = lim;
        @memset(self.t, 1 << 31);
    }

    pub fn update(self: *StateMap1, y: u1) void {
        const p0 = self.t[self.cxt];
        const n = p0 & 1023;
        const pr1 = p0 >> 12;
        var new_p0 = p0 +% @as(u32, @intFromBool(n < self.limit));
        const diff = (@as(i32, y) << 20) -% @as(i32, @intCast(pr1));
        const dt_val = dt_table[n];
        const term = (diff *% dt_val +% 512) & ~@as(i32, 1023);
        new_p0 = new_p0 +% @as(u32, @bitCast(term));
        self.t[self.cxt] = new_p0;
    }

    pub fn set_cxt(self: *StateMap1, c: usize, y: u1) void {
        self.update(y);
        self.cxt = c & self.mask;
        self.pr = self.t[self.cxt] >> 20;
    }
};

pub const RunContextMap = struct {
    t: []u8 = &.{},
    cp_offset: usize = 0,
    rc: [512]i16 = [_]i16{0} ** 512,
    n: u32 = 0,

    pub fn deinit(self: *RunContextMap, allocator: std.mem.Allocator) void {
        allocator.free(self.t);
    }

    pub fn init_map(self: *RunContextMap, allocator: std.mem.Allocator, m: usize, rcm_ml: i32) !void {
        self.t = try allocator.alloc(u8, m);
        @memset(self.t, 0);
        self.n = @intCast(m / 4 - 1);
        self.cp_offset = 1;
        var r: usize = 0;
        while (r < 256) : (r += 1) {
            var c = @as(i32, ilog[r]) * 8;
            if ((r & 1) == 0) {
                c = @divTrunc(c * rcm_ml, 4);
            }
            self.rc[r + 256] = @intCast(clp(c));
            self.rc[r] = @intCast(clp(-c));
        }
    }

    pub fn find(self: *RunContextMap, i_in: u32) usize {
        var i = i_in;
        const chk = @as(u16, @intCast(((i >> 16) ^ i) & 0xffff));
        i = (i *% 4) & self.n;
        var j: usize = 0;
        var p_idx: usize = 0;
        while (j < 4) : (j += 1) {
            p_idx = (i + j) * 4;
            const p_chk = std.mem.readInt(u16, self.t[p_idx..][0..2], .little);
            if (self.t[p_idx + 2] == 0) {
                std.mem.writeInt(u16, self.t[p_idx..][0..2], chk, .little);
                break;
            }
            if (p_chk == chk) break;
        }
        if (j == 0) return p_idx + 1;
        var tmp = [_]u8{0} ** 4;
        if (j == 4) {
            j -= 1;
            std.mem.writeInt(u16, tmp[0..2], chk, .little);
            if (self.t[(i + j) * 4 + 2] > self.t[(i + j - 1) * 4 + 2]) {
                j -= 1;
            }
        } else {
            @memcpy(tmp[0..4], self.t[p_idx .. p_idx + 4]);
        }
        const src_start = (i) * 4;
        const dst_start = (i + 1) * 4;
        const len = j * 4;
        std.mem.copyBackwards(u8, self.t[dst_start .. dst_start + len], self.t[src_start .. src_start + len]);
        @memcpy(self.t[src_start .. src_start + 4], tmp[0..4]);
        return src_start + 1;
    }

    pub fn set_cxt(self: *RunContextMap, cx: u32, c1_val: u8) void {
        const cp_count = self.t[self.cp_offset];
        const cp_val = self.t[self.cp_offset + 1];
        if (cp_count == 0) {
            self.t[self.cp_offset] = 2;
            self.t[self.cp_offset + 1] = c1_val;
        } else if (cp_val != c1_val) {
            self.t[self.cp_offset] = 1;
            self.t[self.cp_offset + 1] = c1_val;
        } else if (cp_count < 254) {
            self.t[self.cp_offset] = cp_count + 2;
        }
        self.cp_offset = self.find(cx);
    }

    pub fn get_pred(self: *const RunContextMap, c0shift_bpos: i32, bposshift: u5) i16 {
        const cp_count = self.t[self.cp_offset];
        const cp_val = self.t[self.cp_offset + 1];
        const b = c0shift_bpos ^ (@as(i32, cp_val) >> bposshift);
        if (b <= 1) {
            return self.rc[@intCast(b * 256 + cp_count)];
        } else {
            return 0;
        }
    }
};

pub const SmallStationaryContextMap = struct {
    data: []u16 = &.{},
    context: usize = 0,
    mask: usize = 0,
    stride: usize = 0,
    bCount: usize = 0,
    bTotal: usize = 0,
    b: usize = 0,
    cp_offset: usize = 0,

    pub fn deinit(self: *SmallStationaryContextMap, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
    }

    pub fn init_map(self: *SmallStationaryContextMap, allocator: std.mem.Allocator, bits_of_context: usize, input_bits: usize) !void {
        self.mask = (@as(usize, 1) << @as(u6, @intCast(bits_of_context))) - 1;
        self.stride = (@as(usize, 1) << @as(u6, @intCast(input_bits))) - 1;
        self.bCount = 0;
        self.bTotal = input_bits;
        self.b = 0;
        const n = (@as(usize, 1) << @as(u6, @intCast(bits_of_context))) * ((@as(usize, 1) << @as(u6, @intCast(input_bits))) - 1);
        self.data = try allocator.alloc(u16, n);
        @memset(self.data, 0x7fff);
        self.cp_offset = 0;
    }

    pub fn set_cxt(self: *SmallStationaryContextMap, ctx_val: u32) void {
        self.context = (ctx_val & self.mask) * self.stride;
        self.bCount = 0;
        self.b = 0;
    }

    pub fn mix(self: *SmallStationaryContextMap, inputs: *Inputs, r: i32, y: u1) void {
        const rate_val = @as(u5, @intCast(r + 7));
        const old_val = self.data[self.cp_offset];
        const diff = (@as(i32, y) << 16) - @as(i32, old_val);
        const term = (diff + (@as(i32, 1) << (rate_val - 1))) >> rate_val;
        self.data[self.cp_offset] = @intCast(@as(i32, old_val) +% term);

        self.b += @intFromBool(y != 0 and self.b > 0);
        self.cp_offset = self.context + self.b;
        const pred = @as(i32, self.data[self.cp_offset]) >> 4;

        inputs.add(@divTrunc(stretch(@intCast(pred)), 4));
        inputs.add(@divTrunc(pred - 2048, 8));
        prediction_index -= 1;

        self.bCount += 1;
        self.b = self.b *% 2 +% 1;
        if (self.bCount == self.bTotal) {
            self.bCount = 0;
            self.b = 0;
        }
    }
};

pub const E = struct {
    chk: [7]u16 = [_]u16{0} ** 7,
    last: u8 = 0,
    bh: [7][7]u8 = [_][7]u8{[_]u8{0} ** 7} ** 7,

    pub fn get(self: *E, ch: u16, keep: i32) *u8 {
        if (self.chk[self.last & 15] == ch) {
            return &self.bh[self.last & 15][0];
        }
        var b: i32 = 0xffff;
        var bi: usize = 0;
        var i: usize = 0;
        while (i < 7) : (i += 1) {
            if (self.chk[i] == ch) {
                self.last = @intCast(((@as(u32, self.last) << 4) & 0xff) | i);
                return &self.bh[i][0];
            }
            const pri = self.bh[i][0];
            if (pri < b and (self.last & 15) != i and (self.last >> 4) != i) {
                b = pri;
                bi = i;
            }
        }
        self.last = @intCast(((@as(u32, self.last) << 4) & 0xff) | bi | @as(u32, @intCast(keep)));
        self.chk[bi] = ch;
        @memset(&self.bh[bi], 0);
        return &self.bh[bi][0];
    }
};

pub const E1 = struct {
    chk: [3]u16 = [_]u16{0} ** 3,
    last: u8 = 0,
    bh: [3][7]u8 = [_][7]u8{[_]u8{0} ** 7} ** 3,
    pad: [4]u8 = [_]u8{0} ** 4,

    pub fn get(self: *E1, ch: u16, keep: i32) *u8 {
        if (self.chk[self.last & 15] == ch) {
            return &self.bh[self.last & 15][0];
        }
        var b: i32 = 0xffff;
        var bi: usize = 0;
        var i: usize = 0;
        while (i < 3) : (i += 1) {
            if (self.chk[i] == ch) {
                self.last = @intCast(((@as(u32, self.last) << 4) & 0xff) | i);
                return &self.bh[i][0];
            }
            const pri = self.bh[i][0];
            if (pri < b and (self.last & 15) != i and (self.last >> 4) != i) {
                b = pri;
                bi = i;
            }
        }
        self.last = @intCast(((@as(u32, self.last) << 4) & 0xff) | bi | @as(u32, @intCast(keep)));
        self.chk[bi] = ch;
        @memset(&self.bh[bi], 0);
        return &self.bh[bi][0];
    }
};

pub const E2 = struct {
    chk: [14]u16 = [_]u16{0} ** 14,
    last: u8 = 0,
    bh: [14][7]u8 = [_][7]u8{[_]u8{0} ** 7} ** 14,
    pad: [1]u8 = [_]u8{0} ** 1,

    pub fn get(self: *E2, ch: u16, keep: i32) *u8 {
        if (self.chk[self.last & 15] == ch) {
            return &self.bh[self.last & 15][0];
        }
        var b: i32 = 0xffff;
        var bi: usize = 0;
        var i: usize = 0;
        while (i < 14) : (i += 1) {
            if (self.chk[i] == ch) {
                self.last = @intCast(((@as(u32, self.last) << 4) & 0xff) | i);
                return &self.bh[i][0];
            }
            const pri = self.bh[i][0];
            if (pri < b and (self.last & 15) != i and (self.last >> 4) != i) {
                b = pri;
                bi = i;
            }
        }
        self.last = @intCast(((@as(u32, self.last) << 4) & 0xff) | bi | @as(u32, @intCast(keep)));
        self.chk[bi] = ch;
        @memset(&self.bh[bi], 0);
        return &self.bh[bi][0];
    }
};

pub inline fn getStateByteLocation(bpos: i32, c0: i32) u32 {
    const smask = (@as(u32, 0x31031010) >> @as(u5, @intCast(bpos << 2))) & 0x0F;
    const pis = smask + (@as(u32, @intCast(c0)) & smask);
    return pis;
}

pub fn GenericContextMap(comptime E_type: type, comptime shift_val: u5) type {
    return struct {
        const Self = @This();
        C: u8 = 0,
        cp: [8]?[*]u8 = [_]?[*]u8{null} ** 8,
        cp0: [8]?[*]u8 = [_]?[*]u8{null} ** 8,
        runp: [8]?[*]u8 = [_]?[*]u8{null} ** 8,

        cxt: [8]u32 = [_]u32{0} ** 8,
        sm: []StateMap = &.{},
        cn: usize = 0,
        result: i32 = 0,
        rc1: [512]i16 = [_]i16{0} ** 512,
        st1: [4096]i16 = [_]i16{0} ** 4096,
        st2: []const i16 = &.{},
        st32: [256]i16 = [_]i16{0} ** 256,
        st8: [256]i16 = [_]i16{0} ** 256,
        cms: i32 = 0,
        cms3: i32 = 0,
        cms4: i32 = 0,
        kep: i32 = 0,
        nn: []const u8 = &.{},
        t: []E_type = &.{},
        t_ptr: []E_type = &.{},
        tmask: u32 = 0,
        skip2: i32 = 0,
        cxtMask: u16 = 0,

        pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
            for (self.sm) |*s| {
                s.deinit(allocator);
            }
            allocator.free(self.sm);
            allocator.free(self.t_ptr);
        }

        pub fn next_val(self: *const Self, i: usize, y: usize) u8 {
            return self.nn[y + i * 4];
        }

        pub fn pre(self: *const Self, state: usize) u32 {
            const n0 = @as(u32, self.next_val(state, 2)) * 3 + 1;
            const n1 = @as(u32, self.next_val(state, 3)) * 3 + 1;
            return (n1 << 12) / (n0 + n1);
        }

        pub fn init_map(self: *Self, allocator: std.mem.Allocator, m_in: u32, c: i32, s3: i32, nn1: []const u8, cs4: i32, k: i32, u: i32, st: []const i16) !void {
            self.C = @intCast(c & 255);
            var m = m_in;
            if (E_type == E2) {
                m = m_in * 2;
            }
            self.tmask = (m >> shift_val) - 1;
            self.cn = 0;
            self.cxtMask = ((@as(u16, 1) << @as(u4, @intCast(self.C))) - 1) * 2;
            self.result = 0;
            self.kep = k;
            self.nn = nn1;

            const num_elements = (m >> shift_val) + 64 * (if (E_type == E2) @as(u32, 2) else @as(u32, 1));
            self.t_ptr = try allocator.alloc(E_type, num_elements);
            @memset(self.t_ptr, std.mem.zeroes(E_type));
            self.t = self.t_ptr;

            const cmul = @as(i32, @intCast((c >> 8) & 255));
            self.cms = (c >> 16) & 255;
            self.cms4 = cs4;
            self.cms3 = s3;
            self.skip2 = u;

            self.sm = try allocator.alloc(StateMap, self.C);
            for (self.sm) |*s| {
                try s.init_map(allocator, 256, nn1);
            }

            for (self.cp0[0..self.C]) |*p| p.* = self.t[0].bh[0][0..].ptr;
            for (self.cp[0..self.C]) |*p| p.* = self.t[0].bh[0][0..].ptr;
            for (self.runp[0..self.C]) |*p| p.* = self.t[0].bh[0][3..].ptr;

            var rc: usize = 0;
            while (rc < 256) : (rc += 1) {
                var c_val = @as(i32, ilog[rc]);
                c_val = c_val << @as(u3, @intCast(2 + (~rc & 1)));
                if ((rc & 1) == 0) {
                    c_val = @divTrunc(c_val * cmul, 4);
                }
                self.rc1[rc + 256] = @intCast(clp(c_val));
                self.rc1[rc] = @intCast(clp(-c_val));
            }

            self.st2 = st;
            var i: usize = 0;
            while (i < 4096) : (i += 1) {
                self.st1[i] = @intCast(clp(sc(self.cms * stretch(@intCast(i)))));
            }

            var s: usize = 0;
            while (s < 256) : (s += 1) {
                const n0 = if (self.next_val(s, 2) == 0) @as(i32, -1) else @as(i32, 0);
                const n1 = if (self.next_val(s, 3) == 0) @as(i32, -1) else @as(i32, 0);
                var r_flag = false;
                var sp0: i32 = 0;
                if ((n1 - n0) == 1) {
                    sp0 = 0;
                    r_flag = true;
                }
                if ((n1 - n0) == -1) {
                    sp0 = 4095;
                    r_flag = true;
                }
                if (r_flag) {
                    self.st8[s] = @intCast(clp(sc(self.cms4 * (@as(i32, @intCast(self.pre(s))) - sp0))));
                    self.st32[s] = @intCast(clp(sc(self.cms3 * stretch(self.pre(s)))));
                    if (s < 8) self.st32[s] = 0;
                } else {
                    self.st8[s] = 0;
                    self.st32[s] = 0;
                }
            }
        }

        pub fn set(self: *Self, cx: u32) void {
            const i = self.cn;
            self.cn += 1;
            const permuted = (cx *% 987654323) +% @as(u32, @intCast(i));
            const rotated = (permuted << 16) | (permuted >> 16);
            self.cxt[i] = (rotated *% 123456791) +% @as(u32, @intCast(i));
            self.cxtMask = self.cxtMask *% 2;
        }

        pub fn sets(self: *Self) void {
            self.cn += 1;
            self.cxtMask = (self.cxtMask +% 1) *% 2;
        }

        fn mix3(self: *Self, inputs: *Inputs, s: u8, sm: *StateMap, y: u1) i32 {
            if (s == 0) {
                inputs.add(0);
                if (self.skip2 == 1) inputs.add(0);
                inputs.add(0);
                inputs.add(0);
                inputs.add(32 * 2);
                prediction_index -= 1;
                return 0;
            } else {
                sm.set_cxt(s, y);
                const p1 = sm.pr;
                inputs.add(self.st1[p1]);
                if (self.skip2 == 1) inputs.add(self.st2[p1]);
                inputs.add(self.st8[s]);
                inputs.add(self.st32[s]);
                inputs.add(0);
                prediction_index -= 1;
                return 1;
            }
        }

        fn mix4(self: *Self, inputs: *Inputs) void {
            inputs.add(0);
            if (self.skip2 == 1) inputs.add(0);
            inputs.add(0);
            inputs.add(0);
            inputs.add(32 * 2);
            prediction_index -= 1;
            inputs.add(0);
        }

        pub fn mix(self: *Self, inputs: *Inputs, cc: i32, bp: i32, c1_val: u8, c4: u8, y: u1) i32 {
            _ = c4;
            self.result = 0;
            var i: usize = 0;
            while (i < self.cn) : (i += 1) {
                if (((self.cxtMask >> @as(u4, @intCast(self.cn - i))) & 1) != 0) {
                    self.mix4(inputs);
                } else {
                    if (self.cp[i]) |cp_ptr| {
                        cp_ptr[0] = self.next_val(cp_ptr[0], y);
                    }

                    var s: u8 = 0;
                    const runp_ptr = self.runp[i].?;
                    if (bp > 1 and runp_ptr[0] == 0) {
                        self.cp[i] = null;
                    } else {
                        const chksum = @as(u16, @intCast(((self.cxt[i] >> 16) ^ @as(u32, @intCast(i))) & 0xffff));
                        if (bp != 0) {
                            if (bp == 2 or bp == 5) {
                                const idx = (self.cxt[i] +% @as(u32, @intCast(cc))) & self.tmask;
                                self.cp0[i] = @ptrCast(self.t[idx].get(chksum, self.kep));
                                self.cp[i] = self.cp0[i];
                            } else {
                                self.cp[i] = self.cp0[i].? + getStateByteLocation(bp, cc);
                            }
                        } else {
                            const idx = (self.cxt[i] +% @as(u32, @intCast(cc))) & self.tmask;
                            self.cp0[i] = @ptrCast(self.t[idx].get(chksum, self.kep));
                            self.cp[i] = self.cp0[i];

                            if (self.cp0[i].?[3] == 2) {
                                const c_val = @as(u32, self.cp0[i].?[4]) + 256;
                                const idx1 = (self.cxt[i] +% (c_val >> 6)) & self.tmask;
                                var p: [*]u8 = @ptrCast(self.t[idx1].get(chksum, self.kep));
                                p[0] = @intCast(1 + ((c_val >> 5) & 1));
                                p[@intCast(1 + ((c_val >> 5) & 1))] = @intCast(1 + ((c_val >> 4) & 1));
                                p[@intCast(3 + ((c_val >> 4) & 3))] = @intCast(1 + ((c_val >> 3) & 1));

                                const idx2 = (self.cxt[i] +% (c_val >> 3)) & self.tmask;
                                p = @ptrCast(self.t[idx2].get(chksum, self.kep));
                                p[0] = @intCast(1 + ((c_val >> 2) & 1));
                                p[@intCast(1 + ((c_val >> 2) & 1))] = @intCast(1 + ((c_val >> 1) & 1));
                                p[@intCast(3 + ((c_val >> 1) & 3))] = @intCast(1 + (c_val & 1));

                                self.cp0[i].?[6] = 0;
                            }

                            if (self.runp[i].?[0] == 0) {
                                self.runp[i].?[0] = 2;
                                self.runp[i].?[1] = c1_val;
                            } else if (self.runp[i].?[1] != c1_val) {
                                self.runp[i].?[0] = 1;
                                self.runp[i].?[1] = c1_val;
                            } else if (self.runp[i].?[0] < 254) {
                                self.runp[i].?[0] += 2;
                            }
                            self.runp[i] = self.cp0[i].? + 3;
                        }
                        s = self.cp[i].?[0];
                    }

                    self.result += self.mix3(inputs, s, &self.sm[i], y);

                    const b = cc_shift_bpos(cc, bp) ^ @as(i32, self.runp[i].?[1] >> @as(u3, @intCast(7 - bp)));
                    if (b <= 1) {
                        inputs.add(self.rc1[@intCast(self.runp[i].?[0] + b * 256)]);
                    } else {
                        inputs.add(0);
                    }
                }
            }
            if (bp == 7) {
                self.cn = 0;
                self.cxtMask = 0;
            }
            return self.result;
        }
    };
}

pub const ContextMap = GenericContextMap(E, 6);
pub const ContextMap1 = GenericContextMap(E1, 6);
pub const ContextMap2 = GenericContextMap(E2, 7);

pub const Word = struct {
    letters: [64]u8 = [_]u8{0} ** 64,
    start: u8 = 0,
    end: u8 = 0,
    hash: u32 = 0,
    type: u32 = 0,
    suffix: u32 = 0,
    prefix: u32 = 0,

    pub fn init() Word {
        return .{};
    }

    pub fn eqlStr(self: *const Word, s: []const u8) bool {
        const len = s.len;
        const self_len = self.length();
        if (self_len != len) return false;
        var i: usize = 0;
        while (i < len) : (i += 1) {
            if (self.letters[self.start + i] != s[i]) return false;
        }
        return true;
    }

    pub fn append(self: *Word, c: u8) void {
        if (c > 0 and self.end < 63) {
            if (self.letters[self.end] > 0) {
                self.end += 1;
            }
            self.letters[self.end] = c;
        }
    }

    pub fn at(self: *const Word, i: u8) u8 {
        if (self.end - self.start >= i) {
            return self.letters[self.start + i];
        }
        return 0;
    }

    pub fn fromEnd(self: *const Word, i: u8) u8 {
        if (self.end - self.start >= i) {
            return self.letters[self.end - i];
        }
        return 0;
    }

    pub fn trimStartingApostrophe(self: *Word) bool {
        var result = false;
        var cnt: i32 = 0;
        const APOSTROPHE = '\'';
        while (self.start != self.end and self.at(0) == APOSTROPHE) {
            result = true;
            self.start += 1;
            cnt += 1;
        }
        while (self.start != self.end and self.fromEnd(0) == APOSTROPHE) {
            if (cnt == 0) break;
            self.end -= 1;
            cnt -= 1;
        }
        if (self.fromEnd(0) == '-') {
            self.end -= 1;
        }
        return result;
    }

    pub fn markYsAsConsonants(self: *Word) void {
        if (self.at(0) == 'y') {
            self.letters[self.start] = 'Y';
        }
        var i: usize = self.start + 1;
        while (i <= self.end) : (i += 1) {
            if (isVowel(self.letters[i - 1]) and self.letters[i] == 'y') {
                self.letters[i] = 'Y';
            }
        }
    }

    pub fn getRegion1(self: *const Word) u32 {
        var i: usize = self.start + 1;
        while (i <= self.end) : (i += 1) {
            if (!isVowel(self.letters[i - 1]) and isVowel(self.letters[i])) {
                return @intCast(i + 1 - self.start);
            }
        }
        return self.length();
    }

    pub fn getRegion(self: *const Word, r1: u32) u32 {
        var i: usize = self.start + r1 + 1;
        while (i <= self.end) : (i += 1) {
            if (!isVowel(self.letters[i - 1]) and isVowel(self.letters[i])) {
                return @intCast(i + 1 - self.start);
            }
        }
        return self.length();
    }

    pub fn suffixInRn(self: *const Word, rn: u32, suffix: []const u8) bool {
        const len = suffix.len;
        const self_len = self.length();
        if (self_len <= len) return false;
        const offset = @as(u32, self.end) + 1 - @as(u32, @intCast(len)) - self.start;
        if (offset < rn) return false;
        return std.mem.eql(u8, self.letters[self.end + 1 - len .. self.end + 1], suffix);
    }

    pub fn hasVowels(self: *const Word) bool {
        var i: usize = self.start;
        while (i <= self.end) : (i += 1) {
            if (isVowel(self.letters[i])) return true;
        }
        return false;
    }

    pub fn endsInShortSyllable(self: *const Word) bool {
        const len = self.length();
        if (len < 2) return false;
        const char0 = self.fromEnd(0);
        const char1 = self.fromEnd(1);
        const char2 = self.fromEnd(2);
        if (isVowel(char0) and !isVowel(char1) and isVowel(char2) and char0 != 'w' and char0 != 'x' and char0 != 'y') return true;
        if (len == 2 and isVowel(char1) and !isVowel(char0)) return true;
        return false;
    }

    pub fn isShortWord(self: *const Word) bool {
        return self.endsInShortSyllable() and self.getRegion1() == self.length();
    }

    pub fn length(self: *const Word) u32 {
        if (self.letters[self.start] != 0) {
            return @as(u32, self.end) - self.start + 1;
        }
        return 0;
    }

    pub fn changeSuffix(self: *Word, old_suffix: []const u8, new_suffix: []const u8) bool {
        const len = old_suffix.len;
        const self_len = self.length();
        if (self_len > len) {
            const start_idx = self.end + 1 - len;
            if (std.mem.eql(u8, self.letters[start_idx .. self.end + 1], old_suffix)) {
                const n = new_suffix.len;
                if (n > 0) {
                    const limit = @min(63, @as(u32, self.end) + n) - self.end;
                    @memcpy(self.letters[start_idx .. start_idx + limit], new_suffix[0..limit]);
                    self.end = @intCast(@min(63, @as(u32, self.end) - len + n));
                } else {
                    self.end -= @intCast(len);
                }
                return true;
            }
        }
        return false;
    }

    pub fn matchesAny(self: *Word, a: []const []const u8) bool {
        const len = self.length();
        for (a) |s| {
            if (len == s.len) {
                if (std.mem.eql(u8, self.letters[self.start .. self.start + len], s)) {
                    return true;
                }
            }
        }
        return false;
    }

    pub fn endsWith(self: *const Word, suffix: []const u8) bool {
        const len = suffix.len;
        const self_len = self.length();
        if (self_len > len) {
            const start_idx = self.end + 1 - len;
            return std.mem.eql(u8, self.letters[start_idx .. self.end + 1], suffix);
        }
        return false;
    }

    pub fn startsWith(self: *const Word, prefix: []const u8) bool {
        const len = prefix.len;
        const self_len = self.length();
        if (self_len > len) {
            return std.mem.eql(u8, self.letters[self.start .. self.start + len], prefix);
        }
        return false;
    }

    pub fn hashWord(self: *Word) void {
        var h: u32 = 0xb0a710ad;
        var i: usize = self.start;
        while (i <= self.end) : (i += 1) {
            h = (h *% 263 *% 32) +% self.letters[i];
        }
        self.hash = h;
    }
};

pub const Verb: u32 = 1 << 0;
pub const Noun: u32 = 1 << 1;
pub const Adjective: u32 = 1 << 2;
pub const Plural: u32 = 1 << 3;
pub const PastTense: u32 = (1 << 5) | Verb;
pub const PresentParticiple: u32 = (1 << 4) | Verb;
pub const AdjectiveSuperlative: u32 = (1 << 5) | Adjective;
pub const AdjectiveWithout: u32 = (1 << 6) | Adjective;
pub const AdjectiveFull: u32 = (1 << 7) | Adjective;
pub const AdverbOfManner: u32 = 1 << 8;
pub const Suffix: u32 = 1 << 9;
pub const Prefix: u32 = 1 << 10;
pub const Male: u32 = 1 << 11;
pub const Female: u32 = 1 << 13;
pub const Article: u32 = 1 << 14;
pub const Conjunction: u32 = 1 << 15;
pub const Adposition: u32 = 1 << 16;
pub const Number: u32 = 1 << 17;
pub const Preposition: u32 = 1 << 18;
pub const ConjunctiveAdverb: u32 = 1 << 19;

pub const Negation: u32 = 1 << 0;
pub const PrefixIrr: u32 = (1 << 1) | Negation;
pub const PrefixOver: u32 = 1 << 2;
pub const PrefixUnder: u32 = 1 << 3;
pub const PrefixUnn: u32 = (1 << 4) | Negation;
pub const PrefixNon: u32 = (1 << 5) | Negation;
pub const PrefixAnti: u32 = (1 << 6) | Negation;
pub const PrefixDis: u32 = (1 << 7) | Negation;

pub const SuffixNESS: u32 = 1 << 0;
pub const SuffixITY: u32 = (1 << 1) | Noun;
pub const SuffixCapable: u32 = 1 << 2;
pub const SuffixNCE: u32 = 1 << 3;
pub const SuffixNT: u32 = 1 << 4;
pub const SuffixION: u32 = 1 << 5;
pub const SuffixAL: u32 = (1 << 6) | Adjective;
pub const SuffixIC: u32 = (1 << 7) | Adjective;
pub const SuffixIVE: u32 = 1 << 8;
pub const SuffixOUS: u32 = (1 << 9) | Adjective;

pub const Vowels = "aeiouy";
pub const Doubles = "bdfgmnprt";
pub const LiEndings = "cdeghkmnrt";
pub const NonShortConsonants = "wxY";

pub inline fn charInArray(c: u8, a: []const u8) bool {
    for (a) |x_val| {
        if (x_val == c) return true;
    }
    return false;
}

pub inline fn isVowel(c: u8) bool {
    return charInArray(c, Vowels);
}
pub inline fn isConsonant(c: u8) bool {
    return !isVowel(c);
}
pub inline fn isShortConsonant(c: u8) bool {
    return !charInArray(c, NonShortConsonants);
}
pub inline fn isDouble(c: u8) bool {
    return charInArray(c, Doubles);
}
pub inline fn isLiEnding(c: u8) bool {
    return charInArray(c, LiEndings);
}

pub const ExceptionsRegion1 = [_][]const u8{ "gener", "arsen", "commun" };
pub const Exceptions1 = [_][2][]const u8{
    .{ "skis", "ski" },
    .{ "skies", "sky" },
    .{ "dying", "die" },
    .{ "lying", "lie" },
    .{ "tying", "tie" },
    .{ "idly", "idle" },
    .{ "gently", "gentle" },
    .{ "ugly", "ugli" },
    .{ "early", "earli" },
    .{ "only", "onli" },
    .{ "singly", "singl" },
    .{ "sky", "sky" },
    .{ "news", "news" },
    .{ "howe", "howe" },
    .{ "atlas", "atlas" },
    .{ "cosmos", "cosmos" },
    .{ "bias", "bias" },
    .{ "andes", "andes" },
    .{ "texas", "texas" },
};
pub const TypesExceptions1 = [_]u32{
    Noun | Plural,
    Noun | Plural,
    PresentParticiple,
    PresentParticiple,
    PresentParticiple,
    AdverbOfManner,
    AdverbOfManner,
    Adjective,
    Adjective | AdverbOfManner,
    0,
    AdverbOfManner,
    Noun,
    Noun,
    0,
    Noun,
    Noun,
    Noun,
    Noun | Plural,
    Noun,
};

pub const Exceptions2 = [_][]const u8{ "inning", "outing", "canning", "herring", "earring", "proceed", "exceed", "succeed" };
pub const TypesExceptions2 = [_]u32{ Noun, Noun, Noun, Noun, Noun, Verb, Verb, Verb };

pub const MaleWords = [_][]const u8{ "he", "him", "his", "himself", "man", "men", "boy", "husband", "actor" };
pub const FemaleWords = [_][]const u8{ "she", "her", "herself", "woman", "women", "girl", "wife", "actress" };
pub const ArticleWords = [_][]const u8{ "a", "an", "the" };
pub const ConjWords = [_][]const u8{ "for", "and", "nor", "but", "or", "yet", "so", "than", "as", "that", "if", "when", "because", "while", "where", "after", "though", "whether", "before", "although", "like", "once", "unless", "now", "except" };
pub const ApoWords = [_][]const u8{ "in", "during", "at", "on", "since", "until", "above", "across", "against", "along", "among", "around", "behind", "below", "beneath", "beside", "between", "by", "down", "from", "into", "near", "of", "off", "to", "toward", "under", "upon", "with", "within" };
pub const ConAdVerPrepWords = [_][]const u8{ "also", "thus" };
pub const VerbWords1 = [_][]const u8{ "has", "had", "have", "was", "were", "may", "might", "must", "shall", "should", "can", "could", "will", "would", "is", "am", "are", "be", "being", "been", "do", "does", "did" };
pub const Numbers = [_][]const u8{ "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million" };

pub const SuffixesStep0 = [_][]const u8{ "'s'", "'s", "'" };
pub const SuffixesStep1b = [_][]const u8{ "eedly", "eed", "ed", "edly", "ing", "ingly" };
pub const TypesStep1b = [_]u32{ AdverbOfManner, 0, PastTense, AdverbOfManner | PastTense, PresentParticiple, AdverbOfManner | PresentParticiple };

pub const SuffixesStep2 = [_][2][]const u8{
    .{ "ization", "ize" },
    .{ "ational", "ate" },
    .{ "ousness", "ous" },
    .{ "iveness", "ive" },
    .{ "fulness", "ful" },
    .{ "tional", "tion" },
    .{ "lessli", "less" },
    .{ "biliti", "ble" },
    .{ "entli", "ent" },
    .{ "ation", "ate" },
    .{ "alism", "al" },
    .{ "aliti", "al" },
    .{ "fulli", "ful" },
    .{ "ousli", "ous" },
    .{ "iviti", "ive" },
    .{ "enci", "ence" },
    .{ "anci", "ance" },
    .{ "abli", "able" },
    .{ "izer", "ize" },
    .{ "ator", "ate" },
    .{ "alli", "al" },
    .{ "bli", "ble" },
};
pub const TypesStep2 = [_]u32{
    Suffix,
    Suffix | Adjective,
    Suffix,
    Suffix,
    Suffix,
    Suffix | Adjective,
    AdverbOfManner,
    AdverbOfManner | Noun | Suffix,
    AdverbOfManner,
    Suffix,
    0,
    Noun | Suffix,
    AdverbOfManner,
    AdverbOfManner,
    Noun | Suffix,
    0,
    0,
    AdverbOfManner,
    0,
    0,
    AdverbOfManner,
    AdverbOfManner,
};
pub const TypesStep2Suffix = [_]u32{
    SuffixION,
    SuffixION | SuffixAL,
    SuffixNESS,
    SuffixNESS,
    SuffixNESS,
    SuffixION | SuffixAL,
    0,
    SuffixITY,
    0,
    SuffixION,
    0,
    SuffixITY,
    0,
    0,
    SuffixITY,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
};

pub const SuffixesStep3 = [_][2][]const u8{
    .{ "ational", "ate" },
    .{ "tional", "tion" },
    .{ "alize", "al" },
    .{ "icate", "ic" },
    .{ "iciti", "ic" },
    .{ "ical", "ic" },
    .{ "ful", "" },
    .{ "ness", "" },
};
pub const TypesStep3 = [_]u32{
    Suffix | Adjective,
    Suffix | Adjective,
    0,
    0,
    Noun | Suffix,
    Suffix | Adjective,
    AdjectiveFull,
    Suffix,
};
pub const TypesStep3Suffix = [_]u32{
    SuffixION | SuffixAL,
    SuffixION | SuffixAL,
    0,
    0,
    SuffixITY,
    SuffixAL,
    0,
    SuffixNESS,
};

pub const SuffixesStep4 = [_][]const u8{ "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize", "sion", "tion" };
pub const TypesStep4 = [_]u32{ Suffix | Adjective, Suffix, Suffix, 0, Suffix | Adjective, Suffix, Suffix, Suffix, 0, 0, Suffix, 0, 0, 0, Suffix | Noun, Suffix | Adjective, Suffix, 0, Suffix, Suffix };
pub const TypesStep4Suffix = [_]u32{ SuffixAL, SuffixNCE, SuffixNCE, 0, SuffixIC, SuffixCapable, SuffixCapable, SuffixNT, 0, 0, SuffixNT, 0, 0, 0, SuffixITY, SuffixOUS, SuffixIVE, 0, SuffixION, SuffixION };

pub const EnglishStemmer = struct {
    fn processPrefixes(self: *const EnglishStemmer, W: *Word) bool {
        _ = self;
        if (W.startsWith("irr") and W.length() > 5 and (W.letters[W.start + 3] == 'a' or W.letters[W.start + 3] == 'e')) {
            W.start += 2;
            W.type |= Prefix;
            W.prefix |= PrefixIrr;
        } else if (W.startsWith("over") and W.length() > 5) {
            W.start += 4;
            W.type |= Prefix;
            W.prefix |= PrefixOver;
        } else if (W.startsWith("under") and W.length() > 6) {
            W.start += 5;
            W.type |= Prefix;
            W.prefix |= PrefixUnder;
        } else if (W.startsWith("unn") and W.length() > 5) {
            W.start += 2;
            W.type |= Prefix;
            W.prefix |= PrefixUnn;
        } else if (W.startsWith("non") and W.length() > (5 + @as(u32, @intFromBool(W.letters[W.start + 3] == '-')))) {
            const add = 2 + @as(u8, @intFromBool(W.letters[W.start + 3] == '-'));
            W.start += add;
            W.type |= Prefix;
            W.prefix |= PrefixNon;
        } else if (W.startsWith("anti") and W.length() > 6 and (W.letters[W.start + 4] == '-')) {
            const add = 4 + @as(u8, @intFromBool(W.letters[W.start + 4] == '-'));
            W.start += add;
            W.type |= Prefix;
            W.prefix |= PrefixAnti;
        } else if (W.startsWith("dis") and W.length() > 5 and (W.letters[W.start + 3] == '-')) {
            const add = 2 + @as(u8, @intFromBool(W.letters[W.start + 3] == '-'));
            W.start += add;
            W.type |= Prefix;
            W.prefix |= PrefixDis;
        } else {
            return false;
        }
        return true;
    }

    fn processSuperlatives(self: *const EnglishStemmer, W: *Word) bool {
        _ = self;
        if (W.endsWith("est") and W.length() > 4) {
            const i = W.end;
            W.end -= 3;
            W.type |= AdjectiveSuperlative;

            if (W.fromEnd(0) == W.fromEnd(1) and W.fromEnd(0) != 'r' and !(W.length() >= 4 and std.mem.eql(u8, W.letters[W.end - 3 .. W.end + 1], "sugg"))) {
                const term1 = (W.fromEnd(0) != 'f' and W.fromEnd(0) != 'l' and W.fromEnd(0) != 's') or
                    (W.length() > 4 and W.fromEnd(1) == 'l' and (W.fromEnd(2) == 'u' or W.fromEnd(3) == 'u' or W.fromEnd(3) == 'v'));
                const term2 = !(W.length() == 3 and W.fromEnd(1) == 'd' and W.fromEnd(2) == 'o');
                if (term1 and term2) {
                    W.end -= 1;
                }
                if (W.length() == 2 and (W.at(0) != 'i' or W.at(1) != 'n')) {
                    W.end = i;
                    W.type &= ~AdjectiveSuperlative;
                }
            } else {
                switch (W.fromEnd(0)) {
                    'd', 'k', 'm', 'y' => {},
                    'g' => {
                        if (!(W.length() > 3 and (W.fromEnd(1) == 'n' or W.fromEnd(2) == 'r') and !std.mem.eql(u8, W.letters[W.end - 3 .. W.end + 1], "cong"))) {
                            W.end = i;
                            W.type &= ~AdjectiveSuperlative;
                        } else {
                            if (W.fromEnd(2) == 'a') {
                                W.end += 1;
                            }
                        }
                    },
                    'i' => {
                        W.letters[W.end] = 'y';
                    },
                    'l' => {
                        if (W.end == W.start + 1 or std.mem.eql(u8, W.letters[W.end - 2 .. W.end], "mo")) {
                            W.end = i;
                            W.type &= ~AdjectiveSuperlative;
                        } else {
                            if (isConsonant(W.fromEnd(1))) {
                                W.end += 1;
                            }
                        }
                    },
                    'n' => {
                        if (W.length() < 3 or isConsonant(W.fromEnd(1)) or isConsonant(W.fromEnd(2))) {
                            W.end = i;
                            W.type &= ~AdjectiveSuperlative;
                        }
                    },
                    'r' => {
                        if (W.length() > 3 and isVowel(W.fromEnd(1)) and isVowel(W.fromEnd(2))) {
                            if (W.fromEnd(2) == 'u' and (W.fromEnd(1) == 'a' or W.fromEnd(1) == 'i')) {
                                W.end += 1;
                            }
                        } else {
                            W.end = i;
                            W.type &= ~AdjectiveSuperlative;
                        }
                    },
                    's' => {
                        W.end += 1;
                    },
                    'w' => {
                        if (!(W.length() > 2 and isVowel(W.fromEnd(1)))) {
                            W.end = i;
                            W.type &= ~AdjectiveSuperlative;
                        }
                    },
                    'h' => {
                        if (!(W.length() > 2 and isConsonant(W.fromEnd(1)))) {
                            W.end = i;
                            W.type &= ~AdjectiveSuperlative;
                        }
                    },
                    else => {
                        W.end += 3;
                        W.type &= ~AdjectiveSuperlative;
                    },
                }
            }
        }
        return (W.type & AdjectiveSuperlative) > 0;
    }

    fn step0(self: *const EnglishStemmer, W: *Word) bool {
        _ = self;
        for (SuffixesStep0) |sfx| {
            if (W.endsWith(sfx)) {
                W.end -= @intCast(sfx.len);
                W.type |= Plural;
                return true;
            }
        }
        return false;
    }

    fn step1a(self: *const EnglishStemmer, W: *Word) bool {
        _ = self;
        if (W.endsWith("sses")) {
            W.end -= 2;
            W.type |= Plural;
            return true;
        }
        if (W.endsWith("ied") or W.endsWith("ies")) {
            W.type |= if (W.fromEnd(0) == 'd') PastTense else Plural;
            W.end -= 1 + @as(u8, @intCast(@intFromBool(W.length() > 4)));
            return true;
        }
        if (W.endsWith("us") or W.endsWith("ss")) {
            return false;
        }
        if (W.fromEnd(0) == 's' and W.length() > 2) {
            var i = @as(usize, W.start);
            const limit = @as(usize, W.end) - 2;
            while (i <= limit) : (i += 1) {
                if (isVowel(W.letters[i])) {
                    W.end -= 1;
                    W.type |= Plural;
                    return true;
                }
            }
        }
        if (W.endsWith("n't") and W.length() > 4) {
            switch (W.fromEnd(3)) {
                'a' => {
                    if (W.fromEnd(4) == 'c') {
                        W.end -= 2;
                    } else {
                        _ = W.changeSuffix("n't", "ll");
                    }
                },
                'i' => {
                    _ = W.changeSuffix("in't", "m");
                },
                'o' => {
                    if (W.fromEnd(4) == 'w') {
                        _ = W.changeSuffix("on't", "ill");
                    } else {
                        W.end -= 3;
                    }
                },
                else => {
                    W.end -= 3;
                },
            }
            W.type |= Prefix;
            W.prefix |= Negation;
            return true;
        }
        if (W.endsWith("hood") and W.length() > 7) {
            W.end -= 4;
            return true;
        }
        return false;
    }

    fn step1b(self: *const EnglishStemmer, W: *Word, R1: u32) bool {
        _ = self;
        for (SuffixesStep1b, 0..) |sfx, i| {
            if (W.endsWith(sfx)) {
                switch (i) {
                    0, 1 => {
                        if (W.suffixInRn(R1, sfx)) {
                            W.end -= @intCast(1 + i * 2);
                        }
                    },
                    else => {
                        const j = W.end;
                        W.end -= @intCast(sfx.len);
                        if (W.hasVowels()) {
                            if (W.endsWith("at") or W.endsWith("bl") or W.endsWith("iz") or W.isShortWord()) {
                                W.append('e');
                            } else if (W.length() > 2) {
                                if (W.fromEnd(0) == W.fromEnd(1) and isDouble(W.fromEnd(0))) {
                                    W.end -= 1;
                                } else if (i == 2 or i == 3) {
                                    switch (W.fromEnd(0)) {
                                        'c', 's', 'v' => {
                                            W.end += if (!(W.endsWith("ss") or W.endsWith("ias"))) @as(u8, 1) else @as(u8, 0);
                                        },
                                        'd' => {
                                            const nAllowed = "aeio";
                                            W.end += if (isVowel(W.fromEnd(1)) and !charInArray(W.fromEnd(2), nAllowed)) @as(u8, 1) else @as(u8, 0);
                                        },
                                        'k' => {
                                            W.end += if (W.endsWith("uak")) @as(u8, 1) else @as(u8, 0);
                                        },
                                        'l' => {
                                            const allowed1 = "bcdfgkptyz";
                                            const allowed2 = "aiou";
                                            W.end += if (charInArray(W.fromEnd(1), allowed1) or (charInArray(W.fromEnd(1), allowed2) and isConsonant(W.fromEnd(2)))) @as(u8, 1) else @as(u8, 0);
                                        },
                                        else => {},
                                    }
                                } else if (i >= 4) {
                                    switch (W.fromEnd(0)) {
                                        'd' => {
                                            if (isVowel(W.fromEnd(1)) and W.fromEnd(2) != 'a' and W.fromEnd(2) != 'e' and W.fromEnd(2) != 'o') {
                                                W.append('e');
                                            }
                                        },
                                        'g' => {
                                            const allowed = "adeilru";
                                            if (charInArray(W.fromEnd(1), allowed) or (W.fromEnd(1) == 'n' and (W.fromEnd(2) == 'e' or (W.fromEnd(2) == 'u' and W.fromEnd(3) != 'b' and W.fromEnd(3) != 'd') or (W.fromEnd(2) == 'a' and (W.fromEnd(3) == 'r' or (W.fromEnd(3) == 'h' and W.fromEnd(4) == 'c'))) or (W.endsWith("ring") and (W.fromEnd(4) == 'c' or W.fromEnd(4) == 'f'))))) {
                                                W.append('e');
                                            }
                                        },
                                        'l' => {
                                            if (!(W.fromEnd(1) == 'l' or W.fromEnd(1) == 'r' or W.fromEnd(1) == 'w' or (isVowel(W.fromEnd(1)) and isVowel(W.fromEnd(2))))) {
                                                W.append('e');
                                            }
                                            if (W.endsWith("uell") and W.length() > 4 and W.fromEnd(4) != 'q') {
                                                W.end -= 1;
                                            }
                                        },
                                        'r' => {
                                            if (((W.fromEnd(1) == 'i' and W.fromEnd(2) != 'a' and W.fromEnd(2) != 'e' and W.fromEnd(2) != 'o') or
                                                (W.fromEnd(1) == 'a' and !(W.fromEnd(2) == 'e' or W.fromEnd(2) == 'o' or (W.fromEnd(2) == 'l' and W.fromEnd(3) == 'l'))) or
                                                (W.fromEnd(1) == 'o' and !(W.fromEnd(2) == 'o' or (W.fromEnd(2) == 't' and W.fromEnd(3) != 's'))) or
                                                W.fromEnd(1) == 'c' or W.fromEnd(1) == 't') and !W.endsWith("str"))
                                            {
                                                W.append('e');
                                            }
                                        },
                                        't' => {
                                            if (W.fromEnd(1) == 'o' and W.fromEnd(2) != 'g' and W.fromEnd(2) != 'l' and W.fromEnd(2) != 'i' and W.fromEnd(2) != 'o') {
                                                W.append('e');
                                            }
                                        },
                                        'u' => {
                                            if (!(W.length() > 3 and isVowel(W.fromEnd(1)) and isVowel(W.fromEnd(2)))) {
                                                W.append('e');
                                            }
                                        },
                                        'z' => {
                                            if (W.endsWith("izz") and W.length() > 3 and (W.fromEnd(3) == 'h' or W.fromEnd(3) == 'u')) {
                                                W.end -= 1;
                                            } else if (W.fromEnd(1) != 't' and W.fromEnd(1) != 'z') {
                                                W.append('e');
                                            }
                                        },
                                        'k' => {
                                            if (W.endsWith("uak")) {
                                                W.append('e');
                                            }
                                        },
                                        'b', 'c', 's', 'v' => {
                                            if (!((W.fromEnd(0) == 'b' and (W.fromEnd(1) == 'm' or W.fromEnd(1) == 'r')) or W.endsWith("ss") or W.endsWith("ias") or W.eqlStr("zinc"))) {
                                                W.append('e');
                                            }
                                        },
                                        else => {},
                                    }
                                }
                            }
                        } else {
                            W.end = j;
                            return false;
                        }
                    },
                }
                W.type |= TypesStep1b[i];
                return true;
            }
        }
        return false;
    }

    fn step1c(self: *const EnglishStemmer, W: *Word) bool {
        _ = self;
        if (W.length() > 2 and W.fromEnd(0) == 'y' and isConsonant(W.fromEnd(1))) {
            W.letters[W.end] = 'i';
            return true;
        }
        return false;
    }

    fn step2(self: *const EnglishStemmer, W: *Word, R1: u32) bool {
        _ = self;
        for (SuffixesStep2, 0..) |pair, i| {
            if (W.endsWith(pair[0]) and W.suffixInRn(R1, pair[0])) {
                _ = W.changeSuffix(pair[0], pair[1]);
                W.type |= TypesStep2[i];
                W.suffix |= TypesStep2Suffix[i];
                return true;
            }
        }
        if (W.endsWith("logi") and W.suffixInRn(R1, "ogi")) {
            W.end -= 1;
            return true;
        } else if (W.endsWith("li")) {
            if (W.suffixInRn(R1, "li") and isLiEnding(W.fromEnd(2))) {
                W.end -= 2;
                W.type |= AdverbOfManner;
                return true;
            } else if (W.length() > 3) {
                switch (W.fromEnd(2)) {
                    'b' => {
                        W.letters[W.end] = 'e';
                        W.type |= AdverbOfManner;
                        return true;
                    },
                    'i' => {
                        if (W.length() > 4) {
                            W.end -= 2;
                            W.type |= AdverbOfManner;
                            return true;
                        }
                    },
                    'l' => {
                        if (W.length() > 5 and (W.fromEnd(3) == 'a' or W.fromEnd(3) == 'u')) {
                            W.end -= 2;
                            W.type |= AdverbOfManner;
                            return true;
                        }
                    },
                    's' => {
                        W.end -= 2;
                        W.type |= AdverbOfManner;
                        return true;
                    },
                    'e', 'g', 'm', 'n', 'r', 'w' => {
                        const limit = 4 + @as(u32, @intFromBool(W.fromEnd(2) == 'r'));
                        if (W.length() > limit) {
                            W.end -= 2;
                            W.type |= AdverbOfManner;
                            return true;
                        }
                    },
                    else => {},
                }
            }
        }
        return false;
    }

    fn step3(self: *const EnglishStemmer, W: *Word, R1: u32, R2: u32) bool {
        _ = self;
        var res = false;
        for (SuffixesStep3, 0..) |pair, i| {
            if (W.endsWith(pair[0]) and W.suffixInRn(R1, pair[0])) {
                _ = W.changeSuffix(pair[0], pair[1]);
                W.type |= TypesStep3[i];
                W.suffix |= TypesStep3Suffix[i];
                res = true;
                break;
            }
        }
        if (W.endsWith("ative") and W.suffixInRn(R2, "ative")) {
            W.end -= 5;
            W.type |= Suffix;
            W.suffix |= SuffixIVE;
            return true;
        }
        if (W.length() > 5 and W.endsWith("less")) {
            W.end -= 4;
            W.type |= AdjectiveWithout;
            return true;
        }
        return res;
    }

    fn step4(self: *const EnglishStemmer, W: *Word, R2: u32) bool {
        _ = self;
        var res = false;
        for (SuffixesStep4, 0..) |sfx, i| {
            if (W.endsWith(sfx) and W.suffixInRn(R2, sfx)) {
                const sub = sfx.len - @as(usize, @intFromBool(i > 17));
                W.end -= @intCast(sub);
                if (i != 10 or W.fromEnd(0) != 'm') {
                    W.type |= TypesStep4[i];
                    W.suffix |= TypesStep4Suffix[i];
                }
                if (i == 0 and W.endsWith("nti")) {
                    W.end -= 1;
                    res = true;
                    continue;
                }
                return true;
            }
        }
        return res;
    }

    fn step5(self: *const EnglishStemmer, W: *Word, R1: u32, R2: u32) bool {
        _ = self;
        if (W.fromEnd(0) == 'e' and !W.eqlStr("here")) {
            if (W.suffixInRn(R2, "e")) {
                W.end -= 1;
            } else if (W.suffixInRn(R1, "e")) {
                W.end -= 1;
                W.end += @as(u8, @intCast(@intFromBool(W.endsInShortSyllable())));
            } else {
                return false;
            }
            return true;
        } else if (W.length() > 1 and W.fromEnd(0) == 'l' and W.suffixInRn(R2, "l") and W.fromEnd(1) == 'l') {
            W.end -= 1;
            return true;
        }
        return false;
    }

    pub fn stem(self: *const EnglishStemmer, W: *Word, blpos: u32) bool {
        var res = W.trimStartingApostrophe();
        if (self.processPrefixes(W)) res = true;
        if (self.processSuperlatives(W)) res = true;
        for (Exceptions1, 0..) |ex, i| {
            if (W.eqlStr(ex[0])) {
                if (i < 11) {
                    const len = ex[1].len;
                    @memcpy(W.letters[W.start .. W.start + len], ex[1]);
                    W.end = @intCast(W.start + len - 1);
                }
                W.hashWord();
                W.type |= TypesExceptions1[i];
                return (i < 11);
            }
        }

        W.markYsAsConsonants();
        const R1 = W.getRegion1();
        const R2 = W.getRegion(R1);
        if (self.step0(W)) res = true;
        if (self.step1a(W)) res = true;
        for (Exceptions2, 0..) |ex, i| {
            if (W.eqlStr(ex)) {
                W.hashWord();
                W.type |= TypesExceptions2[i];
                return res;
            }
        }
        if (self.step1b(W, R1)) res = true;
        if (self.step1c(W)) res = true;
        if (self.step2(W, R1)) res = true;
        if (self.step3(W, R1, R2)) res = true;
        if (self.step4(W, R2)) res = true;
        if (self.step5(W, R1, R2)) res = true;

        var i = W.start;
        while (i <= W.end) : (i += 1) {
            if (W.letters[i] == 'Y') {
                W.letters[i] = 'y';
            }
        }
        if (W.type == 0 or W.type == Plural) {
            if (W.matchesAny(&MaleWords)) {
                res = true;
                W.type |= Male;
            } else if (W.matchesAny(&FemaleWords)) {
                res = true;
                W.type |= Female;
            } else if (W.matchesAny(&ArticleWords)) {
                res = true;
                W.type |= Article;
            } else if (W.matchesAny(&ConjWords)) {
                res = true;
                W.type |= Conjunction;
            } else if (W.matchesAny(&ApoWords)) {
                res = true;
                W.type |= Adposition;
            } else if (W.matchesAny(&ConAdVerPrepWords)) {
                res = true;
                W.type |= ConjunctiveAdverb;
            } else if ((blpos < 451531986) and W.matchesAny(&VerbWords1)) {
                res = true;
                W.type |= Verb;
            } else if (W.matchesAny(&Numbers)) {
                res = true;
                W.type |= Number;
            }
        }
        W.hashWord();
        return res;
    }
};

pub const primes = [_]u32{ 0, 257, 251, 241, 239, 233, 229, 227, 223, 211, 199, 197, 193, 191 };
pub const tri = [_]u32{ 0, 4, 3, 7 };
pub const trj = [_]u32{ 0, 6, 6, 12 };

pub const c_r = [_]u32{ 3, 4, 6, 4, 6, 6, 2, 3, 3, 3, 6, 4, 3, 4, 5, 6, 2, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4 };
pub const c_s = [_]u32{ 28, 26, 28, 31, 34, 31, 33, 33, 35, 35, 29, 32, 33, 34, 30, 36, 31, 32, 32, 32, 32, 32, 33, 32, 32, 32, 32 };
pub const c_s3 = [_]u32{ 43, 33, 34, 28, 34, 29, 32, 33, 37, 35, 33, 28, 31, 35, 28, 30, 33, 34, 32, 32, 32, 32, 32, 32, 32, 32, 32 };
pub const c_s4 = [_]u32{ 9, 8, 9, 5, 8, 12, 15, 8, 8, 12, 10, 7, 7, 8, 8, 13, 13, 14, 8, 8, 12, 12, 12, 12, 12, 12, 12 };

pub const e_l = [_]i32{ 1830, 1997, 1973, 1851, 1897, 1690, 1998, 1842 };

pub const st2_p0 = [_]i16{0} ** 4096;
pub const st2_p1 = init_st2_p1: {
    @setEvalBranchQuota(100000);
    var arr = [_]i16{0} ** 4096;
    var i: usize = 0;
    while (i < 4096) : (i += 1) {
        arr[i] = @intCast(clp(sc(12 * (@as(i32, @intCast(i)) - 2048))));
    }
    break :init_st2_p1 arr;
};
pub const st2_p2 = init_st2_p2: {
    @setEvalBranchQuota(100000);
    var arr = [_]i16{0} ** 4096;
    var i: usize = 0;
    while (i < 4096) : (i += 1) {
        arr[i] = @intCast(clp(sc(14 * (@as(i32, @intCast(i)) - 2048))));
    }
    break :init_st2_p2 arr;
};

pub fn vec(comptime T: type, comptime S: usize) type {
    return struct {
        const Self = @This();
        cxt: [S]T = [_]T{std.mem.zeroes(T)} ** S,
        size: usize = 0,

        pub fn size_val(self: *const Self) usize {
            return self.size;
        }

        pub fn push(self: *Self, element: T) void {
            self.cxt[self.size] = element;
            self.size = (self.size + 1) & (S - 1);
        }

        pub fn at(self: *const Self, index: usize) T {
            return self.cxt[index];
        }

        pub fn set_at(self: *Self, index: usize, val: T) void {
            self.cxt[index] = val;
        }

        pub fn inc(self: *Self, index: usize) void {
            self.cxt[index] +%= 1;
        }

        pub fn pop(self: *Self) void {
            if (self.size > 0) {
                self.size -= 1;
                self.cxt[self.size] = std.mem.zeroes(T);
            }
        }

        pub fn reset(self: *Self) void {
            self.cxt[0] = std.mem.zeroes(T);
            self.size = 0;
        }

        pub fn empty(self: *const Self) bool {
            return self.size == 0;
        }

        pub fn prev(self: *const Self) T {
            if (self.size > 1) {
                return self.cxt[self.size - 2];
            }
            return std.mem.zeroes(T);
        }
    };
}

pub fn BracketContext(comptime T: type) type {
    return struct {
        const Self = @This();
        context: u32 = 0,
        active: vec(i32, 512) = .{},
        distance: vec(i32, 512) = .{},
        element: []const T = &.{},
        doPop: bool = false,
        limit: i32 = 0,
        cxt: T = 0,
        dst: T = 0,

        pub fn init_ctx(self: *Self, d: []const T, pop: bool, l: i32) void {
            self.element = d;
            self.context = 0;
            self.doPop = pop;
            self.limit = l;
            self.cxt = 0;
            self.dst = 0;
            self.active.reset();
            self.distance.reset();
        }

        pub fn reset(self: *Self) void {
            self.active.reset();
            self.distance.reset();
            self.context = 0;
            self.cxt = 0;
            self.dst = 0;
        }

        fn find(self: *const Self, b: T) bool {
            var i: usize = 0;
            while (i < self.element.len) : (i += 2) {
                if (self.element[i] == b) return true;
            }
            return false;
        }

        fn findEnd(self: *const Self, b: T, c: T) bool {
            var i: usize = 0;
            while (i < self.element.len) : (i += 2) {
                if (self.element[i] == b and self.element[i + 1] == c) return true;
            }
            return false;
        }

        pub fn last(self: *const Self) i32 {
            return self.active.prev();
        }

        pub fn update(self: *Self, byte: T) void {
            var pop = false;
            if (!self.active.empty()) {
                const last_active = self.active.at(self.active.size_val() - 1);
                const last_dist = self.distance.at(self.distance.size_val() - 1);
                if (self.findEnd(@intCast(last_active), byte) or last_dist >= self.limit) {
                    self.active.pop();
                    self.distance.pop();
                    pop = self.doPop;
                } else {
                    self.distance.inc(self.distance.size_val() - 1);
                }
            }
            if (!pop and self.find(byte)) {
                self.active.push(byte);
                self.distance.push(0);
            }
            if (!self.active.empty()) {
                self.cxt = @intCast(self.active.at(self.active.size_val() - 1));
                const dist_val = self.distance.at(self.distance.size_val() - 1);
                const limit_t = (1 << (@sizeOf(T) * 8)) - 1;
                self.dst = @intCast(@min(dist_val, limit_t));
                self.context = @as(u32, 1 << (@sizeOf(T) * 8)) * @as(u32, self.cxt) + self.dst;
            } else {
                self.context = 0;
                self.cxt = 0;
                self.dst = 0;
            }
        }
    };
}

pub const Column = struct {
    linepos: u32 = 0,
    fc: u8 = 0,
    bytes: vec(u8, 2048) = .{},
};

pub const ColumnContext = struct {
    col: [4]Column = [_]Column{.{}} ** 4,
    cell: [4]vec(u32, 32) = [_]vec(u32, 32){.{}} ** 4,
    rows: i32 = 0,
    cellCount: i32 = 0,
    cells: i32 = 0,
    abovecellpos: u32 = 0,
    abovecellpos1: u32 = 0,
    NL: bool = false,
    isTemp: bool = false,
    limit: i32 = 0,
    nlChar: u8 = 0,

    pub fn init_ctx(self: *ColumnContext, l: i32) void {
        self.rows = 0;
        self.abovecellpos = 0;
        self.cellCount = 0;
        self.abovecellpos1 = 0;
        self.nlChar = 10;
        self.limit = l;
        for (&self.col) |*c| c.bytes.reset();
        for (&self.cell) |*c| c.reset();
        self.NL = false;
        self.isTemp = false;
    }

    pub fn lastfc(self: *const ColumnContext, i: i32) u8 {
        return self.col[@intCast((self.rows - i) & 3)].fc;
    }

    pub fn isNewLine(self: *const ColumnContext) bool {
        return self.NL;
    }

    pub fn collen(self: *const ColumnContext, i: i32, l: i32) i32 {
        const actual_l = if (l != 0) l else self.limit;
        const size_val = self.col[@intCast((self.rows - i) & 3)].bytes.size_val();
        return @intCast(@min(@as(usize, @intCast(actual_l)), size_val + 1));
    }

    pub fn nlpos(self: *const ColumnContext, i: i32) u32 {
        return self.col[@intCast((self.rows - i) & 3)].linepos;
    }

    pub fn colb(self: *const ColumnContext, i: i32, j: i32) u8 {
        if (self.collen(0, 0) < self.collen(i, 0)) {
            const idx = @as(i32, @intCast(self.collen(0, 0))) - (1 + j);
            if (idx >= 0) {
                return self.col[@intCast((self.rows - i) & 3)].bytes.at(@intCast(idx));
            }
        }
        return 0;
    }

    pub fn update(self: *ColumnContext, byte: u8, b2: u32, blpos: u32) void {
        if (b2 == (('}' << 16) + ('}' << 8) + '|')) {
            self.nlChar = '-';
        } else if (b2 == (('|' << 16) + ('}' << 8) + '}')) {
            self.nlChar = 10;
            self.resetCells();
        }

        if (byte != '{' and (b2 & 0xff00) == ('{' << 8) and (b2 & 0xff0000) != ('{' << 16)) {
            self.isTemp = true;
        } else if (self.isTemp and byte == '}') {
            self.isTemp = false;
        }

        self.NL = false;
        if (byte == 10) {
            self.col[@intCast(self.rows)].bytes.push(byte);
            self.rows = (self.rows + 1) & 3;
            self.col[@intCast(self.rows)].bytes.reset();
            self.col[@intCast(self.rows)].fc = 0;
            self.col[@intCast(self.rows)].linepos = blpos -% 1;
        } else {
            self.col[@intCast(self.rows)].bytes.push(byte);
            if (self.collen(0, 0) == 2) {
                self.col[@intCast(self.rows)].fc = @min(byte, 96);
                self.NL = true;
                if (self.col[@intCast(self.rows)].fc == '>' and !isPre) {
                    self.nlChar = '>';
                }
                if (self.col[@intCast(self.rows)].fc == '[' and self.nlChar == '>') {
                    self.nlChar = 10;
                }
            }
        }

        if (self.nlChar == '-') {
            if ((b2 & 0xffff) == ('-' + '|' * 256)) {
                self.cells = (self.cells + 1) & 3;
                self.cell[@intCast(self.cells)].reset();
                self.cell[@intCast(self.cells)].push(blpos);
                self.cellCount = 0;
                self.abovecellpos = 0;
                self.abovecellpos1 = 0;
            }
            var newcell = false;
            if ((b2 & 0xffff) == ('|' + '|' * 256) or
                (b2 & 0xffff00) == (('|' + 10 * 256) * 256) or
                ((b2 & 0xffff00) == (('|' + 10 * 256) * 256) and byte != '|'))
            {
                self.cell[@intCast(self.cells)].push(blpos);
                self.cellCount += 1;
                newcell = true;
            }
            if (self.abovecellpos != 0) {
                self.abovecellpos += 1;
                if (self.abovecellpos > self.abovecellpos1) {
                    self.abovecellpos = 0;
                    self.abovecellpos1 = 0;
                }
            }
            if (newcell and self.cellsCount(1) > 0) {
                self.abovecellpos = self.cellPos(self.cellCount - 1, 1);
                self.abovecellpos1 = self.cellPos(self.cellCount, 1);
            }
        }

        if (self.nlChar == '>') {
            if ((b2 & 0xffff) == ('>' + 10 * 256)) {
                self.cells = (self.cells + 1) & 3;
                self.cell[@intCast(self.cells)].reset();
                self.cell[@intCast(self.cells)].push(blpos);
                self.cellCount = 0;
                self.abovecellpos = 0;
                self.abovecellpos1 = 0;
            } else {
                var newcell = false;
                if ((b2 & 0xff) == '>') {
                    self.cell[@intCast(self.cells)].push(blpos);
                    self.cellCount += 1;
                    newcell = true;
                }
                if (self.abovecellpos != 0) {
                    self.abovecellpos += 1;
                    if (self.abovecellpos > self.abovecellpos1) {
                        self.abovecellpos = 0;
                        self.abovecellpos1 = 0;
                    }
                }
                if (newcell and self.cellsCount(1) > 0) {
                    self.abovecellpos = self.cellPos(self.cellCount - 1, 1);
                    self.abovecellpos1 = self.cellPos(self.cellCount, 1);
                }
            }
        }
    }

    pub fn cellsCount(self: *const ColumnContext, row: i32) i32 {
        return @intCast(self.cell[@intCast((self.cells - row) & 3)].size_val());
    }

    pub fn cellPos(self: *const ColumnContext, cellID: i32, row: i32) u32 {
        var total = self.cellsCount(row) - 1;
        total = @min(total, cellID);
        if (total >= 0) {
            return self.cell[@intCast((self.cells - row) & 3)].at(@intCast(total));
        }
        return 0;
    }

    pub fn resetCells(self: *ColumnContext) void {
        for (&self.cell) |*c| c.reset();
    }
};

pub const WordsContext = struct {
    sbytes: vec(u16, 256) = .{},
    type: vec(u32, 256) = .{},
    stem: vec(u32, 256) = .{},
    capital: vec(u8, 256) = .{},
    fword: u32 = 0,
    ftype: u32 = 0,
    pbyte: u8 = 0,
    wordcount: i32 = 0,
    upper: i32 = 0,
    ref: i32 = 0,

    pub fn init_ctx(self: *WordsContext) void {
        self.sbytes.reset();
        self.type.reset();
        self.stem.reset();
        self.capital.reset();
        self.fword = 0;
        self.ftype = 0;
        self.pbyte = 0;
        self.wordcount = 0;
        self.upper = 0;
        self.ref = 0;
    }

    pub fn reset(self: *WordsContext) void {
        self.sbytes.reset();
        self.type.reset();
        self.stem.reset();
        self.capital.reset();
        self.fword = 0;
        self.ftype = 0;
        self.pbyte = 0;
        self.wordcount = 0;
        self.upper = 0;
        self.ref = 0;
    }

    pub fn set_pbyte(self: *WordsContext, b: u8, a: i32) void {
        self.pbyte = b;
        self.upper = a;
    }

    pub fn update(self: *WordsContext, w: u32, b: u8, t_val: u32, s: u32) void {
        if (self.fword == 0) self.fword = w;
        self.sbytes.push(@intCast(@as(u32, self.pbyte) * 256 + b));
        self.type.push(t_val);
        self.stem.push(s);
        self.capital.push(@intCast(self.upper));
        self.pbyte = 0;
        self.wordcount += 1;
        if (self.ftype == 0 and t_val != 0) self.ftype = t_val;
    }

    pub fn remove(self: *WordsContext) void {
        const num = self.stem.size_val();
        if (num > 0) {
            self.sbytes.pop();
            self.type.pop();
            self.stem.pop();
            self.capital.pop();
            self.wordcount -= 1;
        }
    }

    pub fn word(self: *const WordsContext, i: usize) u32 {
        const num = self.stem.size_val();
        if (num >= i) return self.stem.at(num - i);
        return 0;
    }

    pub fn sBytes(self: *const WordsContext, i: usize) u16 {
        const num = self.sbytes.size_val();
        if (num >= i) return self.sbytes.at(num - i);
        return 0;
    }

    pub fn type_val(self: *const WordsContext, i: usize) u32 {
        const num = self.type.size_val();
        if (num >= i) return self.type.at(num - i);
        return 0;
    }

    pub fn capital_val(self: *const WordsContext, i: usize) u8 {
        const num = self.capital.size_val();
        if (num >= i) return self.capital.at(num - i);
        return 0;
    }

    pub fn last(self: *const WordsContext, j: usize, t_val: u32) u32 {
        const num = self.type.size_val();
        if (t_val == 0) return self.word(j);
        if (num >= j) {
            var i = j;
            while (i < num) : (i += 1) {
                const typ = self.type_val(i);
                if ((typ & t_val) != 0) return self.word(i);
            }
        }
        return self.word(j);
    }

    pub fn lastIf(self: *const WordsContext, j: usize, t_val: u32) u32 {
        const num = self.type.size_val();
        if (t_val == 0) return self.word(j);
        if (num >= j) {
            var i = j;
            while (i < num) : (i += 1) {
                const typ = self.type_val(i);
                if ((typ & t_val) != 0) return self.word(i);
            }
        }
        return 0;
    }

    pub fn lastIdx(self: *const WordsContext, j: usize, t_val: u32) usize {
        const num = self.type.size_val();
        if (t_val == 0) return 0;
        if (num >= j) {
            var i = j;
            while (i < num) : (i += 1) {
                const typ = self.type_val(i);
                if ((typ & t_val) != 0) return i;
            }
        }
        return 0;
    }

    pub fn removeWordsL(self: *WordsContext, len_val: usize, c: u8, d: u8) void {
        if ((self.sBytes(1) & 0xff) == d) {
            var i: usize = 1;
            while (i < len_val) : (i += 1) {
                if ((self.sBytes(i) >> 8) == c) {
                    while ((self.sBytes(1) >> 8) != c) {
                        self.remove();
                    }
                    self.remove();
                    break;
                }
            }
        }
    }

    pub fn removeWordsR(self: *WordsContext, len_val: usize, c: u8, d: u8) void {
        if ((self.sBytes(1) & 0xff) == d) {
            var i: usize = 1;
            while (i < len_val) : (i += 1) {
                if ((self.sBytes(i) & 0xff) == c) {
                    while ((self.sBytes(1) & 0xff) != c) {
                        self.remove();
                    }
                    self.remove();
                    break;
                }
            }
        }
    }
};

pub inline fn hash(a: u32, b: u32, c: u32) u32 {
    const h = a *% 110002499 +% b *% 30005491 +% c *% 50004239;
    return h ^ (h >> 9) ^ (a >> 3) ^ (b >> 3) ^ (c >> 4);
}

pub inline fn charSwap(c: u8) u8 {
    var res = c;
    if (c >= '{' and c < 127) {
        res = c +% (@as(u8, 'P') -% @as(u8, '{'));
    } else if (c >= 'P' and c < 'T') {
        res = c -% (@as(u8, 'P') -% @as(u8, '{'));
    } else if ((c >= ':' and c <= '?') or (c >= 'J' and c <= 'O')) {
        res = c ^ 0x70;
    }
    if (res == 'X' or res == '`') {
        res ^= 'X' ^ '`';
    }
    return res;
}

pub const MTFList = struct {
    root: i32 = 0,
    index: i32 = 0,
    previous: [4]i32 = [_]i32{0} ** 4,
    next: [4]i32 = [_]i32{0} ** 4,

    pub fn init_list(self: *MTFList) void {
        self.root = 0;
        self.index = 0;
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            self.previous[i] = @intCast(@as(i32, @intCast(i)) - 1);
            self.next[i] = @intCast(@as(i32, @intCast(i)) + 1);
        }
        self.next[3] = -1;
    }

    pub fn getFirst(self: *MTFList) i32 {
        self.index = self.root;
        return self.index;
    }

    pub fn getNext(self: *MTFList) i32 {
        if (self.index >= 0) {
            self.index = self.next[@intCast(self.index)];
            return self.index;
        }
        return self.index;
    }

    pub fn moveToFront(self: *MTFList, i: usize) void {
        self.index = @intCast(i);
        if (self.index == self.root) return;
        const p = self.previous[@intCast(self.index)];
        const n = self.next[@intCast(self.index)];
        if (p >= 0) self.next[@intCast(p)] = self.next[@intCast(self.index)];
        if (n >= 0) self.previous[@intCast(n)] = self.previous[@intCast(self.index)];
        self.previous[@intCast(self.root)] = self.index;
        self.next[@intCast(self.index)] = self.root;
        self.root = self.index;
        self.previous[@intCast(self.root)] = -1;
    }
};

pub const SparseMatchModel = struct {
    const Config = struct {
        stride: u32,
        minLen: u32,
    };
    const sparse = [4]Config{
        .{ .stride = 1, .minLen = 3 },
        .{ .stride = 1, .minLen = 4 },
        .{ .stride = 2, .minLen = 6 },
        .{ .stride = 1, .minLen = 5 },
    };

    Table: []u32 = &.{},
    list: MTFList = .{},
    hashes: [4]u32 = [_]u32{0} ** 4,
    hashIndex: u32 = 0,
    length: u32 = 0,
    index: u32 = 0,
    expectedByte: u8 = 0,
    valid: bool = false,

    pub fn deinit(self: *SparseMatchModel, allocator: std.mem.Allocator) void {
        allocator.free(self.Table);
    }

    pub fn init_model(self: *SparseMatchModel, allocator: std.mem.Allocator) !void {
        self.Table = try allocator.alloc(u32, 1024 * 1024);
        @memset(self.Table, 0);
        self.hashIndex = 0;
        self.length = 0;
        self.index = 0;
        self.expectedByte = 0;
        self.valid = false;
        self.list.init_list();
    }

    fn update_model(self: *SparseMatchModel, pos_val: u32, buffer_ptr: []const u8) void {
        const mask = 1024 * 1024 - 1;
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            self.hashes[i] = @intCast((i + 1) * 191);
            var j: u32 = 0;
            var k: u32 = 1;
            while (j < sparse[i].minLen) : ({
                j += 1;
                k += sparse[i].stride;
            }) {
                const b_val = buffer_ptr[(pos_val -% k) & BMASK];
                self.hashes[i] = (self.hashes[i] *% 191) +% (@as(u32, b_val) << @as(u5, @intCast(i)));
            }
            self.hashes[i] &= mask;
        }

        if (self.length != 0) {
            self.index = self.index +% 1;
            if (self.length < 64) {
                self.length += 1;
            }
        } else {
            var list_idx = self.list.getFirst();
            while (list_idx >= 0) : (list_idx = self.list.getNext()) {
                const u_idx = @as(usize, @intCast(list_idx));
                self.index = self.Table[self.hashes[u_idx]];
                if (self.index > 0) {
                    var offset: u32 = 1;
                    while (self.length < sparse[u_idx].minLen and
                        (buffer_ptr[(pos_val -% offset) & BMASK] ^ buffer_ptr[(self.index -% offset) & BMASK]) == 0)
                    {
                        self.length += 1;
                        offset += sparse[u_idx].stride;
                    }
                    if (self.length >= sparse[u_idx].minLen) {
                        self.length -= (sparse[u_idx].minLen - 1);
                        self.hashIndex = @intCast(u_idx);
                        self.list.moveToFront(u_idx);
                        break;
                    }
                }
                self.length = 0;
                self.index = 0;
            }
        }

        i = 0;
        while (i < 4) : (i += 1) {
            self.Table[self.hashes[i]] = pos_val;
        }

        self.expectedByte = buffer_ptr[self.index & BMASK];
        self.valid = self.length > 1;
    }

    pub fn predict(self: *SparseMatchModel, inputs: *Inputs, pos_val: u32, buffer_ptr: []const u8, c0: i32, bpos: i32) u32 {
        const c0b = @as(u8, @intCast(c0 << @as(u3, @intCast(8 - bpos))));
        if (bpos == 0) {
            self.update_model(pos_val, buffer_ptr);
        }

        if (self.length > 0 and (((self.expectedByte ^ c0b) >> @as(u3, @intCast(8 - bpos))) != 0)) {
            self.length = 0;
        }

        if (self.valid) {
            if (self.length > 1) {
                const expectedBit = (self.expectedByte >> @as(u3, @intCast(7 - bpos))) & 1;
                const sign: i32 = if (expectedBit != 0) 1 else -1;
                inputs.add(sign * (@as(i32, @intCast(@min(self.length - 1, 32))) << 5));
                const term1 = @as(i32, 1) << @as(u5, @intCast(@min(self.length - 2, 3)));
                const term2 = @as(i32, @intCast(@min(self.length - 1, 8)));
                inputs.add(sign * term1 * term2 << 4);
            } else {
                inputs.add(0);
                inputs.add(0);
            }
        } else {
            inputs.add(0);
            inputs.add(0);
        }

        return self.length;
    }
};

pub const brackets = [_]u8{ '(', ')', '{', '}', '[', ']', '<', '>' };
pub const quotes = [_]u8{ '\'', '\'', '"', '"' };
pub const fchar = [_]u8{ 64, 10, 96, 10, 'J', 10, '<', '>', 'M', 10, '[', ']', '{', '}', '*', 10, '|', 10, 31, 10 };
pub const html = [_]u16{ '&' * 256 + 'L', '&' * 256 + 'N' };

pub const fcy = init_fcy: {
    var arr = [_]u8{0} ** 128;
    arr['('] = 1;
    arr['['] = 2;
    arr[']'] = 3;
    arr['{'] = 4;
    arr['"'] = 5;
    arr['\''] = 6;
    break :init_fcy arr;
};

pub const fcq = init_fcq: {
    var arr = [_]u8{0} ** 128;
    arr['@'] = 1;
    arr['{'] = 2;
    arr['|'] = 2;
    arr['}'] = 2;
    arr['['] = 3;
    arr[']'] = 4;
    arr['*'] = 5;
    arr[':'] = 6;
    arr[';'] = 7;
    arr['<'] = 7;
    arr['>'] = 7;
    arr['='] = 7;
    arr['?'] = 7;
    break :init_fcq arr;
};

pub const HashElementForMatchPositions = struct {
    matchPositions: [4]u32 = [_]u32{0} ** 4,
    pub fn Add(self: *HashElementForMatchPositions, pos_val: u32) void {
        std.mem.copyBackwards(u32, self.matchPositions[1..4], self.matchPositions[0..3]);
        self.matchPositions[0] = pos_val;
    }
};

pub const MatchInfo = struct {
    length: u32 = 0,
    index: u32 = 0,
    lengthBak: u32 = 0,
    indexBak: u32 = 0,
    expectedByte: u8 = 0,
    delta: bool = false,

    pub fn init_info(self: *MatchInfo) void {
        self.length = 0;
        self.index = 0;
        self.lengthBak = 0;
        self.indexBak = 0;
        self.expectedByte = 0;
        self.delta = false;
    }

    pub fn isInNoMatchMode(self: *const MatchInfo) bool {
        return self.length == 0 and !self.delta and self.lengthBak == 0;
    }

    pub fn isInPreRecoveryMode(self: *const MatchInfo) bool {
        return self.length == 0 and !self.delta and self.lengthBak != 0;
    }

    pub fn isInRecoveryMode(self: *const MatchInfo) bool {
        return self.length != 0 and self.lengthBak != 0;
    }

    pub fn recoveryModePos(self: *const MatchInfo) u32 {
        return self.length - self.lengthBak;
    }

    pub fn prio(self: *const MatchInfo) u32 {
        const cond_norm = if (self.length != 0) @as(u32, 1) << 31 else 0;
        const cond_delta = if (self.delta) @as(u32, 1) << 30 else 0;
        const halflen = if (self.delta) (self.lengthBak >> 1) else (self.length >> 1);
        const cond_len = halflen << 24;
        const cond_recent = self.index & 0x00ffffff;
        return cond_norm | cond_delta | cond_len | cond_recent;
    }

    pub fn isBetterThan(self: *const MatchInfo, other: *const MatchInfo) bool {
        return self.prio() > other.prio();
    }

    pub fn update(self: *MatchInfo, buffer_ptr: []const u8) void {
        if (self.length != 0) {
            const expectedBit = (self.expectedByte >> @as(u3, @intCast((8 - x.bpos) & 7))) & 1;
            if (x.y != expectedBit) {
                if (self.isInRecoveryMode()) {
                    self.lengthBak = 0;
                    self.indexBak = 0;
                } else {
                    self.lengthBak = self.length;
                    self.indexBak = self.index;
                    self.delta = true;
                }
                self.length = 0;
            }
        }

        if (x.bpos == 0) {
            if (self.isInPreRecoveryMode()) {
                self.indexBak = self.indexBak +% 1;
                if (self.lengthBak < 62) {
                    self.lengthBak += 1;
                }
                const c1_val = @as(u8, @intCast(x.c4 & 0xff));
                if (buffer_ptr[self.indexBak & BMASK] == c1_val) {
                    self.length = self.lengthBak;
                    self.index = self.indexBak;
                } else {
                    self.lengthBak = 0;
                    self.indexBak = 0;
                }
            }
            if (self.length != 0) {
                self.index = self.index +% 1;
                if (self.length < 62) {
                    self.length += 1;
                }
                if (self.isInRecoveryMode() and self.recoveryModePos() >= 3) {
                    self.lengthBak = 0;
                    self.indexBak = 0;
                }
            }
            self.delta = false;
        }
    }

    pub fn registerMatch(self: *MatchInfo, pos_val: u32, len: u32) void {
        self.length = len - 5 + 1;
        self.index = pos_val;
        self.lengthBak = 0;
        self.indexBak = 0;
        self.expectedByte = 0;
        self.delta = false;
    }
};

pub var matchCandidates: [4]MatchInfo = [_]MatchInfo{.{}} ** 4;
pub var numberOfActiveCandidates: u32 = 0;
pub var mhashtable: []HashElementForMatchPositions = &.{};
pub var mhashtablemask: u32 = 0;
pub var ctx: [3]u32 = [_]u32{0} ** 3;

pub fn isMMatch(pos_val: u32, min_len: u32, buffer_ptr: []const u8) bool {
    var length: u32 = 1;
    while (length <= min_len) : (length += 1) {
        if (buffer_ptr[(pos - length) & BMASK] != buffer_ptr[(pos_val - length) & BMASK]) {
            return false;
        }
    }
    return true;
}

pub fn addCandidates(matches: *HashElementForMatchPositions, len: u32, buffer_ptr: []const u8) void {
    var i: usize = 0;
    while (numberOfActiveCandidates < 4 and i < 4) : (i += 1) {
        const matchpos = matches.matchPositions[i];
        if (matchpos == 0) break;
        if (isMMatch(matchpos, len, buffer_ptr)) {
            var isSame = false;
            var j: usize = 0;
            while (j < numberOfActiveCandidates) : (j += 1) {
                if (matchCandidates[j].index == matchpos) {
                    isSame = true;
                    break;
                }
            }
            if (!isSame) {
                matchCandidates[numberOfActiveCandidates].registerMatch(matchpos, len);
                numberOfActiveCandidates += 1;
            }
        }
    }
}

pub fn matchModel2update(buffer_ptr: []const u8) void {
    const n = @max(numberOfActiveCandidates, 1);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        var matchInfo = &matchCandidates[i];
        matchInfo.update(buffer_ptr);
        if (numberOfActiveCandidates != 0 and matchInfo.isInNoMatchMode()) {
            numberOfActiveCandidates -= 1;
            if (numberOfActiveCandidates == i) break;
            var k = i;
            while (k < numberOfActiveCandidates) : (k += 1) {
                matchCandidates[k] = matchCandidates[k + 1];
            }
            i -= 1;
        }
    }

    if (x.bpos == 0) {
        var hash_val = t[9];
        var matches = &mhashtable[hash_val & mhashtablemask];
        if (numberOfActiveCandidates < 4) addCandidates(matches, 9, buffer_ptr);
        matches.Add(pos);

        hash_val = t[7];
        matches = &mhashtable[hash_val & mhashtablemask];
        if (numberOfActiveCandidates < 4) addCandidates(matches, 7, buffer_ptr);
        matches.Add(pos);

        hash_val = t[5];
        matches = &mhashtable[hash_val & mhashtablemask];
        if (numberOfActiveCandidates < 4) addCandidates(matches, 5, buffer_ptr);
        matches.Add(pos);

        hash_val = worcxt.word(1);
        matches = &mhashtable[hash_val & mhashtablemask];
        if (numberOfActiveCandidates < 4) addCandidates(matches, 5, buffer_ptr);
        matches.Add(pos);

        i = 0;
        while (i < numberOfActiveCandidates) : (i += 1) {
            matchCandidates[i].expectedByte = buffer_ptr[matchCandidates[i].index & BMASK];
        }
    }
}

pub fn matchModel2mix(inputs: *Inputs, buffer_ptr: []const u8) u32 {
    matchModel2update(buffer_ptr);
    @memset(&ctx, 0);

    var bestCandidateIdx: usize = 0;
    var i: usize = 1;
    while (i < numberOfActiveCandidates) : (i += 1) {
        if (matchCandidates[i].isBetterThan(&matchCandidates[bestCandidateIdx])) {
            bestCandidateIdx = i;
        }
    }

    const length = matchCandidates[bestCandidateIdx].length;
    const expectedByte = matchCandidates[bestCandidateIdx].expectedByte;
    const isInDeltaMode = matchCandidates[bestCandidateIdx].delta;
    const expectedBit = if (length != 0) (expectedByte >> @as(u3, @intCast((7 - x.bpos) & 7))) & 1 else 0;

    var denselength: u32 = 0;
    if (length != 0) {
        if (length <= 16) {
            denselength = length - 1;
        } else {
            denselength = 12 + (length >> 2);
        }
        ctx[0] = (denselength << 4) | (@as(u32, expectedBit) << 3) | @as(u32, @intCast(x.bpos));
        const c1_val = @as(u32, @intCast(x.c4 & 0xff));
        ctx[1] = (@as(u32, expectedByte) << 11) | (@as(u32, @intCast(x.bpos)) << 8) | c1_val;
        const sign: i32 = if (expectedBit != 0) 1 else -1;
        inputs.add(sign * (@as(i32, @intCast(length)) << 5));
    } else {
        inputs.add(0);
    }

    if (isInDeltaMode) {
        ctx[2] = (@as(u32, expectedByte) << 8) | @as(u32, @intCast(x.c0));
    }

    i = 0;
    while (i < 3) : (i += 1) {
        const c_val = ctx[i];
        if (c_val != 0) {
            smA[i].set_cxt(c_val, x.y);
            const p1 = smA[i].pr;
            const st = stretch(p1);
            inputs.add(st >> 2);
            inputs.add((@as(i32, @intCast(p1)) - 2048) >> 3);
        } else {
            inputs.add(0);
            inputs.add(0);
        }
    }
    return length;
}

pub fn getWT(wt_type: u32) u8 {
    if ((wt_type & Verb) != 0) return 1;
    if ((wt_type & Noun) != 0) return 2;
    if ((wt_type & Adjective) != 0) return 3;
    if ((wt_type & Male) != 0) return 4;
    if ((wt_type & Female) != 0) return 5;
    if ((wt_type & Article) != 0) return 6;
    if ((wt_type & Conjunction) != 0) return 7;
    if ((wt_type & Adposition) != 0) return 8;
    if ((wt_type & ConjunctiveAdverb) != 0) return 9;
    if ((wt_type & AdverbOfManner) != 0) return 11;
    if ((wt_type & Suffix) != 0) return 12;
    if ((wt_type & Prefix) != 0) return 13;
    if ((wt_type & Plural) != 0) return 10;
    if (wt_type != 0) return 14;
    return 15;
}

pub fn setbufstem(c: u8) void {
    if ((c >= 'a' and c <= 'z') or (c == '\'' and c2 != '\'') or (c == '-' and StemWords[cWordIdx].length() > 0)) {
        StemWords[cWordIdx].append(c);
    } else if (StemWords[cWordIdx].length() > 0 and (c == ']') and fccxt.cxt != 31 and isParagraph != 0) {
        // no-op
    } else if (StemWords[cWordIdx].length() > 0) {
        _ = StemmerEN.stem(&StemWords[cWordIdx], x.blpos);
        pWordIdx = cWordIdx;
        cWordIdx = (cWordIdx + 1) & 3;
        StemWords[cWordIdx] = .{};

        if ((StemWords[pWordIdx].type & Verb) != 0) {
            sVerb = StemWords[pWordIdx].hash;
        }

        if (lastArt) {
            StemWords[pWordIdx].type |= Noun;
        }
        if (StemWords[pWordIdx].type == Article and buffer1(5) == ' ' and buffer1(4) == 't' and buffer1(3) == 'h' and buffer1(2) == 'e') {
            lastArt = true;
        } else {
            lastArt = false;
        }
        var whash = if (isMath) word0 else StemWords[pWordIdx].hash;
        lastWT = (lastWT *% 16) +% getWT(StemWords[pWordIdx].type);
        if (StemWords[pWordIdx].type == Number and worcxt.type_val(1) == Number) {
            const sb = worcxt.sBytes(1);
            whash = whash +% worcxt.word(1);
            worcxt.remove();
            worcxt.set_pbyte(@intCast(sb >> 8), 0);
        }
        const c1_val = @as(u8, @intCast(x.c4 & 0xff));
        worcxt.update(word0, c1_val, StemWords[pWordIdx].type, whash);
        if (((StemWords[pWordIdx].type & (Conjunction | Article | Male | Female | Number | ConjunctiveAdverb)) == 0) and brcxt.cxt != '<') {
            worcxt1.update(word0, c1_val, StemWords[pWordIdx].type, whash);
        }
        if (((StemWords[pWordIdx].type & (Conjunction | Article | Male | Female | Adposition | Number | AdverbOfManner | ConjunctiveAdverb)) == 0) and brcxt.cxt != '<') {
            if (StemWords[pWordIdx].type != 0) {
                worcxt2.update(word0, c1_val, StemWords[pWordIdx].type, whash);
            }
        }
    }
}

pub fn setbuf(c: u8) void {
    cwbuf[cwpos & CBMASK] = c;
    cwpos = (cwpos +% 1) & 0xffffffff;
    setbufstem(c);
}

pub fn procWord() void {
    if (dcwl > 0) {
        var dcw2: u32 = 0;
        if (dcwl == 2) {
            dcw2 = (dcw >> 8) | ((dcw & 255) << 8);
        } else if (dcwl == 3) {
            dcw2 = (dcw >> 16) | (dcw & 0xff00) | ((dcw & 255) << 16);
        }
        decodeWord(dcw2);
        dcw = 0;
        dcwl = 0;
        for (so) |ch| {
            setbuf(ch);
        }
    }
}

pub var pos: u32 = 0;
pub var buffer: []u8 = &.{};
pub const BMASK: u32 = 0xffffff;
pub var cwbuf: [0x1000]u8 = [_]u8{0} ** 0x1000;
pub const CBMASK: u32 = 0xfff;
pub var cwpos: u32 = 0;

pub fn buf(i: u32) u8 {
    return buffer[(pos -% i) & BMASK];
}
pub fn bufr(i: u32) u8 {
    return buffer[i & BMASK];
}
pub fn buffer1(i: u32) u8 {
    return cwbuf[(cwpos -% i) & CBMASK];
}

pub var c1: u8 = 0;
pub var c2: u8 = 0;
pub var c3: u8 = 0;
pub var words: u8 = 0;
pub var spaces: u8 = 0;
pub var numbers: u8 = 0;
pub var word0: u32 = 0;
pub var word00: u32 = 0;
pub var word1: u32 = 0;
pub var word2: u32 = 0;
pub var word3: u32 = 0;
pub var wshift: u32 = 0;
pub var x4: u32 = 0;
pub var x5: u32 = 0;
pub var isMatch: u32 = 0;
pub var firstWord: u32 = 0;
pub var linkword: u32 = 0;
pub var senword: u32 = 0;
pub var number0: u32 = 0;
pub var number1: u32 = 0;
pub var numlen0: u32 = 0;
pub var numlen1: u32 = 0;
pub var mybenum: u32 = 0;
pub var FcIdx: u32 = 0;
pub var BrFcIdx: u32 = 0;
pub var AH1: u32 = 0;
pub var AH2: u32 = 0x765BA55C;
pub var fails: u32 = 0;
pub var failz: u32 = 0;
pub var failcount: u32 = 0;
pub var nl: u32 = 0;
pub var nl1: u32 = 0;
pub var col: i32 = 0;
pub var fc: u32 = 0;
pub var t1: [0x100]u32 = [_]u32{0} ** 0x100;
pub var t2: [0x10000]u32 = [_]u32{0} ** 0x10000;
pub var wp: [0x10000]u32 = [_]u32{0} ** 0x10000;
pub var ind3: []u16 = &.{};
pub var indirectBrByte: u32 = 0;
pub var indirectByte: u32 = 0;
pub var indirectWord0Pos: u32 = 0;
pub var indirectWord: u32 = 0;
pub var u8w: u32 = 0;
pub var context1_ind3: u32 = 0;
pub var cxtind3: u32 = 0;
pub var lastWT: u32 = 0;
pub var o3bState: u32 = 0;
pub var n3bState: u32 = 0;
pub var stream3bR: u32 = 0;
pub var stream3b: u32 = 0;
pub var stream3bMask: u32 = 0;
pub var stream3bMask1: u32 = 0;
pub var stream3bRMask1: u32 = 0;
pub var stream3bRMask2: u32 = 0;
pub var o2bState: u32 = 0;
pub var n2bState: u32 = 0;
pub var stream2bR: u32 = 0;
pub var stream2b: u32 = 0;
pub var stream2bMask: u32 = 0;
pub var o4bState: u32 = 0;
pub var n4bState: u32 = 0;
pub var stream4bR: u32 = 0;
pub var stream4b: u32 = 0;
pub var ordX: i32 = 0;
pub var ordW: i32 = 0;
pub var StemWords: [4]Word = [_]Word{.{}} ** 4;
pub var cWordIdx: usize = 0;
pub var pWordIdx: usize = 3;
pub var StemmerEN: EnglishStemmer = .{};
pub var dcw: u32 = 0;
pub var dcwl: u32 = 0;
pub var sVerb: u32 = 0;
pub var lastArt: bool = false;
pub var isNowiki: bool = false;
pub var colonstr: []const u8 = "";
pub var deccode: u32 = 0;
pub var wrtcxt: u64 = 0;
pub var isText: bool = false;
pub var utf8left: u32 = 0;
pub var pr: i32 = 2048;

pub var smA: [3]StateMap1 = [_]StateMap1{.{}} ** 3;
pub var scmA: [7]SmallStationaryContextMap = [_]SmallStationaryContextMap{.{}} ** 7;
pub var mxA: [12]Mixer1 = [_]Mixer1{.{}} ** 12;
pub var cmC: [6]ContextMap = [_]ContextMap{.{}} ** 6;
pub var cmC1: [8]ContextMap1 = [_]ContextMap1{.{}} ** 8;
pub var cmC2: [18]ContextMap2 = [_]ContextMap2{.{}} ** 18;

pub fn APM(comptime S: usize) type {
    return struct {
        const Self = @This();
        index: usize = 0,
        t: [S * 33]u16 = [_]u16{0} ** (S * 33),

        pub fn p(self: *Self, pr_val: i32, cxt: i32, rate_val: i32, y: u1) i32 {
            const stretched_pr = stretch(@intCast(pr_val));
            const g = (@as(i32, y) << 16) + (@as(i32, y) << @as(u5, @intCast(rate_val))) - @as(i32, y) * 2;
            self.t[self.index] = @intCast(@as(i32, self.t[self.index]) +% ((g - @as(i32, self.t[self.index])) >> @as(u5, @intCast(rate_val))));
            self.t[self.index + 1] = @intCast(@as(i32, self.t[self.index + 1]) +% ((g - @as(i32, self.t[self.index + 1])) >> @as(u5, @intCast(rate_val))));
            const w = stretched_pr & 127; // interpolation weight (33 points)
            self.index = @intCast(((stretched_pr + 2048) >> 7) + cxt * 33);
            return @intCast((@as(i32, self.t[self.index]) * (128 - w) + @as(i32, self.t[self.index + 1]) * w) >> 11);
        }

        pub fn init_apm(self: *Self) void {
            self.index = 0;
            var j: usize = 0;
            while (j < 33) : (j += 1) {
                self.t[j] = @intCast(squash((@as(i32, @intCast(j)) - 16) * 128) * 16);
            }
            var i: usize = 33;
            while (i < S * 33) : (i += 1) {
                self.t[i] = self.t[i - 33];
            }
        }
    };
}

pub var apmA0: APM(256) = .{};
pub var apmA1: APM(0x8000 * 2) = .{};
pub var apmA2: APM(0x8000 * 2) = .{};
pub var apmA3: APM(0x20000 * 2) = .{};
pub var apmA4: APM(0x20000 * 2) = .{};
pub var apmA5: APM(0x20000 * 2) = .{};
pub var rcmA: [1]RunContextMap = [_]RunContextMap{.{}} ** 1;
pub var brcxt: BracketContext(u8) = .{};
pub var qocxt: BracketContext(u8) = .{};
pub var fccxt: BracketContext(u8) = .{};
pub var colcxt: ColumnContext = .{};
pub var worcxt: WordsContext = .{};
pub var worcxt1: WordsContext = .{};
pub var worcxt2: WordsContext = .{};
pub var htcxt: BracketContext(u16) = .{};
pub var smatch: SparseMatchModel = .{};

pub var sscmrate: i32 = 0;
pub var isMath: bool = false;
pub var isPre: bool = false;
pub var isParagraph: i32 = 0;
pub var t: [14]u32 = [_]u32{0} ** 14;

pub const Fxcm = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) !Fxcm {
        initIlog();

        buffer = try allocator.alloc(u8, 0x1000000);
        @memset(buffer, 0);
        ind3 = try allocator.alloc(u16, 0x2000000);
        @memset(ind3, 0);

        n3bState = 0xffffffff;
        n2bState = 0xffffffff;
        pr = 2048;

        try smA[0].init_map(allocator, 1 << 9, 1023);
        try smA[1].init_map(allocator, 1 << 19, 1023);
        try smA[2].init_map(allocator, 1 << 16, 1023);

        try scmA[0].init_map(allocator, 8, 8);
        try scmA[1].init_map(allocator, 8, 8);
        try scmA[2].init_map(allocator, 8, 8);
        try scmA[3].init_map(allocator, 9, 8);
        try scmA[4].init_map(allocator, 8, 8);
        try scmA[5].init_map(allocator, 8, 8);
        try scmA[6].init_map(allocator, 7, 8);

        mxA[0].init_mixer(2048, 237, 8, 69);
        mxA[1].init_mixer(6 * 256, 204, 8, 19);
        mxA[2].init_mixer(6 * 256 * 4, 70, 1, 34);
        mxA[3].init_mixer(8 * 256, 54, 1, 23);
        mxA[4].init_mixer(6 * 256, 55, 1, 24);
        mxA[5].init_mixer(7 * 256 * 4, 55, 1, 24);
        mxA[6].init_mixer(0x4000, 70, 1, 34);
        mxA[7].init_mixer(0x4000, 55, 1, 24);
        mxA[8].init_mixer(0x20000, 55, 1, 24);
        mxA[9].init_mixer(0x20000, 55, 1, 24);
        mxA[10].init_mixer(8 * 7 * 2 * 2, 6, 0, 4);
        mxA[11].init_mixer(1, 6, 0, 4);

        apmA0.init_apm();
        apmA1.init_apm();
        apmA2.init_apm();
        apmA3.init_apm();
        apmA4.init_apm();
        apmA5.init_apm();
        try rcmA[0].init_map(allocator, 1 * 4096 * 4096, 6);

        x.init();

        try mxA[0].setTxWx(allocator, x.mxInputs1.ncount, &x.mxInputs1.n);
        try mxA[1].setTxWx(allocator, x.mxInputs1.ncount, &x.mxInputs1.n);
        try mxA[2].setTxWx(allocator, x.mxInputs1.ncount, &x.mxInputs1.n);
        try mxA[3].setTxWx(allocator, x.mxInputs1.ncount, &x.mxInputs1.n);
        try mxA[4].setTxWx(allocator, x.mxInputs1.ncount, &x.mxInputs1.n);
        try mxA[5].setTxWx(allocator, x.mxInputs1.ncount, &x.mxInputs1.n);
        try mxA[6].setTxWx(allocator, x.mxInputs1.ncount, &x.mxInputs1.n);
        try mxA[7].setTxWx(allocator, x.mxInputs1.ncount, &x.mxInputs1.n);
        try mxA[8].setTxWx(allocator, x.mxInputs1.ncount, &x.mxInputs1.n);
        try mxA[9].setTxWx(allocator, x.mxInputs1.ncount, &x.mxInputs1.n);

        try mxA[10].setTxWx(allocator, x.mxInputs2.ncount, &x.mxInputs2.n);
        try mxA[11].setTxWx(allocator, x.mxInputs2.ncount, &x.mxInputs2.n);

        var statetable = StateTable{};
        statetable.init_table(28, 28, 31, 29, 23, 4, 17, std.mem.sliceAsBytes(&STA1));
        statetable.init_table(32, 28, 31, 28, 21, 5, 6, std.mem.sliceAsBytes(&STA2));
        statetable.init_table(31, 27, 30, 27, 24, 4, 27, std.mem.sliceAsBytes(&STA4));
        statetable.init_table(33, 31, 31, 24, 20, 4, 33, std.mem.sliceAsBytes(&STA5));
        statetable.init_table(28, 29, 30, 30, 23, 3, 22, std.mem.sliceAsBytes(&STA6));
        statetable.init_table(28, 29, 33, 23, 23, 6, 14, std.mem.sliceAsBytes(&STA7));

        var idx: usize = 0;
        while (idx < 256) : (idx += 1) {
            const n0 = @as(u32, STA7[idx][2]) * 3 + 1;
            const n1 = @as(u32, STA7[idx][3]) * 3 + 1;
            const div = (n1 << 12) / (n0 + n1);
            pre1[idx] = @intCast(clp(stretch(div)) >> 2);
        }

        try cmC2[0].init_map(allocator, 8 * 4096 * 4096, @intCast(3 | (c_r[0] << 8) | (c_s[0] << 16)), @intCast(c_s3[0]), std.mem.sliceAsBytes(&STA6), @intCast(c_s4[0]), 0xf0, 1, &st2_p1);
        try cmC2[1].init_map(allocator, 16 * 4096 * 4096, @intCast(1 | (c_r[1] << 8) | (c_s[1] << 16)), @intCast(c_s3[1]), std.mem.sliceAsBytes(&STA6), @intCast(c_s4[1]), 0xf0, 1, &st2_p1);
        try cmC2[2].init_map(allocator, 8 * 4096 * 4096, @intCast(1 | (c_r[2] << 8) | (c_s[2] << 16)), @intCast(c_s3[2]), std.mem.sliceAsBytes(&STA6), @intCast(c_s4[2]), 0xf0, 1, &st2_p1);
        try cmC2[3].init_map(allocator, 8 * 4096 * 4096, @intCast(1 | (c_r[3] << 8) | (c_s[3] << 16)), @intCast(c_s3[3]), std.mem.sliceAsBytes(&STA6), @intCast(c_s4[3]), 0xf0, 1, &st2_p1);
        try cmC2[4].init_map(allocator, 8 * 4096 * 4096, @intCast(2 | (c_r[4] << 8) | (c_s[4] << 16)), @intCast(c_s3[4]), std.mem.sliceAsBytes(&STA6), @intCast(c_s4[4]), 0xf0, 1, &st2_p1);
        try cmC2[5].init_map(allocator, 8 * 4096 * 4096, @intCast(6 | (c_r[5] << 8) | (c_s[5] << 16)), @intCast(c_s3[5]), std.mem.sliceAsBytes(&STA6), @intCast(c_s4[5]), 0xf0, 1, &st2_p1);
        try cmC2[6].init_map(allocator, 1 * 4096 * 4096 / 64, @intCast(1 | (c_r[6] << 8) | (c_s[6] << 16)), @intCast(c_s3[6]), std.mem.sliceAsBytes(&STA1), @intCast(c_s4[6]), 0, 1, &st2_p1);
        try cmC2[7].init_map(allocator, 2 * 4096 * 4096, @intCast(1 | (c_r[7] << 8) | (c_s[7] << 16)), @intCast(c_s3[7]), std.mem.sliceAsBytes(&STA5), @intCast(c_s4[7]), 0xf0, 1, &st2_p1);
        try cmC2[8].init_map(allocator, 8 * 4096 * 4096 / 2, @intCast(4 | (c_r[8] << 8) | (c_s[8] << 16)), @intCast(c_s3[8]), std.mem.sliceAsBytes(&STA4), @intCast(c_s4[8]), 0, 1, &st2_p1);

        try cmC1[0].init_map(allocator, 32 * 4096, @intCast(2 | (c_r[9] << 8) | (c_s[9] << 16)), @intCast(c_s3[9]), std.mem.sliceAsBytes(&STA6), @intCast(c_s4[9]), 0x0, 0, &st2_p0);
        try cmC1[1].init_map(allocator, 2 * 32 * 4096, @intCast(3 | (c_r[10] << 8) | (c_s[10] << 16)), @intCast(c_s3[10]), std.mem.sliceAsBytes(&STA7), @intCast(c_s4[10]), 0, 1, &st2_p1);
        try cmC1[2].init_map(allocator, 32 * 4096, @intCast(4 | (c_r[11] << 8) | (c_s[11] << 16)), @intCast(c_s3[11]), std.mem.sliceAsBytes(&STA2), @intCast(c_s4[11]), 0, 1, &st2_p1);
        try cmC1[4].init_map(allocator, 16 * 4096, @intCast(5 | (c_r[12] << 8) | (c_s[12] << 16)), @intCast(c_s3[12]), std.mem.sliceAsBytes(&STA7), @intCast(c_s4[12]), 0, 1, &st2_p1);

        try cmC[0].init_map(allocator, 16 * 4096, @intCast(7 | (c_r[13] << 8) | (c_s[13] << 16)), @intCast(c_s3[13]), std.mem.sliceAsBytes(&STA2), @intCast(c_s4[13]), 0, 1, &st2_p1);
        try cmC[1].init_map(allocator, 64 * 2 * 4096, @intCast(3 | (c_r[14] << 8) | (c_s[14] << 16)), @intCast(c_s3[14]), std.mem.sliceAsBytes(&STA5), @intCast(c_s4[14]), 0xf0, 0, &st2_p0);
        try cmC[2].init_map(allocator, 2 * 4096, @intCast(2 | (c_r[15] << 8) | (c_s[15] << 16)), @intCast(c_s3[15]), std.mem.sliceAsBytes(&STA2), @intCast(c_s4[15]), 0xf0, 0, &st2_p0);

        try cmC1[3].init_map(allocator, 128 * 4096, @intCast(2 | (c_r[16] << 8) | (c_s[16] << 16)), @intCast(c_s3[16]), std.mem.sliceAsBytes(&STA1), @intCast(c_s4[16]), 0, 0, &st2_p0);
        try cmC2[9].init_map(allocator, 8 * 4096 * 4096, @intCast(4 | (c_r[17] << 8) | (c_s[17] << 16)), @intCast(c_s3[17]), std.mem.sliceAsBytes(&STA6), @intCast(c_s4[17]), 0xf0, 1, &st2_p1);
        try cmC2[10].init_map(allocator, 8 * 4096 * 4096, @intCast(6 | (c_r[18] << 8) | (c_s[18] << 16)), @intCast(c_s3[18]), std.mem.sliceAsBytes(&STA5), @intCast(c_s4[18]), 0xf0, 1, &st2_p1);
        try cmC2[11].init_map(allocator, 8 * 4096 * 4096, @intCast(5 | (c_r[19] << 8) | (c_s[19] << 16)), @intCast(c_s3[19]), std.mem.sliceAsBytes(&STA5), @intCast(c_s4[19]), 0xf0, 1, &st2_p1);
        try cmC2[12].init_map(allocator, 8 * 4096 * 4096, @intCast(2 | (c_r[20] << 8) | (c_s[20] << 16)), @intCast(c_s3[20]), std.mem.sliceAsBytes(&STA6), @intCast(c_s4[20]), 0xf0, 1, &st2_p1);
        try cmC2[13].init_map(allocator, 16 * 4096 * 4096, @intCast(2 | (c_r[21] << 8) | (c_s[21] << 16)), @intCast(c_s3[21]), std.mem.sliceAsBytes(&STA6), @intCast(c_s4[21]), 0xf0, 1, &st2_p1);

        try cmC[3].init_map(allocator, 32 * 4096, @intCast(2 | (c_r[22] << 8) | (c_s[22] << 16)), @intCast(c_s3[22]), std.mem.sliceAsBytes(&STA2), @intCast(c_s4[22]), 0x00, 1, &st2_p2);

        try cmC2[14].init_map(allocator, 4 * 4096 * 4096 / 2, @intCast(1 | (c_r[23] << 8) | (c_s[23] << 16)), @intCast(c_s3[23]), std.mem.sliceAsBytes(&STA6), @intCast(c_s4[23]), 0xf0, 1, &st2_p1);
        try cmC2[15].init_map(allocator, 8 * 64 * 4096, @intCast(1 | (c_r[24] << 8) | (c_s[24] << 16)), @intCast(c_s3[24]), std.mem.sliceAsBytes(&STA1), @intCast(c_s4[24]), 0, 0, &st2_p0);

        try cmC[4].init_map(allocator, 512 * 4096, @intCast(1 | (c_r[25] << 8) | (c_s[25] << 16)), @intCast(c_s3[25]), std.mem.sliceAsBytes(&STA1), @intCast(c_s4[25]), 0xf0, 1, &st2_p1);
        try cmC[5].init_map(allocator, 512 * 4096, @intCast(1 | (c_r[26] << 8) | (c_s[26] << 16)), @intCast(c_s3[26]), std.mem.sliceAsBytes(&STA1), @intCast(c_s4[26]), 0xf0, 1, &st2_p1);

        try cmC2[16].init_map(allocator, 1 * 4096 * 4096 / 2, @intCast(1 | (c_r[17] << 8) | (c_s[17] << 16)), @intCast(c_s3[17]), std.mem.sliceAsBytes(&STA6), @intCast(c_s4[17]), 0xf0, 1, &st2_p1);
        try cmC2[17].init_map(allocator, 2 * 4096 * 4096, @intCast(2 | (c_r[17] << 8) | (c_s[17] << 16)), @intCast(c_s3[17]), std.mem.sliceAsBytes(&STA6), @intCast(c_s4[17]), 0xf0, 1, &st2_p1);

        try cmC1[6].init_map(allocator, 1 * 16 * 4096, @intCast(1 | (c_r[5] << 8) | (c_s[5] << 16)), @intCast(c_s3[5]), std.mem.sliceAsBytes(&STA6), @intCast(c_s4[5]), 0, 0, &st2_p1);
        try cmC1[7].init_map(allocator, 16 * 4096, @intCast(4 | (c_r[12] << 8) | (c_s[12] << 16)), @intCast(c_s3[12]), std.mem.sliceAsBytes(&STA2), @intCast(c_s4[12]), 0, 1, &st2_p1);

        brcxt.init_ctx(&brackets, false, 0);
        qocxt.init_ctx(&quotes, true, 0);
        fccxt.init_ctx(&fchar, false, 0);
        colcxt.init_ctx(31);
        worcxt.init_ctx();
        worcxt1.init_ctx();
        worcxt2.init_ctx();
        htcxt.init_ctx(&html, false, 0xfff);

        try smatch.init_model(allocator);

        mhashtablemask = 0x200000 * 1 - 1;
        mhashtable = try allocator.alloc(HashElementForMatchPositions, 0x200000 * 1 + 32);
        @memset(mhashtable, std.mem.zeroes(HashElementForMatchPositions));

        cWordIdx = 0;
        pWordIdx = 3;
        so = "";
        colonstr = so;

        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Fxcm) void {
        self.allocator.free(buffer);
        self.allocator.free(ind3);
        self.allocator.free(mhashtable);
        for (&smA) |*sm| sm.deinit(self.allocator);
        for (&scmA) |*scm| scm.deinit(self.allocator);
        for (&mxA) |*mx| mx.deinit(self.allocator);
        for (&cmC) |*cm| cm.deinit(self.allocator);
        for (&cmC1) |*cm| cm.deinit(self.allocator);
        for (&cmC2) |*cm| cm.deinit(self.allocator);
        rcmA[0].deinit(self.allocator);
        smatch.deinit(self.allocator);
    }

    pub fn predict(self: *Fxcm, lstmpr_val: i32, lstmex_val: i32) f32 {
        _ = self;
        resetPredictions();
        wrtcxt = deccode;
        mxA[8].cxt = @intCast(deccode);

        const c0b = x.c0 << @as(u3, @intCast(8 - x.bpos));

        scmA[0].mix(&x.mxInputs1, sscmrate, x.y);
        scmA[1].mix(&x.mxInputs1, sscmrate, x.y);
        scmA[2].mix(&x.mxInputs1, sscmrate, x.y);
        scmA[3].mix(&x.mxInputs1, sscmrate, x.y);
        scmA[4].mix(&x.mxInputs1, sscmrate, x.y);
        scmA[5].mix(&x.mxInputs1, sscmrate, x.y);
        scmA[6].mix(&x.mxInputs1, sscmrate, x.y);

        isMatch = matchModel2mix(&x.mxInputs1, buffer);
        _ = smatch.predict(&x.mxInputs1, pos, buffer, x.c0, x.bpos);

        ordX = 0;
        if (cmC2[0].cxtMask != 0) ordX = 2;
        ordX = ordX + cmC2[0].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        if (ordX == 3) ordX = 2;

        ordX = ordX + cmC2[1].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        ordX = ordX + cmC2[2].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        ordX = ordX + cmC2[3].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        ordW = cmC2[4].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        ordW = ordW + cmC2[5].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        if (ordW > 3) ordW = 3;
        _ = cmC2[6].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC2[7].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC2[8].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC1[0].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC1[1].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC1[2].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC1[4].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC[0].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC[1].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC[2].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC1[3].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC2[9].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC2[10].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC2[11].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC2[12].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);

        ordW = ordW + cmC2[13].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC[3].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        ordW = ordW + cmC2[14].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC2[15].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC[4].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC[5].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC2[16].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC2[17].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC1[6].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = cmC1[7].mix(&x.mxInputs1, x.c0, x.bpos, @intCast(c1), @intCast(x.c4 & 0xff), x.y);
        _ = rcmA[0].get_pred(cc_shift_bpos(x.c0, x.bpos), x.bposshift);

        addPrediction(squash(64));

        if (x.bpos == 0) {
            mxA[0].cxt = @intCast((stream2b & 255) * 8 + (stream3b & 7));
        } else if (x.bpos > 3) {
            const c_val = wrt_2b[@intCast(c0b & 255)];
            mxA[0].cxt = @intCast((((stream2b << 2) & 255) + c_val) * 8 + BrFcIdx);
        } else {
            mxA[0].cxt = @intCast((stream2b & 255) * 8 + BrFcIdx);
        }

        var c_val: i32 = 0;
        if (x.bpos != 0) {
            c_val = @intCast(c0b);
            if (x.bpos == 1) {
                c_val = c_val + @as(i32, @intCast(16 * (words * 2 & 4)));
            } else if (x.bpos > 3) {
                c_val = @as(i32, @intCast(wrt_2b[@intCast(c0b & 255)])) * 64;
            }
            c_val = (if (x.bpos < 5) x.bpos else 5) * 256 + @as(i32, @intCast(stream3bR & 7)) + @as(i32, @intCast(FcIdx * 8)) + (c_val & 192);
        } else {
            c_val = @as(i32, @intCast((words & 12) * 16)) + @as(i32, @intCast(stream3bR & 7)) + @as(i32, @intCast(BrFcIdx * 8));
        }
        mxA[1].cxt = @intCast(c_val);

        mxA[2].cxt = @intCast(((4 * @as(i32, @intCast(words))) & 0xf0) * 4 + ordX * 256 * 4 + @as(i32, @intCast(stream2b & 63)));
        mxA[6].cxt = @intCast((@as(i32, @intCast(stream3bR)) & 0xff8) * 4 + ((2 * @as(i32, @intCast(words))) & 0x1c) + @as(i32, @intCast(stream2b & 3)));

        mxA[3].cxt = @intCast(x.bpos * 256 + ((((~@as(i32, @intCast(numbers | words)) << @as(u3, @intCast(x.bpos))) & 255) >> @as(u3, @intCast(x.bpos))) | @as(i32, @intCast(c0b))));

        mxA[10].cxt = @intCast((ordX * 8 + @as(i32, @intFromBool(BrFcIdx != 0)) * 4 + @as(i32, @intCast(stream2b & 3))) * 2 + (words & 1));

        c_val = @intCast(c0b);
        if (x.bpos != 0) {
            if (x.bpos == 1) {
                c_val = c_val + @as(i32, @intCast(16 * (stream3b & 7)));
            } else if (x.bpos == 2) {
                c_val = c_val + @as(i32, @intCast(16 * (stream2b & 3)));
            } else if (x.bpos == 3) {
                c_val = c_val + @as(i32, @intCast(16 * (words & 1)));
            } else {
                c_val = x.bpos + (c_val & 0xf0);
            }
            if (x.bpos < 5) {
                c_val = x.bpos + (c_val & 0xf0);
            }
        } else {
            c_val = 16 * @as(i32, @intCast(stream2b & 0xf));
        }
        var ordX_term = ordX - 1;
        if (ordX_term < 0) ordX_term = 0;
        if (isMatch != 0) ordX_term += 1;
        mxA[4].cxt = @intCast(c_val + ordX_term * 256 + 8 * isParagraph);

        mxA[5].cxt = @intCast((ordW * 256 + @as(i32, @intCast(stream2b & 0xf0)) + @as(i32, @intCast((stream3b & 0x38) >> 2))) * 4 + @as(i32, @intCast(FcIdx)));

        if (x.bpos > 2) {
            mxA[7].cxt = @intCast((@as(i32, @intCast(stream3b & 7)) * 8 + @as(i32, @intCast(wrt_3b[@intCast(c0b & 255)]))) * 256 + @as(i32, @intCast(BrFcIdx * 32)) + @as(i32, @intCast((words & 7) * 4)) + isParagraph + if (isMatch != 0) @as(i32, 2) else @as(i32, 0));
        } else {
            mxA[7].cxt = @intCast((@as(i32, @intCast(stream3b & 63)) * 256 + @as(i32, @intCast(BrFcIdx * 16)) + @as(i32, @intCast((words & 7) * 2)) + isParagraph) | if (isMatch != 0) @as(i32, 128) else @as(i32, 0));
        }

        mxA[9].cxt = @intCast((x.bpos << 8) * 4 + @as(i32, @intCast(fails & 3)) * 256 + lstmex_val);

        x.mxInputs1.add(stretch(@intCast(lstmpr_val)));
        prediction_index -= 1;
        x.mxInputs2.add(@intCast(mxA[0].p1()));
        x.mxInputs2.add(@intCast(mxA[1].p1()));
        x.mxInputs2.add(@intCast(mxA[2].p1()));
        x.mxInputs2.add(@intCast(mxA[3].p1()));
        x.mxInputs2.add(@intCast(mxA[4].p1()));
        x.mxInputs2.add(@intCast(mxA[5].p1()));
        x.mxInputs2.add(@intCast(mxA[6].p1()));
        x.mxInputs2.add(@intCast(mxA[7].p1()));
        x.mxInputs2.add(@intCast(mxA[8].p1()));
        x.mxInputs2.add(@intCast(mxA[9].p1()));
        x.mxInputs2.add(@intCast(@divTrunc(stretch(@intCast(lstmpr_val)), 2)));
        prediction_index -= 1;

        const final_p = @as(f32, @floatFromInt(squash((mxA[10].p1() * 7 + mxA[11].p1() + 4) >> 3))) * (1.0 / 4095.0);

        if (x.blpos < 5 and x.bpos == 0) {
            std.debug.print("[DEBUG FXCM] predict. prediction_index={d}, final_p={d:.4}, first_pred={d:.4}, last_pred={d:.4}\n", .{
                prediction_index,
                final_p,
                model_predictions[0],
                model_predictions[430],
            });
        }

        return final_p;
    }

    pub fn perceive(self: *Fxcm, bit: u1, lstmpr_val: i32, lstmex_val: i32) void {
        x.y = bit;

        x.c0 += x.c0 + x.y;
        if (x.c0 >= 256) {
            x.c4 = (x.c4 << 8) + @as(u32, @intCast(x.c0 & 0xff));
            x.c0 = 1;
            x.blpos += 1;
            if ((fails & 255) == 0) {
                var i: usize = 0;
                while (i < 10) : (i += 1) {
                    mxA[i].elim = @max(256, mxA[i].elim + 1);
                }
            } else {
                var i: usize = 0;
                while (i < 10) : (i += 1) {
                    mxA[i].elim = @max(0, @min(16, mxA[i].elim - 1));
                }
            }
            sscmrate = @intFromBool(x.blpos > 14 * 256 * 1024);
            rate = 6 + @as(u5, @intCast(@intFromBool(x.blpos > 14 * 256 * 1024))) + @as(u5, @intCast(@intFromBool(x.blpos > 28 * 512 * 1024)));
        }

        x.bpos = (x.bpos + 1) & 7;
        x.bposshift = @intCast(7 - x.bpos);
        x.c0shift_bpos = (x.c0 << 1) ^ (@as(i32, 256) >> x.bposshift);

        mxA[0].update(x.y);
        mxA[1].update(x.y);
        mxA[2].update(x.y);
        mxA[3].update(x.y);
        mxA[4].update(x.y);
        mxA[5].update(x.y);
        mxA[6].update(x.y);
        mxA[7].update(x.y);
        mxA[8].update(x.y);
        mxA[9].update(x.y);
        mxA[10].update(x.y);
        mxA[11].update(x.y);

        x.mxInputs1.ncount = 0;
        x.mxInputs2.ncount = 0;

        if ((fails & 0x80) != 0) failcount -= 1;
        fails = fails *% 2;
        failz = failz *% 2;

        if (x.y != 0) pr = 4095 - pr;
        if (pr >= e_l[@intCast(x.bpos)]) {
            fails += 1;
            failcount += 1;
        }
        if (pr >= 848) failz += 1;

        pr = @intFromFloat(self.predict(lstmpr_val, lstmex_val) * 4095.0);
        addPrediction(pr);

        var pu = (apmA0.p(pr, @intCast(x.c0), 3, x.y) +% 7 *% pr +% 4) >> 3;
        var pv: i32 = 0;
        var pz = failcount + 1;

        pz += tri[(fails >> 5) & 3];
        pz += trj[(fails >> 3) & 3];
        pz += trj[(fails >> 1) & 3];
        if ((fails & 1) != 0) pz += 8;
        pz = pz / 2;

        pu = apmA3.p(pu, @intCast(((@as(u32, @intCast(x.c0)) * 2) ^ AH1) & 0x3ffff), rate, x.y);
        addPrediction(@intCast(pu));

        pv = apmA1.p(@intCast(pr), @intCast(((@as(u32, @intCast(x.c0)) * 8) ^ hash(29, failz & 2047, 0)) & 0xffff), rate + 1, x.y);
        addPrediction(@intCast(pv));

        if ((fails & 255) != 0) {
            pv = apmA4.p(pv, @intCast(hash(@intCast(x.c0), stream2b & 0xfffc, stream3bR & 0x1ff) & 0x3ffff), rate, x.y);
        } else {
            pv = apmA4.p(pv, @intCast(hash(@intCast(x.c0), (stream2bR & 0xfffc) + 0x10000, stream3bR & 0x1ff) & 0x3ffff), rate, x.y);
        }
        addPrediction(@intCast(pv));

        const pt = apmA2.p(@intCast(pr), @intCast(((@as(u32, @intCast(x.c0)) * 32) ^ AH2) & 0xffff), rate, x.y);
        addPrediction(@intCast(pt));

        pz = @intCast(apmA5.p(pu, @intCast(((@as(u32, @intCast(x.c0)) * 4) ^ hash(pz, x5 & 0x80ff, 0)) & 0x3ffff), rate, x.y));
        addPrediction(@intCast(pz));

        if ((fails & 255) != 0) {
            pr = @intCast((pt *% 6 +% pu +% pv *% 11 +% @as(i32, @intCast(pz)) *% 14 +% 31) >> 5);
        } else {
            pr = @intCast((pt *% 4 +% pu *% 5 +% pv *% 12 +% @as(i32, @intCast(pz)) *% 11 +% 31) >> 5);
        }
        addPrediction(pr);

        if (x.bpos == 0) {
            c3 = c2;
            c2 = c1;
            c1 = @intCast(x.c4 & 0xff);

            n2bState = wrt_2b[c1];
            n3bState = wrt_3b[c1];
            n4bState = wrt_4b[c1];

            stream2b = stream2b *% 4 +% n2bState;
            stream4b = stream4b *% 16 +% n4bState;
            buffer[pos & BMASK] = c1;
            pos = pos +% 1;

            if (c2 == '>' and isText) {
                isText = false;
                if (c1 == '\'' or c1 == 64) {
                    colcxt.update(10, 0, x.blpos);
                    worcxt.reset();
                    worcxt1.reset();
                    fc = 0;
                    isParagraph = 0;
                    firstWord = 0;
                    nl1 = nl;
                    nl = pos - 2;
                }
            }

            colcxt.update(c1, x.c4 & 0xffffff, x.blpos);
            if (c1 < 'a') {
                brcxt.update(c1);
            }
            if (c1 == ' ' and c2 == '<') {
                brcxt.update('>');
            }
            cmC[4].set((brcxt.context << 8) + c1);
            qocxt.update(c1);
            if (htcxt.context != 0 and c2 == 'L' and (c1 == ' ' or c1 == '!' or c1 < 128)) {
                htcxt.update('&' * 256 + 'N');
            }
            htcxt.update(@intCast(x.c4 & 0xffff));

            if (c1 == '$' or c1 == ']' or c1 == '|' or c1 == ')' or c1 == '[') {
                if (c1 != c2) {
                    var i: usize = 13;
                    while (i > 0) : (i -= 1) {
                        t[i] = t[i - 1] *% primes[i];
                    }
                }
                x4 = (x4 << 8) + c2;
                stream2b = stream2b *% 4 +% n2bState;
                stream2bR = (stream2bR << 2) +% n2bState;
                stream3bR = (stream3bR << 3) +% n3bState;
            }

            x4 = (x4 << 8) + c1;
            var i: usize = 13;
            while (i > 0) : (i -= 1) {
                t[i] = t[i - 1] *% primes[i] +% c1 +% @as(u32, @intCast(i * 256));
            }

            if (fc == ' ' and c1 == ' ') {
                cmC2[0].sets();
                cmC2[0].sets();
                cmC2[0].sets();
            } else {
                var k: usize = 3;
                while (k < 6) : (k += 1) {
                    cmC2[0].set(t[k]);
                }
            }
            cmC2[1].set(t[6]);
            cmC2[2].set(t[8]);
            cmC2[3].set(t[13]);

            words = words << 1;
            spaces = spaces << 1;
            numbers = numbers << 1;
            const j = c1;
            if (j >= 'a' and j <= 'z' or (c1 > 127 and c2 != 12)) {
                if (word0 == 0) {
                    if (isMath and c2 == '/' and c3 == '<') {
                        isMath = false;
                    }
                    var reChar = c2;
                    if (c2 == 64 or c2 == 7) {
                        if (c3 != '\'') {
                            reChar = c3;
                        } else if (buf(4) != '\'') {
                            reChar = buf(4);
                        } else if (buf(5) != '\'') {
                            reChar = buf(5);
                        } else if (buf(6) != '\'') {
                            reChar = buf(6);
                        } else {
                            reChar = c3;
                        }
                    } else if (c2 == '/' and c3 == '<') {
                        reChar = c3;
                    }
                    worcxt.set_pbyte(reChar, if (c2 == 64) 1 else 0);
                    worcxt1.set_pbyte(reChar, 0);
                }

                words = words | 1;
                word0 = word0 *% 2104 +% j;
                word00 = word0;
                u8w = 0;
                if (brcxt.cxt == '[' and fccxt.cxt != 31 and fc != 30) {
                    linkword = linkword *% 2104 +% j;
                }
                if (isParagraph != 0 and fccxt.cxt != 31 and !colcxt.isTemp) {
                    senword = senword *% 2104 +% j;
                }
                const word3bit = words & 7;
                if ((word3bit == 5 and c2 == '\'') or
                    (word3bit == 1 and c3 == ']' and c2 == '\'') or
                    (word3bit == 1 and (numbers & 4) != 0 and c2 == '\''))
                {
                    qocxt.update(qocxt.cxt);
                }
                if (c1 > 127) {
                    dcw = dcw *% 256 +% c1;
                    dcwl += 1;
                    if (x.blpos > 6) {
                        var dcw2: u32 = 0;
                        if (dcwl == 2) {
                            dcw2 = (dcw >> 8) | ((dcw & 255) << 8);
                        } else if (dcwl == 3) {
                            dcw2 = (dcw >> 16) | (dcw & 0xff00) | ((dcw & 255) << 16);
                        }
                        const dict_idx = decodeCodeWord(dcw2);
                        if (dict_idx > 0) {
                            deccode = @intCast(dict_idx);
                        }
                    }
                } else if (dcw != 0) {
                    procWord();
                    if (x.blpos < 448131719) {
                        deccode = @intCast(lastCW);
                    }
                }
                if (c1 == 10 or c1 == 9 or (c1 > 31 and c1 < 128)) {
                    setbuf(charSwap(c1));
                }
            } else {
                if (word0 != 0) {
                    procWord();
                    if (x.blpos < 448131719) {
                        deccode = @intCast(lastCW);
                    }
                } else {
                    deccode = 0x10000 + (stream2b & 0xffff);
                }
                if (c1 == 10 or c1 == 9 or (c1 > 31 and c1 < 128)) {
                    setbuf(charSwap(c1));
                }
                if (c1 >= '0' and c1 <= '9') {
                    numbers = numbers +% 1;
                    if ((numbers & 4) != 0 and c2 == ',') {
                        number0 = number1;
                        number1 = 0;
                        numlen0 = numlen1;
                        numlen1 = 0;
                    }
                    if (mybenum != 0 and numlen1 <= 2) {
                        number0 = number1;
                        number1 = 0;
                        numlen0 = numlen1;
                        numlen1 = 0;
                    }
                    number0 = number0 *% 10 +% (c1 & 0x0f);
                    numlen0 = @min(19, numlen0 + 1);
                    mybenum = 0;
                } else {
                    if (numlen0 != 0 or (numbers & 0xf) == 0) {
                        number1 = number0;
                        numlen1 = numlen0;
                        number0 = 0;
                        numlen0 = 0;
                    }
                    if (numlen1 <= 2 and numlen1 != 0 and (numbers & 5) == 5 and numlen0 == 0 and c2 == '.') {
                        mybenum = 2;
                    } else if (numlen1 <= 2 and numlen1 != 0 and (numbers & 2) != 0 and numlen0 == 0 and c1 == '.') {
                        mybenum = 1;
                    } else if (mybenum == 1 and c1 != '.') {
                        mybenum = 0;
                    }
                }
                const word3bit = words & 7;
                if ((word3bit == 4 and c1 == ' ' and c2 == '\'') or
                    (c1 == 64 and (numbers & 4) != 0 and c2 == '\'') or
                    (word3bit == 4 and c1 == 64 and c2 == '\'') or
                    (word3bit == 4 and (numbers & 1) != 0 and c2 == '\''))
                {
                    qocxt.update(qocxt.cxt);
                }
                if (word00 != 0 and fccxt.cxt != '[') {
                    word00 = 0;
                }
                if (word0 != 0) {
                    if (x.blpos > 463139793 or (StemWords[pWordIdx].type & (ConjunctiveAdverb | Conjunction)) == 0) {
                        word3 = word2 *% 47;
                        word2 = word1 *% 53;
                        word1 = word0 *% 83;
                    }
                    if (worcxt.type_val(1) == Number) {
                        stream3bR = (stream3bR << 7) +% 1;
                        stream3b = (stream3b << 7) +% 1;
                    }
                    if (firstWord == 0 and fccxt.cxt != '[') {
                        firstWord = word0;
                    }
                    if ((worcxt.type_val(0) & Conjunction) != 0) {
                        stream3bR = stream3bR << 7;
                        stream3b = stream3b << 7;
                        if (isParagraph != 0) senword = 0;
                    }
                    if ((worcxt.type_val(0) & Article) != 0) {
                        stream3bR = (stream3bR << 7) +% 2;
                        stream3b = (stream3b << 7) +% 2;
                    }
                    if ((worcxt.type_val(0) & Adposition) != 0 or (isParagraph != 0 and (worcxt.type_val(0) & PresentParticiple) != 0)) {
                        stream2bR = (stream2bR << 2) +% (stream2bR & 3);
                        stream2b = (stream2b << 2) +% (stream2b & 3);
                    }
                    if ((worcxt.type_val(0) & AdverbOfManner) != 0) {
                        if (isParagraph != 0) {
                            worcxt.remove();
                        }
                    }
                    if ((worcxt.type_val(0) & Noun) != 0 and (worcxt.type_val(2) & Article) != 0) {
                        stream3bR = (stream3bR << 6) +% 1;
                        stream3b = (stream3b << 6) +% 1;
                        const sb = worcxt.sBytes(1);
                        const w = worcxt.word(1);
                        const t_val = worcxt.type_val(1);
                        const ca = worcxt.capital_val(1);
                        worcxt.remove();
                        worcxt.remove();
                        worcxt.set_pbyte(@intCast(sb >> 8), ca);
                        worcxt.update(w, c1, t_val, w);
                    }
                    stream3bRMask2 = stream3bRMask1;
                    stream3bMask1 = stream3bMask;
                    stream3bRMask1 = 0;
                    stream3bMask = 0;
                    stream2bMask = 0;
                } else if (c1 == '|' and colcxt.isTemp) {
                    const sb = worcxt.sBytes(1);
                    const w = worcxt.word(1);
                    const t_val = worcxt.type_val(1);
                    const ca = worcxt.capital_val(1);
                    worcxt.remove();
                    worcxt.set_pbyte(@intCast(sb >> 8), ca);
                    worcxt.update(w, c1, t_val, w);
                }

                if (buffer1(6) == charSwap('<') and buffer1(5) == 't' and !isText and c1 == ' ' and std.mem.eql(u8, so, "text")) {
                    isText = true;
                    so = "";
                }
                if (buffer1(8) == charSwap('<') and !isNowiki and std.mem.eql(u8, so, "nowiki")) {
                    isNowiki = true;
                } else if (buffer1(9) == '/' and c1 == '>' and isNowiki and std.mem.eql(u8, so, "nowiki")) {
                    isNowiki = false;
                    isPre = false;
                    so = "";
                }

                if (isMath and ((c1 == ' ' and colcxt.lastfc(0) != ':') or c1 == ',') and c2 == '>' and std.mem.eql(u8, so, "math")) {
                    isMath = false;
                    so = "";
                }
                if (isMath and c1 == '/' and c2 == '<' and c3 == '>' and buffer1(4) == 'h') {
                    isMath = false;
                    so = "";
                }

                if (!isNowiki and buffer1(6) == charSwap('<') and buffer1(5) == 'm' and !isMath and c1 != '.' and buffer1(7) != '&' and buffer1(8) != '&' and std.mem.eql(u8, so, "math")) {
                    isMath = true;
                } else if (buffer1(6) == '/' and (c1 == '>' or c1 == '&') and isMath and std.mem.eql(u8, so, "math")) {
                    isMath = false;
                    so = "";
                }

                if (buffer1(5) == charSwap('<') and c1 == '>' and buffer1(4) == 'p' and !isPre and std.mem.eql(u8, so, "pre")) {
                    isPre = true;
                    so = "";
                } else if (buffer1(5) == '/' and c1 == '>' and buffer1(4) == 'p' and std.mem.eql(u8, so, "pre")) {
                    isPre = false;
                    so = "";
                }

                if (buffer1(6) == '/' and c1 == '>' and buffer1(5) == 'p' and std.mem.eql(u8, so, "page")) {
                    isPre = false;
                    isMath = false;
                    isNowiki = false;
                }

                wp[word0 & 0xffff] = pos;
                word0 = 0;

                if (linkword != 0 and c1 == ':') linkword = 0;
                if (c1 == '-' and c2 == ' ') {
                    worcxt1.reset();
                    sVerb = 0;
                }

                if (c1 == ' ') {
                    spaces = spaces +% 1;
                } else if (c1 == 10) {
                    fc = 0;
                    isParagraph = 0;
                    firstWord = 0;
                    lastWT = 0;
                    nl1 = nl;
                    nl = pos - 1;
                    stream3bR = stream3bR << 7;
                    stream2b = stream2b | 0x3fc;
                    words = 0xfc;
                    worcxt.reset();
                    worcxt1.reset();
                    stream2bR = stream2bR << 2;
                    stream4b = stream4b | 0xfff0;
                    if (c2 == 10) isNowiki = false;
                } else if (c1 == '.' or c1 == ')' or c1 == '?') {
                    lastWT = lastWT *% 16;
                    stream3bR = stream3bR << 7;
                    stream3b = stream3b << 7;
                    words = words | 0xfe;
                    x5 = (x5 << 8) + @as(u32, @intCast(x.c4 & 0xff));
                    stream2b = stream2b | 204;
                    stream4b = ((stream4b & 0xffff0) << 8) + (stream4b & 0xf);
                    stream2bR = stream2bR & 0xffffffc0;
                    if (c1 == '.') {
                        wshift = 1;
                        if (!(fccxt.cxt == '[' or fccxt.cxt == '(' or colcxt.nlChar == '-' or colcxt.lastfc(0) == '*')) {
                            worcxt.reset();
                        }
                        senword = 0;
                    }
                    if (c1 == ')') senword = 0;
                } else if (c1 == ',') {
                    words = words | 0xfc;
                    senword = 0;
                } else if (c1 == '(') {
                    senword = 0;
                } else if (c1 == ';') {
                    worcxt.reset();
                } else if (c1 == ':') {
                    stream3b = (stream3b & 0xfffffff8) +% 4;
                    stream2b = stream2b | 12;
                    x5 = (x5 << 8) + @as(u32, @intCast(x.c4 & 0xff));
                    senword = 0;
                } else if (c1 == '}' or c1 == '{') {
                    words = words | 0xfc;
                    stream3bR = stream3bR & 0xffffffc0;
                    x5 = (x5 << 8) + @as(u32, @intCast(x.c4 & 0xff));
                    stream3b = (stream3b & 0xfffffff8) +% 3;
                } else if (c1 == ']') {
                    stream3b = (stream3b & 0xfffffff8) +% 3;
                    linkword = 0;
                } else if (c1 == '<' or c2 == '&') {
                    words = words | 0xfc;
                }

                if (c1 == '-' and colcxt.lastfc(0) == '*' and brcxt.cxt != '[' and isParagraph == 0) {
                    isParagraph = 1;
                    fc = 64;
                }
                if (c1 == '=') {
                    stream3b = (stream3b & 0xfffffff8) +% 4;
                    c2 = '.';
                    words = words *% 2;
                }
                if (c1 == '!' and c2 == '&') {
                    c1 = ' ';
                    x.c4 = (x.c4 & 0xffffff00) + ' ';
                    stream2b = (stream2b & 0xfffffffc) + wrt_2b[' '];
                    stream3b = (stream3b & 0xfffffff8) + wrt_3b[' '];
                } else if (colcxt.lastfc(0) == '*' and (c1 == ',' or c1 == ' ') and c2 == ']' and isParagraph == 0) {
                    isParagraph = 1;
                    fc = 64;
                }
            }

            x5 = (x5 << 8) + @as(u32, @intCast(x.c4 & 0xff));

            if (o2bState != n2bState) {
                stream2bR = (stream2bR << 2) +% n2bState;
                o2bState = n2bState;
            }
            stream2bMask = (stream2bMask << 2) +% 3;

            if (o3bState != n3bState) {
                stream3bR = (stream3bR << 3) +% n3bState;
                stream3bRMask1 = (stream3bRMask1 << 3) +% 7;
                stream3bRMask2 = (stream3bRMask2 << 3) +% 7;
                o3bState = n3bState;
            }
            stream3b = (stream3b << 3) +% n3bState;
            stream3bMask = (stream3bMask << 3) +% 7;
            stream3bMask1 = (stream3bMask1 << 3) +% 7;
            const brcontext = brcxt.cxt;

            BrFcIdx = 0;
            if (brcxt.context != 0) BrFcIdx = fcy[brcontext];
            if (brcxt.context == 0 and qocxt.context != 0) BrFcIdx = fcy[qocxt.context >> 8];

            col = colcxt.collen(0, 0);
            var above = buffer[(nl1 +% @as(u32, @intCast(col))) & BMASK];
            var above1 = buffer[(nl1 +% @as(u32, @intCast(col)) -% 1) & BMASK];
            if (colcxt.nlChar == '>') {
                above = colcxt.colb(1, 0);
                above1 = colcxt.colb(1, 1);
            }
            if (colcxt.isNewLine()) {
                if ((colcxt.nlpos(0) + 2 - colcxt.nlpos(1)) < 4) {
                    fccxt.reset();
                    brcxt.reset();
                    qocxt.reset();
                    htcxt.reset();
                }
                fc = colcxt.lastfc(0);
                if (fc == '>') {
                    fccxt.reset();
                }
                if (fc == 64) {
                    isParagraph = 1;
                } else {
                    isParagraph = 0;
                }
                fccxt.update(@intCast(fc));
            }

            if (col > 2 and c1 > 64 and !isMath) {
                if (fccxt.cxt == '|' and (c1 == ']' or c1 == '}')) {
                    while (fccxt.cxt == '|') fccxt.update(10);
                }
                if ((fccxt.cxt == ':' or fccxt.cxt == 31) and c1 == ']') {
                    while (fccxt.cxt == ':' or fccxt.cxt == 31) fccxt.update(10);
                }
                if (c1 < 128) {
                    fccxt.update(c1);
                }
            }

            if (c1 == ':' and (words & 2) == 2) {
                colonstr = so;
            }
            if (c1 == ' ' and fccxt.cxt == ':' and colcxt.lastfc(0) != ':' and colcxt.nlChar != '-') {
                if (!std.mem.eql(u8, colonstr, "image")) {
                    while (fccxt.cxt == ':') fccxt.update(10);
                }
            }
            if (c1 == ':' and (std.mem.eql(u8, colonstr, "category") or std.mem.eql(u8, colonstr, "wikipedia"))) {
                fccxt.update(10);
                worcxt.remove();
            }
            if (c1 == ' ' and c2 == '<') {
                fccxt.update('>');
            }
            if (fccxt.cxt == ':' and c2 == '/' and c1 == '/') {
                fccxt.update(10);
                fccxt.update(31);
            }
            if (colcxt.lastfc(0) == '[' and c1 == ' ' and isParagraph == 0) {
                if (c2 == ']' or c3 == ']') {
                    fc = 64;
                    isParagraph = 1;
                    fccxt.reset();
                    fccxt.update(@intCast(fc));
                }
            }
            if (fc == ' ' and c1 != ' ') {
                fc = @min(c1, 96);
                if (fc == 64) {
                    isParagraph = 1;
                } else {
                    isParagraph = 0;
                }
                fccxt.update(@intCast(fc));
            }
            const fccontext = fccxt.cxt;
            if (BrFcIdx == 0 and fccxt.context != 0) BrFcIdx = fcy[fccontext];
            FcIdx = fcq[fccontext];

            cmC[5].set((@as(u32, fccxt.context) & 0xff00) + c1 + (stream2b & 12) * 256 + ((@as(u32, brcontext) + @as(u32, @intCast(brcxt.last()))) << 24));

            if (fc == '*' and c1 != ' ') {
                fc = @min(c1, 96);
            }
            if (fc == '&' and c1 == '<') {
                fc = 30;
            }
            if (c2 == '>' and fc == '<' and c1 == '\'') {
                fc = '\'';
            }
            if ((colcxt.lastfc(0) == '\'' or (fc == '\'' and colcxt.lastfc(0) != '*')) and (c1 == ' ')) {
                if (c2 == '\'' or c3 == '\'') {
                    fc = 64;
                    isParagraph = 1;
                    fccxt.reset();
                    fccxt.update(@intCast(fc));
                }
            }
            if (fc != 31 and (x.c4 & 0xffffff) == 0x4a2f2f) {
                fc = 31;
            }

            worcxt.removeWordsL(8, '(', ')');
            worcxt1.removeWordsL(8, '(', ')');
            worcxt.removeWordsL(8, '[', '|');
            worcxt1.removeWordsL(8, '[', '|');
            worcxt.removeWordsL(8, '<', ':');
            if (colcxt.isTemp) worcxt.removeWordsR(10, '=', '|');
            worcxt.removeWordsL(8, '<', '>');
            worcxt1.removeWordsL(8, '<', '>');

            indirectWord = (x.c4 >> 8) & 0xffff;
            t2[indirectWord] = (t2[indirectWord] << 8) | c1;
            indirectWord = x.c4 & 0xffff;
            indirectWord = indirectWord | (t2[indirectWord] << 16);
            indirectByte = (x.c4 >> 8) & 0xff;
            t1[indirectByte] = (t1[indirectByte] << 8) | c1;
            indirectByte = c1 | (t1[c1] << 8);

            t1[brcontext] = (t1[brcontext] << 2) | (stream2b & 3);
            indirectBrByte = (stream3b & 7) | (t1[brcontext] << 3);

            indirectWord0Pos = pos -% wp[word0 & 0xffff];
            if (indirectWord0Pos > 255) {
                indirectWord0Pos = 256 +% (@as(u32, c1) << 16);
            } else {
                indirectWord0Pos = indirectWord0Pos +% (@as(u32, buf(indirectWord0Pos)) << 8) +% (@as(u32, c1) << 16);
            }
            ind3[context1_ind3] = @intCast((cxtind3 * 32 + c1) & (0x2000000 - 1));
            context1_ind3 = (context1_ind3 * 32 + c1) & (0x2000000 - 1);
            cxtind3 = ind3[context1_ind3];

            if (c2 == 12) {
                if (utf8left == 0) {
                    if ((c1 >> 5) == 6) {
                        utf8left = 1;
                        u8w = u8w *% 191 +% c1;
                    } else if ((c1 >> 4) == 0xE) {
                        utf8left = 2;
                        u8w = u8w *% 191 +% c1;
                    } else if ((c1 >> 3) == 0x1E) {
                        utf8left = 3;
                        u8w = u8w *% 191 +% c1;
                    } else {
                        utf8left = 0;
                    }
                } else {
                    utf8left -= 1;
                    if ((c1 >> 6) != 2) utf8left = 0;
                }
            }

            h_hash = h_hash +% c1;

            rcmA[0].set_cxt(word3 *% 53 +% c1 +% 193 *% (stream3b & 0x7fff), c1);
            if (col < 2 or fc == ' ') {
                cmC2[4].sets();
                cmC2[4].sets();
                cmC2[17].sets();
            } else {
                cmC2[4].set(word00 +% (number0 *% 191 +% numlen0) +% u8w);
                if (colcxt.lastfc(0) == '&' or utf8left != 0) {
                    cmC2[4].sets();
                } else {
                    cmC2[4].set(h_hash +% word1);
                }

                if (brcxt.cxt == '<') {
                    cmC2[17].sets();
                } else {
                    cmC2[17].set(worcxt1.word(1) *% 53 +% worcxt1.word(2) *% 11 +% h_hash +% (lastWT & 0xf));
                }
            }
            if (c1 == 12 or col < 2 or utf8left != 0 or fc == ' ') {
                cmC2[5].sets();
            } else {
                cmC2[5].set(h_hash +% word2 *% 71);
            }
            if (fc == ' ' or brcontext == '<') {
                cmC2[5].sets();
                cmC2[5].sets();
                cmC2[5].sets();
                cmC2[5].sets();
                cmC2[5].sets();
            } else {
                cmC2[5].set(worcxt.word(4) *% 53 +% worcxt1.word(1) +% h_hash +% (stream3b & 511));
                cmC2[5].set(worcxt.last(4, worcxt.type_val(4) ^ Verb) *% 53 +% sVerb +% h_hash +% (stream3bR & 63));
                cmC2[5].set(worcxt.fword *% 53 +% worcxt1.word(1) +% h_hash +% (stream3b & 63));
                cmC2[5].set(worcxt2.word(1) +% worcxt2.word(2) *% 11 +% word00 +% c1);
                const lastParVerb = worcxt2.lastIf(1, worcxt.type_val(1) & Verb);
                if (lastParVerb != 0) {
                    cmC2[5].set(lastParVerb *% 11 +% word00 +% c1);
                } else {
                    cmC2[5].sets();
                }
            }
            cmC1[6].set(h_hash +% (worcxt.type_val(1) & 0x1ff) +% worcxt1.word(1));
            cmC2[6].set(((stream2b & 15) << 16) + (t[2] & 0xffff));

            if (c1 == 12 or utf8left != 0 or fccontext == '{') {
                cmC2[7].set(0);
            } else {
                cmC2[7].set(indirectBrByte);
            }

            cmC2[8].set(((indirectBrByte >> 0) & 0x7ff) *% 32 +% ((stream4b & 0xfff0) << 16) +% BrFcIdx);
            cmC2[8].set((stream3bR & 0x3fffffff) *% 4 +% (stream2b & 3));
            cmC2[8].set((@as(u32, fccontext) *% 4) +% ((stream3bR & 0x3ffff) << 9) +% BrFcIdx);
            if (fccontext == 31) {
                cmC2[8].sets();
            } else {
                cmC2[8].set((x.c4 & 0xffffff) +% ((stream2b << 18) & 0xff000000));
            }

            cmC1[0].set(colcxt.lastfc(0) | (@as(u32, fccontext) << 15) | ((stream3b & 63) << 7) | (@as(u32, brcontext) << 24));
            cmC1[0].set(colcxt.lastfc(0) | ((x.c4 & 0xffffff) << 8));

            cmC1[1].set((stream2b & 3) +% word00 *% 11);
            cmC1[1].set(x.c4 & 0xffff);
            cmC1[1].set(((fc << 11) | c1) +% ((stream2b & 3) << 18));

            cmC1[2].set((stream2b & 15) +% ((stream3b & 7) << 6));
            cmC1[2].set(c1 | (@as(u32, @intCast(col * @intFromBool(c1 == ' '))) << 8) | ((stream2b & 15) << 16));
            cmC1[2].set(if (isParagraph != 0) firstWord else (fc << 11));
            if (c1 == 12 or fc == ' ' or utf8left != 0) {
                cmC1[2].sets();
            } else {
                cmC1[2].set(91 *% 83 *% worcxt.word(1) +% 89 *% word0);
            }

            if (fc == ' ') {
                cmC1[4].sets();
            } else {
                cmC1[4].set(c1 +% ((stream3b & 0xe38) << 6));
            }
            cmC1[4].set(worcxt.fword *% 11 +% BrFcIdx);
            cmC1[4].set(c1 +% word0 +% number0 *% 191);
            cmC1[4].set(((x.c4 & 0xffff) << 16) | (@as(u32, fccontext) << 8) | fc);
            cmC1[4].set(((stream3bR & 0xfff) << 8) +% (stream2b & 0xfc));

            if (c1 == 12) {
                cmC[0].sets();
                cmC[0].sets();
                cmC[0].sets();
                cmC[0].sets();
                cmC[0].sets();
                cmC[0].sets();
            } else {
                if (isParagraph == 1) {
                    cmC[0].set(worcxt.fword *% 3191 +% (stream2b & 3));
                    cmC[0].set(h_hash +% firstWord *% 89);
                    cmC[0].set(word0 *% 53 +% c1 +% BrFcIdx);
                } else {
                    cmC[0].set(above | ((stream3b & 0x3f) << 9) | (@as(u32, @intCast(colcxt.collen(0, 0))) << 19) | ((stream2b & 3) << 16));
                    cmC[0].set(h_hash +% firstWord *% 89);
                    cmC[0].set(above | (@as(u32, c1) << 16) | (@as(u32, @intCast(col + @as(i32, @intCast(numlen0)) + @as(i32, @intCast(BrFcIdx)))) << 8) | (@as(u32, above1) << 24));
                }
                if (colcxt.lastfc(0) == '*') {
                    cmC[0].set(word0 +% (@as(u32, fccontext) << 8) | (BrFcIdx << 16));
                    cmC[0].set(c1);
                    cmC[0].set(word0);
                } else {
                    cmC[0].set(wrt_2b[bufr(colcxt.abovecellpos)] | (@as(u32, fccontext) << 8) | (BrFcIdx << 16));
                    cmC[0].set(bufr(colcxt.abovecellpos) | (@as(u32, c1) << 8));
                    cmC[0].set(word0 +% wrt_2b[bufr(colcxt.abovecellpos)]);
                }
            }

            cmC[1].set((stream3b & 0x7fff) *% word0 +% BrFcIdx);
            cmC[1].set((x4 & 0xff0000ff) | ((stream3b & 0xe07) << 8));
            cmC[1].set((indirectBrByte & 0xffff) | ((stream3b & 0x38) << 16));

            if (isMath) {
                cmC[0].sets();
            } else {
                cmC[0].set((indirectByte & 0xff00) +% 257 *% worcxt.word(1) *% 53 +% c1);
            }

            cmC[2].set((@as(u32, c1) << 8) | (indirectByte >> 2) | (fc << 16));
            cmC[2].set((x.c4 & 0xffff) +% @as(u32, @intFromBool(c2 == c3)));

            cmC1[3].set((stream3b & stream3bMask) *% 256 | (stream2b & stream2bMask & 255));
            cmC1[3].set(x4);

            cmC2[9].set(257 *% StemWords[pWordIdx].hash +% fccontext +% 193 *% (stream3b & stream3bMask));
            cmC2[9].set(fc | ((stream2bR & 0xfff) << 9) | (@as(u32, c1) << 24));
            cmC2[16].set(worcxt.fword *% 83 +% (stream2b & 15) *% 11 +% brcontext);
            cmC2[17].set(worcxt.last(1, Verb) +% worcxt.word(1) *% 83 +% h_hash);

            cmC2[9].set((x4 & 0xffff00) +% brcontext +% (@as(u32, fccontext) << 24));
            if (linkword != 0) {
                cmC2[9].set(linkword);
            } else if (isMath) {
                cmC2[9].sets();
            } else if (senword != 0) {
                cmC2[9].set(senword *% 1471 +% c1);
            } else {
                if (fc == 30 or brcontext == '<') {
                    cmC2[9].sets();
                } else {
                    cmC2[9].set(0);
                }
            }

            cmC2[10].set(indirectByte);
            cmC2[10].set(((indirectByte & 0xffff00) >> 4) | (stream2b & stream2bMask & 0xf) | ((stream3b & 0xfff) << 20));
            cmC2[10].set((x4 >> 16) | ((stream2b & 255) << 24));
            if (c1 > 127) {
                cmC2[10].set((((stream2b & 12) * 256 +% @as(u32, c1)) << 11) | ((indirectWord & 0xffffff) >> 16));
            } else {
                cmC2[10].set((@as(u32, c1) << 11) | (BrFcIdx << 8) | ((indirectWord & 0xffffff) >> 16));
            }
            if (isMath) {
                cmC2[10].sets();
            } else {
                cmC2[10].set((fccontext *% 4 +% BrFcIdx) | ((x.c4 & 0xffff) << 9) | ((stream2b & 0xff) << 24));
            }
            cmC2[10].set((indirectWord >> 16) | ((stream2b & 0x3c) << 25) | ((stream3b & 0x1ff) << 16));

            cmC2[11].set(words +% (@as(u32, spaces) << 8) +% ((stream2b & 15) << 16) +% (((stream3bR >> 3) & 511) << 21) +% (@as(u32, @intCast(isParagraph)) << 30));
            cmC2[11].set(c1 +% ((stream3b << 5) & 0x1fffff00));
            cmC2[11].set(stream2bR *% 16 +% BrFcIdx);
            cmC2[11].set(((indirectByte & 0xffff) >> 8) +% ((64 *% stream2bR) & 0x3ffff00) +% (@as(u32, brcontext) << 25));
            if (fccontext == 64 and brcontext == '[') {
                cmC2[11].sets();
            } else {
                cmC2[11].set(indirectWord0Pos | ((indirectByte & 0xff00) << 16));
            }

            cmC2[12].set((x4 & 0x80f00000) +% ((x4 & 0x0000f0ff) << 12));
            if (isParagraph == 1) {
                if (c1 == 12 or fccontext == 31 or fccontext == '{' or isMath or isPre) {
                    cmC2[12].sets();
                } else {
                    cmC2[12].set(h_hash +% worcxt.word(1) *% 53 *% 79 +% worcxt.word(3) *% 53 *% 47 *% 71);
                }
            } else {
                if (fccontext == 31 or brcontext == '<' or htcxt.context != 0) {
                    cmC2[12].sets();
                } else if (col == 31) {
                    cmC2[12].set(x.c4 << 16);
                } else {
                    cmC2[12].set(above | ((x.c4 & 0xffff) << 16) | (@as(u32, above1) << 8));
                }
            }

            if (c1 == 12 or utf8left != 0 or fccontext == '{' or fccontext == 31 or fc == 30 or htcxt.context != 0 or fc == ' ' or isPre or c1 == '&' or brcontext == '<' or isMath or col < 2 or (worcxt.sBytes(0) >> 8) == '\\') {
                cmC2[13].sets();
                cmC2[13].sets();
            } else {
                cmC2[13].set(worcxt.word(1) *% 83 *% 1471 -% word0 *% 53 +% worcxt.word(2));
                cmC2[13].set(h_hash +% worcxt.word(2) *% 53 *% 79 +% worcxt.word(3) *% 53 *% 47 *% 71);
            }

            cmC[3].set(((stream3bR & 7) << 10) + (stream2b & 3) + fc * 4 + (BrFcIdx << 24));
            cmC[3].set((if (linkword != 0) linkword else word0) *% 3301 +% number0 *% 3191);

            if (c1 == 12 or utf8left != 0 or fccontext == '{' or fccontext == 31 or fc == ' ' or fc == 30 or brcontext == '<' or col < 2 or isMath or (worcxt.sBytes(0) >> 8) == '\\') {
                cmC2[14].sets();
            } else {
                cmC2[14].set(BrFcIdx +% worcxt.word(2) *% (stream3bR & stream3bRMask2) +% (worcxt.type_val(1) & 0x1ff));
            }

            if (c1 == 12 or utf8left != 0 or fc == ' ') {
                cmC1[7].sets();
                cmC1[7].sets();
                cmC1[7].sets();
                cmC1[7].sets();
            } else {
                cmC1[7].set(worcxt1.word(1) +% word00);
                cmC1[7].set(worcxt.word(2) +% word0 *% 191 +% (stream3bR & 63));
                cmC1[7].set(word0 *% 191 +% (stream3bR & 63));
                cmC1[7].set((indirectWord0Pos & 0xffff) *% 191 +% word0 +% (stream3bR & 63));
            }

            scmA[0].set_cxt(c1);
            scmA[1].set_cxt(@intCast(c2 * @as(u32, @intCast(isParagraph))));
            scmA[2].set_cxt((indirectWord & 0xffffff) >> 16);
            scmA[3].set_cxt(stream3b & 0x1ff);
            scmA[4].set_cxt(stream2b & 0xff);
            scmA[5].set_cxt(brcontext);
            scmA[6].set_cxt(@intCast(isParagraph + 2 * @as(i32, @intCast(stream3bR & 0x3f))));

            if (wshift != 0 or c1 == 10) {
                word3 = word3 *% 47;
                word2 = word2 *% 53;
                word1 = word1 *% 83;
                wshift = 0;
                if (c1 == 10) sVerb = 0;
            }

            cmC2[15].set((BrFcIdx * 256) + fc + ((stream3bR & 0xFFF) << 16));
            AH1 = hash((x5 >> 0) & 255, (x5 >> 8) & 255, (x5 >> 16) & 0x80ff);
            AH2 = hash(19, x5 & 0x80ffff, 0);
        }

        resetPredictions();
    }
};

pub var h_hash: u32 = 0;
pub var rate: u5 = 6;
pub var pre1: [256]i16 = [_]i16{0} ** 256;
