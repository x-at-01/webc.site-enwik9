const std = @import("std");

inline fn hash6(c1: u8, c2: u8, c3: u8, c4: u8, c5: u8, c6: u8) u64 {
    return @as(u64, c1) *% 160963 +% @as(u64, c2) *% 90523 +% @as(u64, c3) *% 50683 +% @as(u64, c4) *% 28309 +% @as(u64, c5) *% 10607 +% c6;
}

inline fn hash5(c1: u8, c2: u8, c3: u8, c4: u8, c5: u8) u64 {
    return @as(u64, c1) *% 90523 +% @as(u64, c2) *% 50683 +% @as(u64, c3) *% 28309 +% @as(u64, c4) *% 10607 +% c5;
}

inline fn hash4(c1: u8, c2: u8, c3: u8, c4: u8) u64 {
    return @as(u64, c1) *% 50683 +% @as(u64, c2) *% 28309 +% @as(u64, c3) *% 10607 +% c4;
}

inline fn hash3(c1: u8, c2: u8, c3: u8) u64 {
    return @as(u64, c1) *% 28309 +% @as(u64, c2) *% 10607 +% c3;
}

inline fn hash2(c1: u8, c2: u8) u32 {
    return (@as(u32, c1) *% 257 +% c2) & 0xffff;
}

inline fn hashBytes(bytes: []const u8) u64 {
    var h: u64 = 0xcbf29ce484222325;
    for (bytes) |b| {
        h = (h ^ b) *% 0x100000001b3;
    }
    return h;
}

pub const PpmContext = struct {
    key: u64 = 0,
    bytes: [8]u8 = .{0} ** 8,
    counts: [8]u16 = .{0} ** 8,
    total: u32 = 0,
};

pub const PpmTable = struct {
    entries: []PpmContext,

    pub fn init(allocator: std.mem.Allocator, size: usize) !PpmTable {
        const entries = try allocator.alloc(PpmContext, size);
        @memset(entries, PpmContext{});
        return PpmTable{ .entries = entries };
    }

    pub fn deinit(self: *PpmTable, allocator: std.mem.Allocator) void {
        allocator.free(self.entries);
    }

    pub fn lookup(self: *PpmTable, key: u64) ?*PpmContext {
        const mask = self.entries.len - 1;
        var idx = @as(usize, @intCast(key & mask));
        var i: usize = 0;
        while (i < 16) : (i += 1) {
            const entry = &self.entries[idx];
            if (entry.key == key) {
                if (entry.total == 0) return null;
                return entry;
            }
            if (entry.total == 0) {
                return null;
            }
            idx = (idx + 1) & mask;
        }
        return null;
    }

    pub fn lookupForUpdate(self: *PpmTable, key: u64) *PpmContext {
        const mask = self.entries.len - 1;
        var idx = @as(usize, @intCast(key & mask));
        var min_total_idx = idx;
        var min_total = self.entries[idx].total;
        var i: usize = 0;
        while (i < 16) : (i += 1) {
            const entry = &self.entries[idx];
            if (entry.key == key or entry.total == 0) {
                return entry;
            }
            if (entry.total < min_total) {
                min_total = entry.total;
                min_total_idx = idx;
            }
            idx = (idx + 1) & mask;
        }
        // Evict the entry with the smallest total count
        const evict_entry = &self.entries[min_total_idx];
        evict_entry.key = key;
        @memset(&evict_entry.bytes, 0);
        @memset(&evict_entry.counts, 0);
        evict_entry.total = 0;
        return evict_entry;
    }
};

pub const PpmModel = struct {
    vocab: [256]bool,
    order1: []PpmContext,
    order2: []PpmContext,
    orders: [23]PpmTable,
    tree: [512]f32 = undefined,

    pub fn init(allocator: std.mem.Allocator, vocab: [256]bool) !PpmModel {
        const order1 = try allocator.alloc(PpmContext, 256);
        @memset(order1, PpmContext{});
        const order2 = try allocator.alloc(PpmContext, 262144);
        @memset(order2, PpmContext{});

        var orders: [23]PpmTable = undefined;
        for (0..23) |i| {
            orders[i] = try PpmTable.init(allocator, 524288);
        }

        var self = PpmModel{
            .vocab = vocab,
            .order1 = order1,
            .order2 = order2,
            .orders = orders,
        };
        self.tree = .{1.0} ** 512;
        return self;
    }

    pub fn deinit(self: *PpmModel, allocator: std.mem.Allocator) void {
        allocator.free(self.order1);
        allocator.free(self.order2);
        for (&self.orders) |*table| {
            table.deinit(allocator);
        }
    }

    fn countUnique(ctx: *const PpmContext) u32 {
        var u: u32 = 0;
        for (ctx.counts) |c| {
            if (c > 0) u += 1;
        }
        return u;
    }

    fn updateContext(ctx: *PpmContext, key: u64, b: u8) void {
        if (ctx.key != key) {
            ctx.key = key;
            ctx.total = 0;
            @memset(&ctx.bytes, 0);
            @memset(&ctx.counts, 0);
        }
        ctx.total += 1;
        for (0..8) |i| {
            if (ctx.counts[i] == 0) {
                ctx.bytes[i] = b;
                ctx.counts[i] = 1;
                return;
            }
            if (ctx.bytes[i] == b) {
                ctx.counts[i] += 1;
                return;
            }
        }

        var min_idx: usize = 0;
        var min_val: u16 = ctx.counts[0];
        for (1..8) |i| {
            if (ctx.counts[i] < min_val) {
                min_val = ctx.counts[i];
                min_idx = i;
            }
        }

        var found_zero = false;
        for (0..8) |i| {
            if (ctx.counts[i] > 1) {
                ctx.counts[i] -= 1;
            } else {
                ctx.counts[i] = 0;
                if (!found_zero) {
                    ctx.bytes[i] = b;
                    ctx.counts[i] = 1;
                    found_zero = true;
                }
            }
        }
        if (!found_zero) {
            ctx.bytes[min_idx] = b;
            ctx.counts[min_idx] = 1;
        }

        ctx.total = 0;
        for (0..8) |i| {
            ctx.total += ctx.counts[i];
        }

        if (ctx.total >= 4000) {
            for (0..8) |i| {
                ctx.counts[i] = (ctx.counts[i] + 1) >> 1;
            }
            ctx.total = 0;
            for (0..8) |i| {
                ctx.total += ctx.counts[i];
            }
        }
    }

    pub fn update(self: *PpmModel, history: []const u8, pos: usize, b: u8) void {
        var buf: [25]u8 = undefined;
        for (0..25) |i| {
            if (pos < i + 1) break;
            buf[i] = history[pos - 1 - i];

            if (i == 0) {
                updateContext(&self.order1[buf[0]], buf[0], b);
            } else if (i == 1) {
                const h = hashBytes(buf[0..2]);
                updateContext(&self.order2[h & 0x3ffff], h, b);
            } else {
                const h = hashBytes(buf[0 .. i + 1]);
                const table = &self.orders[i - 2];
                const ctx = table.lookupForUpdate(h);
                updateContext(ctx, h, b);
            }
        }
    }

    pub fn predict(self: *PpmModel, history: []const u8, pos: usize) void {
        var out_probs = [_]f32{0.0} ** 256;
        var escape_prob: f32 = 1.0;
        var seen = [_]bool{false} ** 256;

        var buf: [25]u8 = undefined;
        var max_order: usize = 0;
        for (0..25) |i| {
            if (pos < i + 1) break;
            buf[i] = history[pos - 1 - i];
            max_order = i + 1;
        }

        var o = max_order;
        while (o >= 1) : (o -= 1) {
            const ctx_opt: ?*PpmContext = if (o == 1)
                &self.order1[buf[0]]
            else if (o == 2)
                &self.order2[hashBytes(buf[0..2]) & 0x3ffff]
            else b: {
                const h = hashBytes(buf[0..o]);
                break :b self.orders[o - 3].lookup(h);
            };

            if (ctx_opt) |ctx| {
                const h_expected = if (o == 1) @as(u64, buf[0]) else hashBytes(buf[0..o]);
                if (o == 1 or ctx.key == h_expected) {
                    var T: u32 = 0;
                    var u: u32 = 0;
                    for (0..8) |i| {
                        if (ctx.counts[i] > 0 and !seen[ctx.bytes[i]]) {
                            T += ctx.counts[i];
                            u += 1;
                        }
                    }
                    if (T > 0) {
                        const T_f = @as(f32, @floatFromInt(T));
                        const u_f = @as(f32, @floatFromInt(u));
                        const esc = (0.5 * u_f) / T_f;
                        for (0..8) |i| {
                            if (ctx.counts[i] > 0) {
                                const b = ctx.bytes[i];
                                if (!seen[b]) {
                                    out_probs[b] += escape_prob * ((@as(f32, @floatFromInt(ctx.counts[i])) - 0.5) / T_f);
                                    seen[b] = true;
                                }
                            }
                        }
                        escape_prob *= esc;
                    }
                }
            }
        }

        // Order 0 fallback
        var unseen_count: u32 = 0;
        for (0..256) |b| {
            if (self.vocab[b] and !seen[b]) unseen_count += 1;
        }
        if (unseen_count > 0) {
            const p_unseen = escape_prob / @as(f32, @floatFromInt(unseen_count));
            for (0..256) |b| {
                if (self.vocab[b] and !seen[b]) out_probs[b] += p_unseen;
            }
        }

        for (0..256) |b| {
            if (!self.vocab[b]) {
                out_probs[b] = 0.0;
            }
        }

        // Populate tree
        for (0..256) |b| {
            self.tree[256 + b] = out_probs[b];
        }
        var idx: usize = 255;
        while (idx >= 1) : (idx -= 1) {
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1];
        }
    }

    pub fn predict_bit(self: *PpmModel, bc: u32) f32 {
        const sum = self.tree[bc];
        if (sum <= 0.000001) return 0.5;
        const p1 = self.tree[bc * 2 + 1] / sum;
        if (p1 < 0.0001) return 0.0001;
        if (p1 > 0.9999) return 0.9999;
        return p1;
    }
};
