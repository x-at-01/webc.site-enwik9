const std = @import("std");

pub const BoundedArrayU8 = struct {
    buffer: [10]u8 = undefined,
    len: usize = 0,

    pub fn init() BoundedArrayU8 {
        return .{};
    }
    pub fn append(self: *BoundedArrayU8, val: u8) void {
        if (self.len < 10) {
            self.buffer[self.len] = val;
            self.len += 1;
        }
    }
    pub fn pop(self: *BoundedArrayU8) ?u8 {
        if (self.len == 0) return null;
        self.len -= 1;
        return self.buffer[self.len];
    }
    pub fn orderedRemove(self: *BoundedArrayU8, index: usize) u8 {
        const val = self.buffer[index];
        var i = index;
        while (i < self.len - 1) : (i += 1) {
            self.buffer[i] = self.buffer[i + 1];
        }
        self.len -= 1;
        return val;
    }
};

pub const BoundedArrayU32 = struct {
    buffer: [10]u32 = undefined,
    len: usize = 0,

    pub fn init() BoundedArrayU32 {
        return .{};
    }
    pub fn append(self: *BoundedArrayU32, val: u32) void {
        if (self.len < 10) {
            self.buffer[self.len] = val;
            self.len += 1;
        }
    }
    pub fn pop(self: *BoundedArrayU32) ?u32 {
        if (self.len == 0) return null;
        self.len -= 1;
        return self.buffer[self.len];
    }
    pub fn orderedRemove(self: *BoundedArrayU32, index: usize) u32 {
        const val = self.buffer[index];
        var i = index;
        while (i < self.len - 1) : (i += 1) {
            self.buffer[i] = self.buffer[i + 1];
        }
        self.len -= 1;
        return val;
    }
};

pub const BracketStatsPair = struct {
    first: u32,
    second: u32,
};

pub const BracketModel = struct {
    vocab: [256]bool,
    probs: [256]f32 = undefined,
    top: usize = 255,
    bot: usize = 0,
    active: BoundedArrayU8,
    distance: BoundedArrayU32,
    stats: *[256][200]BracketStatsPair,
    allocator: std.mem.Allocator,

    const brackets_map = init: {
        var map = [_]u8{0} ** 256;
        map['('] = ')';
        map['P'] = 'R';
        map['['] = ']';
        map['L'] = 'N';
        map['\''] = '\'';
        map['"'] = '"';
        break :init map;
    };

    pub fn init(allocator: std.mem.Allocator, vocab: [256]bool) !BracketModel {
        const stats = try allocator.create([256][200]BracketStatsPair);
        for (stats) |*row| {
            for (row) |*pair| {
                pair.* = .{ .first = 1, .second = 256 };
            }
        }
        var self = BracketModel{
            .vocab = vocab,
            .probs = undefined,
            .active = BoundedArrayU8.init(),
            .distance = BoundedArrayU32.init(),
            .stats = stats,
            .allocator = allocator,
        };
        @memset(&self.probs, 1.0 / 256.0);
        return self;
    }

    pub fn deinit(self: *BracketModel) void {
        self.allocator.destroy(self.stats);
    }

    pub inline fn getBracketContext(self: *const BracketModel) u64 {
        if (self.active.len > 0) {
            const last_active = self.active.buffer[self.active.len - 1];
            const last_dist = self.distance.buffer[self.distance.len - 1];
            return 200 *% (@as(u64, last_active) +% 1) +% last_dist;
        } else {
            return 0;
        }
    }

    pub inline fn predict(self: *const BracketModel) f32 {
        const mid = self.bot + ((self.top - self.bot) / 2);
        var num: f32 = 0.0;
        var denom: f32 = 0.0;
        for (self.probs[mid + 1 .. self.top + 1]) |p| {
            num += p;
        }
        for (self.probs[self.bot .. mid + 1]) |p| {
            denom += p;
        }
        denom += num;
        if (denom == 0) return 0.5;
        return num / denom;
    }

    pub inline fn perceive(self: *BracketModel, bit: u1) void {
        const mid = self.bot + ((self.top - self.bot) / 2);
        if (bit == 1) {
            self.bot = mid + 1;
        } else {
            self.top = mid;
        }
    }

    pub fn byteUpdate(self: *BracketModel, byte_val: u8) void {
        self.top = 255;
        self.bot = 0;
        @memset(&self.probs, 1.0 / 256.0);

        const is_bracket = brackets_map[byte_val] != 0;
        const last_active_is_byte = if (self.active.len > 0) self.active.buffer[self.active.len - 1] == byte_val else false;
        const bracket_of_byte_is_byte = brackets_map[byte_val] == byte_val;

        if (self.active.len == 0 or (is_bracket and !(last_active_is_byte and bracket_of_byte_is_byte))) {
            if (is_bracket) {
                self.active.append(byte_val);
                self.distance.append(0);
                if (self.active.len > 10) {
                    _ = self.active.orderedRemove(0);
                    _ = self.distance.orderedRemove(0);
                }
                const p = @as(f32, @floatFromInt(self.stats[byte_val][0].first)) / @as(f32, @floatFromInt(self.stats[byte_val][0].second));
                @memset(&self.probs, (1.0 - p) / 255.0);
                self.probs[brackets_map[byte_val]] = p;
            }
        } else {
            const active = self.active.buffer[self.active.len - 1];
            var distance = self.distance.buffer[self.distance.len - 1];
            self.stats[active][distance].second += 1;
            if (brackets_map[active] == byte_val) {
                self.stats[active][distance].first += 1;
            }
            if (self.stats[active][distance].second > 100000) {
                self.stats[active][distance].first /= 2;
                self.stats[active][distance].second /= 2;
            }
            if (brackets_map[active] == byte_val or distance >= 199) {
                _ = self.active.pop();
                _ = self.distance.pop();
                if (self.active.len > 0) {
                    const act = self.active.buffer[self.active.len - 1];
                    const dist = self.distance.buffer[self.distance.len - 1];
                    const p = @as(f32, @floatFromInt(self.stats[act][dist].first)) / @as(f32, @floatFromInt(self.stats[act][dist].second));
                    @memset(&self.probs, (1.0 - p) / 255.0);
                    self.probs[brackets_map[act]] = p;
                }
            } else {
                self.distance.buffer[self.distance.len - 1] += 1;
                distance += 1;
                const p = @as(f32, @floatFromInt(self.stats[active][distance].first)) / @as(f32, @floatFromInt(self.stats[active][distance].second));
                @memset(&self.probs, (1.0 - p) / 255.0);
                self.probs[brackets_map[active]] = p;
            }
        }

        for (0..256) |i| {
            if (!self.vocab[i]) {
                self.probs[i] = 0.0;
            }
        }
    }
};
