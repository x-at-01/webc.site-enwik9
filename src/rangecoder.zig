const std = @import("std");

pub const Encoder = struct {
    x1: u32 = 0,
    x2: u32 = 0xffffffff,
    out: std.ArrayList(u8) = .empty,

    pub fn init() Encoder {
        return .{};
    }

    pub fn deinit(self: *Encoder, allocator: std.mem.Allocator) void {
        self.out.deinit(allocator);
    }

    pub inline fn discretize(p: f32) u32 {
        const cp = if (p < 0.0) 0.0 else if (p > 1.0) 1.0 else p;
        return 1 + @as(u32, @intFromFloat(65534.0 * cp));
    }

    pub fn encode(self: *Encoder, allocator: std.mem.Allocator, bit: u1, p_val: f32) !void {
        const p = discretize(p_val);
        const diff = self.x2 - self.x1;
        const xmid = self.x1 + (diff >> 16) * p + ((diff & 0xffff) * p >> 16);
        if (bit == 1) {
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }

        while (((self.x1 ^ self.x2) & 0xff000000) == 0) {
            try self.out.append(allocator, @as(u8, @intCast(self.x2 >> 24)));
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) + 255;
        }
    }

    pub fn flush(self: *Encoder, allocator: std.mem.Allocator) !void {
        while (((self.x1 ^ self.x2) & 0xff000000) == 0) {
            try self.out.append(allocator, @as(u8, @intCast(self.x2 >> 24)));
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) + 255;
        }
        try self.out.append(allocator, @as(u8, @intCast(self.x2 >> 24)));
    }
};

pub const Decoder = struct {
    x1: u32 = 0,
    x2: u32 = 0xffffffff,
    x: u32 = 0,
    input: []const u8,
    pos: usize = 0,

    pub fn init(input: []const u8) Decoder {
        var self = Decoder{
            .input = input,
        };
        for (0..4) |_| {
            self.x = (self.x << 8) + @as(u32, self.readByte());
        }
        return self;
    }

    inline fn readByte(self: *Decoder) u8 {
        if (self.pos >= self.input.len) return 0;
        const b = self.input[self.pos];
        self.pos += 1;
        return b;
    }

    pub inline fn discretize(p: f32) u32 {
        const cp = if (p < 0.0) 0.0 else if (p > 1.0) 1.0 else p;
        return 1 + @as(u32, @intFromFloat(65534.0 * cp));
    }

    pub fn decode(self: *Decoder, p_val: f32) u1 {
        const p = discretize(p_val);
        const diff = self.x2 - self.x1;
        const xmid = self.x1 + (diff >> 16) * p + ((diff & 0xffff) * p >> 16);
        var bit: u1 = 0;
        if (self.x <= xmid) {
            bit = 1;
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }

        while (((self.x1 ^ self.x2) & 0xff000000) == 0) {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) + 255;
            self.x = (self.x << 8) + @as(u32, self.readByte());
        }
        return bit;
    }
};
