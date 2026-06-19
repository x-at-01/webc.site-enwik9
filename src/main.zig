const std = @import("std");
const rangecoder = @import("rangecoder.zig");
const predictor = @import("predictor.zig");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const allocator = std.heap.smp_allocator;

    if (args.len < 4) {
        std.debug.print("Usage: {s} [c|d] [input] [output]\n", .{args[0]});
        std.process.exit(1);
    }

    const mode = args[1][0];
    const input_path = args[2];
    const output_path = args[3];

    const start_time = std.Io.Clock.now(.awake, init.io).toMilliseconds();

    if (mode == 'c') {
        try compressFile(init.io, allocator, input_path, output_path);
    } else if (mode == 'd') {
        try decompressFile(init.io, allocator, input_path, output_path);
    } else {
        std.debug.print("Unknown mode: {c}\n", .{mode});
        std.process.exit(1);
    }

    const elapsed = @as(f64, @floatFromInt(std.Io.Clock.now(.awake, init.io).toMilliseconds() - start_time)) / 1000.0;
    std.debug.print("Time: {d:.2} s\n", .{elapsed});
}

fn compressFile(io: std.Io, allocator: std.mem.Allocator, input_path: []const u8, output_path: []const u8) !void {
    const input_file = try std.Io.Dir.openFileAbsolute(io, input_path, .{});
    defer input_file.close(io);
    const original_file_len = try input_file.length(io);
    const raw_data = try allocator.alloc(u8, original_file_len);
    defer allocator.free(raw_data);
    _ = try input_file.readPositionalAll(io, raw_data, 0);

    // Prepend 5-byte segment header matching C++ NoPreprocess
    const file_len = original_file_len + 5;
    const input_data = try allocator.alloc(u8, file_len);
    defer allocator.free(input_data);
    input_data[0] = 0; // DEFAULT
    input_data[1] = @intCast((original_file_len >> 24) & 0xff);
    input_data[2] = @intCast((original_file_len >> 16) & 0xff);
    input_data[3] = @intCast((original_file_len >> 8) & 0xff);
    input_data[4] = @intCast(original_file_len & 0xff);
    @memcpy(input_data[5..], raw_data);

    var vocab = [_]bool{true} ** 256;
    if (file_len >= 10000) {
        @memset(&vocab, false);
        for (input_data) |b| {
            vocab[b] = true;
        }
    }

    var encoder = rangecoder.Encoder.init();
    defer encoder.deinit(allocator);

    var pred = try predictor.Predictor.init(allocator, vocab);
    defer pred.deinit(allocator);

    // Write file size first (8 bytes) as header
    var header_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &header_buf, input_data.len, .little);
    try encoder.out.appendSlice(allocator, &header_buf);

    if (file_len >= 10000) {
        var vocab_bytes = [_]u8{0} ** 32;
        for (0..32) |i| {
            var c: u8 = 0;
            for (0..8) |j| {
                if (vocab[i * 8 + j]) {
                    c |= (@as(u8, 1) << @as(u3, @intCast(j)));
                }
            }
            vocab_bytes[i] = c;
        }
        try encoder.out.appendSlice(allocator, &vocab_bytes);
    }

    const total_bytes = input_data.len;
    const progress_interval = @max(1, total_bytes / 100);
    for (input_data, 0..) |byte, pos| {
        if (pos % progress_interval == 0) {
            std.debug.print("compress progress: {d:.2}%\n", .{@as(f64, @floatFromInt(pos)) * 100.0 / @as(f64, @floatFromInt(total_bytes))});
        }
        var shift: u3 = 7;
        while (true) {
            const bit = @as(u1, @intCast((byte >> shift) & 1));
            const p = pred.predict();
            if (std.math.isNan(p)) {
                std.debug.print("PANIC: pred.predict() returned NaN! bc={d}, tree1=0x{x}, tree3=0x{x}\n", .{
                    pred.bit_context,
                    @as(u32, @bitCast(pred.byte_mixer_tree[1])),
                    @as(u32, @bitCast(pred.byte_mixer_tree[3])),
                });
                std.process.exit(1);
            }

            try encoder.encode(allocator, bit, p);

            pred.perceive(bit);
            if (shift == 0) break;
            shift -= 1;
        }
    }
    try encoder.flush(allocator);

    const output_file = try std.Io.Dir.createFileAbsolute(io, output_path, .{});
    defer output_file.close(io);
    try output_file.writePositionalAll(io, encoder.out.items, 0);

    std.debug.print("Compressed size: {d} bytes\n", .{encoder.out.items.len});
}

fn decompressFile(io: std.Io, allocator: std.mem.Allocator, input_path: []const u8, output_path: []const u8) !void {
    const input_file = try std.Io.Dir.openFileAbsolute(io, input_path, .{});
    defer input_file.close(io);
    const file_len = try input_file.length(io);
    const input_data = try allocator.alloc(u8, file_len);
    defer allocator.free(input_data);
    _ = try input_file.readPositionalAll(io, input_data, 0);

    if (input_data.len < 8) {
        return error.InvalidFile;
    }

    const total_bytes = std.mem.readInt(u64, input_data[0..8], .little);

    var vocab = [_]bool{true} ** 256;
    var payload_offset: usize = 8;
    if (total_bytes >= 10000) {
        if (input_data.len < 8 + 32) {
            return error.InvalidFile;
        }
        @memset(&vocab, false);
        const vocab_bytes = input_data[8 .. 8 + 32];
        for (0..32) |i| {
            const c = vocab_bytes[i];
            for (0..8) |j| {
                if ((c & (@as(u8, 1) << @as(u3, @intCast(j)))) != 0) {
                    vocab[i * 8 + j] = true;
                }
            }
        }
        payload_offset = 8 + 32;
    }
    const compressed_payload = input_data[payload_offset..];

    var decoder = rangecoder.Decoder.init(compressed_payload);

    var pred = try predictor.Predictor.init(allocator, vocab);
    defer pred.deinit(allocator);

    const out_buf = try allocator.alloc(u8, total_bytes);
    defer allocator.free(out_buf);

    const progress_interval = @max(1, total_bytes / 100);
    for (0..total_bytes) |pos| {
        if (pos % progress_interval == 0) {
            std.debug.print("decompress progress: {d:.2}%\n", .{@as(f64, @floatFromInt(pos)) * 100.0 / @as(f64, @floatFromInt(total_bytes))});
        }
        var byte: u8 = 0;
        var bit_idx: u3 = 7;
        while (true) {
            const p = pred.predict();

            const bit = decoder.decode(p);

            pred.perceive(bit);
            byte |= (@as(u8, bit) << bit_idx);
            if (bit_idx == 0) break;
            bit_idx -= 1;
        }
        out_buf[pos] = byte;
    }

    // out_buf contains the preprocessed stream of size `total_bytes`.
    // Parse segments to extract original data.
    var original_data: std.ArrayList(u8) = .empty;
    defer original_data.deinit(allocator);

    var offset: usize = 0;
    while (offset < total_bytes) {
        if (offset + 5 > total_bytes) {
            return error.InvalidFile;
        }
        const segment_type = out_buf[offset];
        if (segment_type != 0) {
            return error.InvalidFile; // Only DEFAULT segments supported
        }
        const segment_len = (@as(usize, out_buf[offset + 1]) << 24) |
                            (@as(usize, out_buf[offset + 2]) << 16) |
                            (@as(usize, out_buf[offset + 3]) << 8) |
                            @as(usize, out_buf[offset + 4]);
        if (offset + 5 + segment_len > total_bytes) {
            return error.InvalidFile;
        }
        try original_data.appendSlice(allocator, out_buf[offset + 5 .. offset + 5 + segment_len]);
        offset += 5 + segment_len;
    }

    const output_file = try std.Io.Dir.createFileAbsolute(io, output_path, .{});
    defer output_file.close(io);
    try output_file.writePositionalAll(io, original_data.items, 0);
}
