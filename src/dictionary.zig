// Generated dictionary file. Do not edit.
const std = @import("std");

pub const word_count: usize = 44515;

const offsets_raw align(@alignOf(u32)) = @embedFile("word_offsets.bin").*;
pub const word_offsets = std.mem.bytesAsSlice(u32, &offsets_raw);

pub const words_data = @embedFile("words_data.bin");

pub fn getWord(index: usize) []const u8 {
    if (index >= word_count) return "";
    const start = word_offsets[index];
    const end = word_offsets[index + 1] - 1;
    return words_data[start..end];
}
