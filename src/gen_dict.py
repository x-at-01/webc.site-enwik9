import os

def main():
    dict_path = "/Users/z/git/test/compress/fx3-cmix/dictionary/english.dic"
    out_path = "/Users/z/git/test/compress/zig/src/dictionary.zig"
    
    if not os.path.exists(dict_path):
        print(f"Error: {dict_path} does not exist!")
        return

    with open(dict_path, "r", encoding="utf-8", errors="ignore") as f:
        words = [line.strip() for line in f]

    print(f"Loaded {len(words)} words.")

    # We will pack all words with null separators
    # and create an array of offsets.
    packed_words = []
    offsets = []
    current_offset = 0

    for w in words:
        offsets.append(current_offset)
        # Convert word to ascii/utf-8 bytes
        word_bytes = w.encode("utf-8")
        packed_words.append(word_bytes)
        current_offset += len(word_bytes) + 1  # +1 for null byte/separator

    # Append final offset for end boundary of last word
    offsets.append(current_offset)

    import struct

    # Write offsets as binary file
    offsets_bin_path = "/Users/z/git/test/compress/zig/src/word_offsets.bin"
    with open(offsets_bin_path, "wb") as f_offsets:
        for o in offsets:
            f_offsets.write(struct.pack("<I", o))
    print(f"Wrote {len(offsets)*4} bytes to {offsets_bin_path}")

    # Write words data as binary file
    total_data = b"".join(w + b"\x00" for w in packed_words)
    words_data_bin_path = "/Users/z/git/test/compress/zig/src/words_data.bin"
    with open(words_data_bin_path, "wb") as f_data:
        f_data.write(total_data)
    print(f"Wrote {len(total_data)} bytes to {words_data_bin_path}")

    # Now write the Zig code
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("// Generated dictionary file. Do not edit.\n")
        f.write("const std = @import(\"std\");\n\n")
        f.write("pub const word_count: usize = {};\n\n".format(len(words)))
        f.write("const offsets_raw align(@alignOf(u32)) = @embedFile(\"word_offsets.bin\").*;\n")
        f.write("pub const word_offsets = std.mem.bytesAsSlice(u32, &offsets_raw);\n\n")
        f.write("pub const words_data = @embedFile(\"words_data.bin\");\n\n")
        f.write(
            """pub fn getWord(index: usize) []const u8 {
    if (index >= word_count) return "";
    const start = word_offsets[index];
    const end = word_offsets[index + 1] - 1;
    return words_data[start..end];
}
"""
        )

    print(f"Successfully generated {out_path}.")

if __name__ == "__main__":
    main()
