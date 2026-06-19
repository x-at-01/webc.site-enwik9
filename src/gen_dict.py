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

    # Now write the Zig code
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("// Generated dictionary file. Do not edit.\n\n")
        f.write("pub const word_count: usize = {};\n\n".format(len(words)))
        
        # Write offsets array
        f.write("pub const word_offsets = [_]u32{\n")
        for i, o in enumerate(offsets):
            f.write("    {},\n".format(o))
        f.write("};\n\n")
        
        # Write packed words data as a single string literal
        f.write("pub const words_data = \n")
        # Split data into readable chunks/lines for Zig compiler safety
        chunk_size = 80
        total_data = b"".join(w + b"\x00" for w in packed_words)
        
        i = 0
        while i < len(total_data):
            chunk = total_data[i:i+chunk_size]
            # Format chunk as escaped string
            escaped = ""
            for b in chunk:
                if b == 0:
                    escaped += "\\x00"
                elif b == ord('\\'):
                    escaped += "\\\\"
                elif b == ord('"'):
                    escaped += "\\\""
                elif 32 <= b <= 126:
                    escaped += chr(b)
                else:
                    escaped += "\\x{:02x}".format(b)
            if i > 0:
                f.write('    ++ "{}"\n'.format(escaped))
            else:
                f.write('    "{}"\n'.format(escaped))
            i += chunk_size
        f.write(";\n\n")
        
        # Write helper function to get word
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
