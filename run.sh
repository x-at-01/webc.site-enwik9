#!/usr/bin/env bash
set -e
DIR=$(realpath $0) && DIR=${DIR%/*}
cd $DIR
# set -x

ENWIK9_1M=/tmp/enwik9_1m


zig build -Doptimize=ReleaseFast -Dcpu=native

COMP=${ENWIK9_1M}.zig.comp
DECOMP=${ENWIK9_1M}.zig.decomp
time ./zig-out/bin/compressor c $ENWIK9_1M $COMP

ZIG_SIZE=$(wc -c <$COMP | tr -d ' ')

echo "Original Size: 1,000,000 bytes"
echo "Zig Compressed:      $ZIG_SIZE bytes (Ratio: $(echo "scale=4; $ZIG_SIZE / 10000" | bc)%)"

# Verify correctness by decompressing and comparing
./zig-out/bin/compressor d $COMP $DECOMP
if cmp -s $ENWIK9_1M $DECOMP; then
    echo "Verification: SUCCESS (Decompressed matches original)"
else
    echo "Verification: FAILED!"
    exit 1
fi

rm -f $COMP $DECOMP

