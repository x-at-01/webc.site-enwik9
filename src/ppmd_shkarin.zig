const std = @import("std");

const N1 = 4;
const N2 = 4;
const N3 = 4;
const N4 = 26;
const N_INDEXES = N1 + N2 + N3 + N4; // 38
const UNIT_SIZE = 12;
const SCALE = 1 << 15;
const PERIOD_BITS = 7;
const BIN_SCALE = 1 << 14;
const INTERVAL = 1 << 7;
const MAX_FREQ = 124;
const O_BOUND = 9;
const MAX_O = 256;

const EscCoef = [_]i8{ 16, -10, 1, 51, 14, 89, 23, 35, 64, 26, -42, 43 };
const ExpEscape = [_]u8{ 51, 43, 18, 12, 11, 9, 8, 7, 6, 5, 4, 3, 3, 2, 2, 2 };

pub const State = extern struct {
    symbol: u8,
    freq: u8,
    successor_bytes: [4]u8,

    pub inline fn getSuccessor(self: @This()) u32 {
        return std.mem.readInt(u32, &self.successor_bytes, .little);
    }
    pub inline fn setSuccessor(self: *@This(), val: u32) void {
        std.mem.writeInt(u32, &self.successor_bytes, val, .little);
    }
};

pub const PpmContext = extern struct {
    num_stats: u8,
    flags: u8,
    summ_freq_bytes: [2]u8,
    i_stats_bytes: [4]u8,
    i_suffix_bytes: [4]u8,

    pub inline fn getSummFreq(self: @This()) u16 {
        return std.mem.readInt(u16, &self.summ_freq_bytes, .little);
    }
    pub inline fn setSummFreq(self: *@This(), val: u16) void {
        std.mem.writeInt(u16, &self.summ_freq_bytes, val, .little);
    }

    pub inline fn getIStats(self: @This()) u32 {
        return std.mem.readInt(u32, &self.i_stats_bytes, .little);
    }
    pub inline fn setIStats(self: *@This(), val: u32) void {
        std.mem.writeInt(u32, &self.i_stats_bytes, val, .little);
    }

    pub inline fn getISuffix(self: @This()) u32 {
        return std.mem.readInt(u32, &self.i_suffix_bytes, .little);
    }
    pub inline fn setISuffix(self: *@This(), val: u32) void {
        std.mem.writeInt(u32, &self.i_suffix_bytes, val, .little);
    }

    pub inline fn oneState(self: *PpmContext) *align(1) State {
        const ptr: [*]align(1) u8 = @ptrCast(self);
        return @ptrCast(ptr + 2);
    }
};

pub const BlkNode = extern struct {
    stamp: u32,
    next_indx: u32,

    pub inline fn avail(self: @This()) bool {
        return self.next_indx != 0;
    }

    pub inline fn node(self: *align(1) BlkNode) *align(1) BlkNode {
        return @ptrCast(self);
    }
};

pub const MemBlk = extern struct {
    stamp: u32,
    next_indx: u32,
    nu: u32,

    pub inline fn avail(self: @This()) bool {
        return self.next_indx != 0;
    }

    pub inline fn node(self: *align(1) MemBlk) *align(1) BlkNode {
        return @ptrCast(self);
    }
};

pub const See2Context = extern struct {
    summ: u16,
    shift: u8,
    count: u8,

    pub fn init(self: *See2Context, init_val: u32) void {
        self.shift = PERIOD_BITS - 4;
        self.summ = @intCast(init_val << @intCast(self.shift));
        self.count = 7;
    }

    pub fn getMean(self: *const See2Context) u32 {
        return @as(u32, self.summ) >> @intCast(self.shift);
    }

    pub fn update(self: *See2Context) void {
        self.count -%= 1;
        if (self.count == 0) {
            self.setShiftRare();
        }
    }

    pub fn setShiftRare(self: *See2Context) void {
        var i = @as(u32, self.summ) >> @intCast(self.shift);
        const diff1: u8 = if (i > 40) 1 else 0;
        const diff2: u8 = if (i > 280) 1 else 0;
        const diff3: u8 = if (i > 1020) 1 else 0;
        i = PERIOD_BITS - diff1 - diff2 - diff3;
        if (i < self.shift) {
            self.summ >>= 1;
            self.shift -= 1;
        } else if (i > self.shift) {
            self.summ <<= 1;
            self.shift += 1;
        }
        self.count = @intCast(@as(u32, 5) << @intCast(self.shift));
    }
};

pub const Qsym = extern struct {
    sym: u16,
    freq: u16,
    total: u16,

    pub fn store(self: *Qsym, _sym: u32, _freq: u32, _total: u32) void {
        self.sym = @intCast(_sym);
        self.freq = @intCast(_freq);
        self.total = @intCast(_total);
    }
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    heap_slice: []u8,
    heap_start: [*]align(1) u8,

    blist: [N_INDEXES + 1]BlkNode,
    glue_count: u32,
    glue_count_1: u32,
    sub_allocator_size: u64,
    p_text: [*]align(1) u8,
    units_start: [*]align(1) u8,
    lo_unit: [*]align(1) u8,
    hi_unit: [*]align(1) u8,
    aux_unit: [*]align(1) u8,

    indx_2_units: [N_INDEXES]u8,
    units_2_indx: [128]u8,
    ns_2_bs_indx: [256]u8,
    q_table: [260]u8,

    max_order: i32,
    cutoff: i32,
    mmax: i32,
    filesize: u32,
    order_fall: i32,

    found_state: ?*align(1) State,
    max_context: *align(1) PpmContext,
    saved_pc: *align(1) PpmContext,

    esc_count: u32,
    char_mask: [256]u32,

    b_summ: i32,
    run_length: i32,
    init_rl: i32,
    num_masked: i32,

    prev_success: i32,
    bin_summ: [25][64]u16,
    see2_cont: [23][32]See2Context,
    dummy_see2_cont: See2Context,

    sq: [1024]Qsym,
    sq_ptr: u32,
    sqp: [256]u32,
    tr_f: [256]u32,
    tr_t: [256]u32,

    cxt: u32,
    y: u32,

    pub fn init(allocator: std.mem.Allocator, order: u32, memory_mb: u32, cutoff: u32, filesize: u32) !Model {
        var self = Model{
            .allocator = allocator,
            .heap_slice = &.{},
            .heap_start = undefined,
            .blist = undefined,
            .glue_count = 0,
            .glue_count_1 = 0,
            .sub_allocator_size = 0,
            .p_text = undefined,
            .units_start = undefined,
            .lo_unit = undefined,
            .hi_unit = undefined,
            .aux_unit = undefined,
            .indx_2_units = undefined,
            .units_2_indx = undefined,
            .ns_2_bs_indx = undefined,
            .q_table = undefined,
            .max_order = @intCast(order),
            .cutoff = @intCast(cutoff),
            .mmax = @intCast(memory_mb),
            .filesize = filesize,
            .order_fall = 0,
            .found_state = null,
            .max_context = undefined,
            .saved_pc = undefined,
            .esc_count = 0,
            .char_mask = undefined,
            .b_summ = 0,
            .run_length = 0,
            .init_rl = 0,
            .num_masked = 0,
            .prev_success = 0,
            .bin_summ = undefined,
            .see2_cont = undefined,
            .dummy_see2_cont = undefined,
            .sq = undefined,
            .sq_ptr = 0,
            .sqp = [_]u32{0} ** 256,
            .tr_f = [_]u32{0} ** 256,
            .tr_t = [_]u32{0} ** 256,
            .cxt = 0,
            .y = 1,
        };

        self.startup();
        if (!try self.startSubAllocator(@intCast(memory_mb))) {
            return error.SubAllocatorInitFailed;
        }
        self.startModelRare();
        return self;
    }

    pub fn deinit(self: *Model) void {
        self.stopSubAllocator();
    }

    inline fn u2b(self: *Model, nu: u32) u32 {
        _ = self;
        return 12 * nu;
    }

    fn ptr2Indx(self: *Model, p: *align(1) const anyopaque) u32 {
        const p_bytes: [*]align(1) const u8 = @ptrCast(p);
        const addr = @intFromPtr(p_bytes) - @intFromPtr(self.heap_start);
        const lim = @intFromPtr(self.units_start) - @intFromPtr(self.heap_start);
        const indx = if (addr >= lim)
            @as(u32, @intCast((addr - lim) / UNIT_SIZE + lim))
        else
            @as(u32, @intCast(addr));
        return indx;
    }

    fn indx2Ptr(self: *Model, indx: u32) *align(1) anyopaque {
        const lim = @intFromPtr(self.units_start) - @intFromPtr(self.heap_start);
        const addr: u64 = if (indx >= lim)
            @as(u64, indx - lim) * UNIT_SIZE + lim
        else
            @as(u64, indx);
        return @ptrCast(self.heap_start + addr);
    }

    fn setNext(self: *Model, this: *align(1) BlkNode, p: *align(1) BlkNode) void {
        this.next_indx = self.ptr2Indx(p);
    }

    fn getNext(self: *Model, this: *align(1) BlkNode) *align(1) BlkNode {
        return @ptrCast(@alignCast(self.indx2Ptr(this.next_indx)));
    }

    fn link(self: *Model, this: *align(1) BlkNode, p: *align(1) BlkNode) void {
        p.next_indx = this.next_indx;
        self.setNext(this, p);
    }

    fn unlink(self: *Model, this: *align(1) BlkNode) void {
        this.next_indx = self.getNext(this).next_indx;
    }

    fn remove(self: *Model, this: *align(1) BlkNode) *align(1) anyopaque {
        const p = self.getNext(this);
        self.unlink(this);
        this.stamp -%= 1;
        return p;
    }

    fn insert(self: *Model, this: *align(1) BlkNode, pv: *align(1) anyopaque, nu: u32) void {
        const p: *align(1) BlkNode = @ptrCast(pv);
        self.link(this, p);
        p.stamp = ~@as(u32, 0);
        const pm: *align(1) MemBlk = @ptrCast(p);
        pm.nu = nu;
        this.stamp +%= 1;
    }

    fn startSubAllocator(self: *Model, sa_size: u64) !bool {
        const t = sa_size << 20;
        self.heap_slice = try self.allocator.alloc(u8, t);
        self.heap_start = @ptrCast(self.heap_slice.ptr);
        self.sub_allocator_size = t;
        return true;
    }

    fn initSubAllocator(self: *Model) void {
        @memset(std.mem.asBytes(&self.blist), 0);
        self.p_text = self.heap_start;
        self.hi_unit = self.heap_start + self.sub_allocator_size;
        const diff = (self.sub_allocator_size / 8 / UNIT_SIZE) * 7 * UNIT_SIZE;
        self.units_start = self.hi_unit - diff;
        self.lo_unit = self.units_start;
        self.glue_count = 0;
        self.glue_count_1 = 0;
    }

    fn getUsedMemory(self: *Model) u64 {
        const diff1 = @intFromPtr(self.hi_unit) - @intFromPtr(self.lo_unit);
        const diff2 = @intFromPtr(self.units_start) - @intFromPtr(self.p_text);
        var ret_val = self.sub_allocator_size - diff1 - diff2;
        var i: usize = 0;
        while (i < N_INDEXES) : (i += 1) {
            const stamp = self.blist[i].stamp;
            const units = self.indx_2_units[i];
            ret_val -%= @as(u64, units) * stamp * 12;
        }
        return ret_val;
    }

    fn stopSubAllocator(self: *Model) void {
        if (self.sub_allocator_size > 0) {
            self.allocator.free(self.heap_slice);
            self.sub_allocator_size = 0;
        }
    }

    fn glueFreeBlocks(self: *Model) void {
        var i: usize = 0;
        var k: u32 = 0;
        var sz: u32 = 0;
        var s0: MemBlk = undefined;
        s0.stamp = 0;
        s0.next_indx = 0;
        s0.nu = 0;
        var p0: *align(1) MemBlk = &s0;

        if (@intFromPtr(self.lo_unit) != @intFromPtr(self.hi_unit)) {
            self.lo_unit[0] = 0;
        }

        s0.next_indx = 0;
        i = 0;
        while (i <= N_INDEXES) : (i += 1) {
            while (self.blist[i].avail()) {
                const p: *align(1) MemBlk = @ptrCast(@alignCast(self.remove(self.blist[i].node())));
                if (p.nu != 0) {
                    var p_many: [*]align(1) MemBlk = @ptrCast(p);
                    while (true) {
                        const p1 = &p_many[p.nu];
                        if (p1.stamp == ~@as(u32, 0)) {
                            p.nu += p1.nu;
                            p1.nu = 0;
                        } else {
                            break;
                        }
                    }
                    self.link(p0.node(), p.node());
                    p0 = p;
                }
            }
        }

        while (s0.avail()) {
            var p: *align(1) MemBlk = @ptrCast(@alignCast(self.remove(s0.node())));
            sz = p.nu;
            if (sz != 0) {
                var p_many: [*]align(1) MemBlk = @ptrCast(p);
                while (sz > 128) {
                    self.insert(self.blist[N_INDEXES - 1].node(), p_many, 128);
                    sz -= 128;
                    p_many += 128;
                }
                p = @ptrCast(p_many);
                i = self.units_2_indx[sz - 1];
                if (self.indx_2_units[i] != sz) {
                    i -= 1;
                    k = sz - self.indx_2_units[i];
                    const p_k = p_many + (sz - k);
                    self.insert(self.blist[k - 1].node(), p_k, k);
                }
                self.insert(self.blist[i].node(), p, self.indx_2_units[i]);
            }
        }

        self.glue_count = @as(u32, 1) << @intCast(13 + self.glue_count_1);
        self.glue_count_1 +%= 1;
    }

    fn splitBlock(self: *Model, pv: *align(1) anyopaque, old_indx: u32, new_indx: u32) void {
        const old_units = self.indx_2_units[old_indx];
        const new_units = self.indx_2_units[new_indx];
        var u_diff = old_units - new_units;
        var p: [*]align(1) u8 = @ptrCast(pv);
        p += self.u2b(new_units);
        var i = self.units_2_indx[u_diff - 1];
        if (self.indx_2_units[i] != u_diff) {
            i -= 1;
            const k = self.indx_2_units[i];
            self.insert(self.blist[i].node(), p, k);
            p += self.u2b(k);
            u_diff -= k;
        }
        self.insert(self.blist[self.units_2_indx[u_diff - 1]].node(), p, u_diff);
    }

    fn allocUnitsRare(self: *Model, indx: u32) ?*align(1) anyopaque {
        var i = indx;
        while (true) {
            i += 1;
            if (i == N_INDEXES) {
                if (self.glue_count == 0) {
                    self.glue_count = ~@as(u32, 0);
                    self.glueFreeBlocks();
                    i = indx;
                    if (self.blist[i].avail()) {
                        return self.remove(self.blist[i].node());
                    }
                } else {
                    self.glue_count -= 1;
                    const size = self.u2b(self.indx_2_units[indx]);
                    const diff = @intFromPtr(self.units_start) - @intFromPtr(self.p_text);
                    if (diff > size) {
                        self.units_start -= size;
                        return self.units_start;
                    } else {
                        return null;
                    }
                }
            }
            if (self.blist[i].avail()) {
                break;
            }
        }

        const ret_val = self.remove(self.blist[i].node());
        self.splitBlock(ret_val, i, indx);
        return ret_val;
    }

    fn allocUnits(self: *Model, nu: u32) ?*align(1) anyopaque {
        const indx = self.units_2_indx[nu - 1];
        if (self.blist[indx].avail()) {
            return self.remove(self.blist[indx].node());
        }
        const ret_val = self.lo_unit;
        const size = self.u2b(self.indx_2_units[indx]);
        self.lo_unit += size;
        if (@intFromPtr(self.lo_unit) <= @intFromPtr(self.hi_unit)) {
            return ret_val;
        }
        self.lo_unit -= size;
        return self.allocUnitsRare(indx);
    }

    fn allocContext(self: *Model) ?*align(1) anyopaque {
        if (@intFromPtr(self.hi_unit) != @intFromPtr(self.lo_unit)) {
            self.hi_unit -= UNIT_SIZE;
            return self.hi_unit;
        }
        if (self.blist[0].avail()) {
            return self.remove(self.blist[0].node());
        } else {
            return self.allocUnitsRare(0);
        }
    }

    fn freeUnits(self: *Model, ptr: *align(1) anyopaque, nu: u32) void {
        const indx = self.units_2_indx[nu - 1];
        self.insert(self.blist[indx].node(), ptr, self.indx_2_units[indx]);
    }

    fn freeUnit(self: *Model, ptr: *align(1) anyopaque) void {
        const limit = self.units_start + 128 * 1024;
        const i: usize = if (@intFromPtr(ptr) > @intFromPtr(limit)) 0 else N_INDEXES;
        self.insert(self.blist[i].node(), ptr, 1);
    }

    fn unitsCpy(self: *Model, dest: *align(1) anyopaque, src: *align(1) const anyopaque, nu: u32) void {
        _ = self;
        const dest_bytes: [*]align(1) u8 = @ptrCast(dest);
        const src_bytes: [*]align(1) const u8 = @ptrCast(src);
        @memcpy(dest_bytes[0 .. 12 * nu], src_bytes[0 .. 12 * nu]);
    }

    fn expandUnits(self: *Model, old_ptr: *align(1) anyopaque, old_nu: u32) ?*align(1) anyopaque {
        const idx0 = self.units_2_indx[old_nu - 1];
        const idx1 = self.units_2_indx[old_nu - 1 + 1];
        if (idx0 == idx1) {
            return old_ptr;
        }
        const ptr = self.allocUnits(old_nu + 1);
        if (ptr) |p| {
            self.unitsCpy(p, old_ptr, old_nu);
            self.insert(self.blist[idx0].node(), old_ptr, old_nu);
        }
        return ptr;
    }

    fn shrinkUnits(self: *Model, old_ptr: *align(1) anyopaque, old_nu: u32, new_nu: u32) *align(1) anyopaque {
        const idx0 = self.units_2_indx[old_nu - 1];
        const idx1 = self.units_2_indx[new_nu - 1];
        if (idx0 == idx1) {
            return old_ptr;
        }
        if (self.blist[idx1].avail()) {
            const ptr = self.remove(self.blist[idx1].node());
            self.unitsCpy(ptr, old_ptr, new_nu);
            self.insert(self.blist[idx0].node(), old_ptr, self.indx_2_units[idx0]);
            return ptr;
        } else {
            self.splitBlock(old_ptr, idx0, idx1);
            return old_ptr;
        }
    }

    fn moveUnitsUp(self: *Model, old_ptr: *align(1) anyopaque, nu: u32) *align(1) anyopaque {
        const indx = self.units_2_indx[nu - 1];
        prefetchData(old_ptr);
        const limit = self.units_start + 128 * 1024;
        const next_node = self.getNext(self.blist[indx].node());
        if (@intFromPtr(old_ptr) > @intFromPtr(limit) or
            @intFromPtr(old_ptr) > @intFromPtr(next_node))
        {
            return old_ptr;
        }

        const ptr = self.remove(self.blist[indx].node());
        self.unitsCpy(ptr, old_ptr, nu);
        self.insert(self.blist[N_INDEXES].node(), old_ptr, self.indx_2_units[indx]);
        return ptr;
    }

    fn prepareTextArea(self: *Model) void {
        const allocated = self.allocContext();
        if (allocated) |aux| {
            self.aux_unit = @ptrCast(aux);
            if (@intFromPtr(self.aux_unit) == @intFromPtr(self.units_start)) {
                self.units_start += UNIT_SIZE;
                self.aux_unit = self.units_start;
            }
        } else {
            self.aux_unit = self.units_start;
        }
    }

    fn expandTextArea(self: *Model) void {
        var count = [_]u32{0} ** N_INDEXES;
        var i: usize = 0;

        if (@intFromPtr(self.aux_unit) != @intFromPtr(self.units_start)) {
            const val = std.mem.readInt(u32, self.aux_unit[0..4], .little);
            if (val != ~@as(u32, 0)) {
                self.units_start += UNIT_SIZE;
            } else {
                self.insert(self.blist[0].node(), self.aux_unit, 1);
            }
        }

        while (true) {
            const p: *align(1) BlkNode = @ptrCast(@alignCast(self.units_start));
            if (p.stamp == ~@as(u32, 0)) {
                const pm: *align(1) MemBlk = @ptrCast(p);
                const pm_many: [*]align(1) MemBlk = @ptrCast(pm);
                self.units_start = @ptrCast(&pm_many[pm.nu]);
                count[self.units_2_indx[pm.nu - 1]] += 1;
                i += 1;
                pm.stamp = 0;
            } else {
                break;
            }
        }

        if (i != 0) {
            var p = &self.blist[N_INDEXES];
            while (p.next_indx != 0) {
                while (p.next_indx != 0) {
                    const next_node = self.getNext(p.node());
                    if (next_node.stamp == 0) {
                        const next_mem: *align(1) MemBlk = @ptrCast(next_node);
                        count[self.units_2_indx[next_mem.nu - 1]] -= 1;
                        self.unlink(p.node());
                        self.blist[N_INDEXES].stamp -%= 1;
                    } else {
                        break;
                    }
                }
                if (p.next_indx == 0) break;
                p = @ptrCast(@alignCast(self.getNext(p.node())));
            }

            i = 0;
            while (i < N_INDEXES) : (i += 1) {
                p = &self.blist[i];
                while (count[i] != 0) {
                    while (true) {
                        const next_node = self.getNext(p.node());
                        if (next_node.stamp == 0) {
                            self.unlink(p.node());
                            self.blist[i].stamp -%= 1;
                            count[i] -= 1;
                            if (count[i] == 0) break;
                        } else {
                            break;
                        }
                    }
                    if (count[i] == 0) break;
                    p = @ptrCast(@alignCast(self.getNext(p.node())));
                }
            }
        }
    }

    fn minVal(self: *Model, x: anytype, y: anytype) @TypeOf(x, y) {
        _ = self;
        return if (x < y) x else y;
    }

    fn maxVal(self: *Model, x: anytype, y: anytype) @TypeOf(x, y) {
        _ = self;
        return if (x > y) x else y;
    }

    fn clamp(self: *Model, x: anytype, lox: anytype, hix: anytype) @TypeOf(x, lox, hix) {
        _ = self;
        return if (x >= lox) (if (x <= hix) x else hix) else lox;
    }

    fn prefetchData(addr: *align(1) const anyopaque) void {
        const ptr: *volatile u8 = @constCast(@ptrCast(addr));
        _ = ptr.*;
    }

    fn startup(self: *Model) void {
        var i: usize = 0;
        var k: u8 = 1;
        while (i < N1) : (i += 1) {
            self.indx_2_units[i] = k;
            k += 1;
        }
        k += 1;
        while (i < N1 + N2) : (i += 1) {
            self.indx_2_units[i] = k;
            k += 2;
        }
        k += 1;
        while (i < N1 + N2 + N3) : (i += 1) {
            self.indx_2_units[i] = k;
            k += 3;
        }
        k += 1;
        while (i < N1 + N2 + N3 + N4) : (i += 1) {
            self.indx_2_units[i] = k;
            k += 4;
        }

        k = 0;
        i = 0;
        while (k < 128) : (k += 1) {
            if (self.indx_2_units[i] < k + 1) {
                i += 1;
            }
            self.units_2_indx[k] = @intCast(i);
        }

        self.ns_2_bs_indx[0] = 2 * 0;
        self.ns_2_bs_indx[1] = 2 * 1;
        self.ns_2_bs_indx[2] = 2 * 1;
        @memset(self.ns_2_bs_indx[3..29], 2 * 2);
        @memset(self.ns_2_bs_indx[29..256], 2 * 3);

        i = 0;
        const UP_FREQ = 5;
        while (i < UP_FREQ) : (i += 1) {
            self.q_table[i] = @intCast(i);
        }

        var m: u8 = UP_FREQ;
        i = UP_FREQ;
        var step: u32 = 1;
        var k_tbl: u32 = 1;
        while (i < 260) : (i += 1) {
            self.q_table[i] = m;
            k_tbl -= 1;
            if (k_tbl == 0) {
                step += 1;
                k_tbl = step;
                m += 1;
            }
        }
    }

    fn getStats(self: *Model, ctx: *align(1) const PpmContext) *align(1) State {
        return @ptrCast(@alignCast(self.indx2Ptr(ctx.getIStats())));
    }

    fn suff(self: *Model, ctx: *align(1) const PpmContext) *align(1) PpmContext {
        return @ptrCast(@alignCast(self.indx2Ptr(ctx.getISuffix())));
    }

    fn getSucc(self: *Model, state: *align(1) const State) *align(1) PpmContext {
        return @ptrCast(@alignCast(self.indx2Ptr(state.getSuccessor())));
    }

    fn swapState(self: *Model, s1: *align(1) State, s2: *align(1) State) void {
        _ = self;
        const t1 = s1.*;
        s1.* = s2.*;
        s2.* = t1;
    }

    fn rescale(self: *Model, q: *align(1) PpmContext, order_fall_val: i32, found_state_val: *align(1) State) *align(1) State {
        q.flags &= 0x14;

        const p1 = self.getStats(q);
        const tmp = found_state_val.*;
        var p: [*]align(1) State = @ptrCast(found_state_val);
        const p1_many: [*]align(1) State = @ptrCast(p1);
        while (@intFromPtr(p) != @intFromPtr(p1_many)) {
            p[0] = (p - 1)[0];
            p -= 1;
        }
        p1_many[0] = tmp;

        const of = @as(i32, if (order_fall_val != 0) 1 else 0);
        const f0 = @as(i32, p1.freq);
        var sf = @as(i32, q.getSummFreq());
        var esc_freq = sf - f0;
        const new_freq = (f0 + of) >> 1;
        q.setSummFreq(@intCast(new_freq));
        p1.freq = @intCast(new_freq);

        var p_curr_many: [*]align(1) State = @ptrCast(p1);
        var p_curr = p1;
        var i: u32 = 0;
        while (i < q.num_stats) : (i += 1) {
            p_curr_many += 1;
            p_curr = @ptrCast(@alignCast(p_curr_many));
            var a = @as(i32, p_curr.freq);
            esc_freq -= a;
            a = (a + of) >> 1;
            p_curr.freq = @intCast(a);
            q.setSummFreq(q.getSummFreq() +% @as(u16, @intCast(a)));
            if (a != 0) {
                if (p_curr.symbol >= 0x40) {
                    q.flags |= 0x08;
                }
            }
            if (a > (@as([*]align(1) State, @ptrCast(p_curr)) - 1)[0].freq) {
                const tmp_state = p_curr.*;
                var p1_back = p_curr;
                while (true) {
                    const prev_ptr = @as([*]align(1) State, @ptrCast(p1_back)) - 1;
                    if (tmp_state.freq > prev_ptr[0].freq) {
                        p1_back.* = prev_ptr[0];
                        p1_back = @ptrCast(@alignCast(prev_ptr));
                    } else {
                        break;
                    }
                }
                p1_back.* = tmp_state;
            }
        }

        if (p_curr.freq == 0) {
            var i_zero: i32 = 0;
            var p_zero: [*]align(1) State = @ptrCast(p_curr);
            while (p_zero[0].freq == 0) {
                i_zero += 1;
                p_zero -= 1;
            }
            p_curr = @ptrCast(@alignCast(p_zero));
            esc_freq += i_zero;
            const old_a = (@as(u32, q.num_stats) + 2) >> 1;
            q.num_stats -= @intCast(i_zero);
            if (q.num_stats == 0) {
                var tmp_state = self.getStats(q).*;
                const num = 2 * @as(i32, tmp_state.freq) + esc_freq - 1;
                const new_f = @divTrunc(num, esc_freq);
                tmp_state.freq = @intCast(self.minVal(MAX_FREQ / 3, new_f));
                q.flags &= 0x18;
                self.freeUnits(self.getStats(q), old_a);
                q.oneState().* = tmp_state;
                return q.oneState();
            }
            const new_a = (@as(u32, q.num_stats) + 2) >> 1;
            const shrunk = self.shrinkUnits(self.getStats(q), old_a, new_a);
            q.setIStats(self.ptr2Indx(shrunk));
        }

        q.setSummFreq(q.getSummFreq() +% @as(u16, @intCast((esc_freq + 1) >> 1)));
        var a_val: u32 = 0;
        if (order_fall_val != 0 or (q.flags & 0x04) == 0) {
            sf -%= esc_freq;
            const a_val_signed = sf - f0;
            const first_stat_freq = self.getStats(q).freq;
            const num = f0 * @as(i32, q.getSummFreq()) - sf * @as(i32, first_stat_freq) + a_val_signed - 1;
            const val_to_clamp = @divTrunc(num, a_val_signed);
            a_val = self.clamp(@as(u32, @intCast(val_to_clamp)), 2, MAX_FREQ / 2 - 18);
        } else {
            a_val = 2;
        }

        const first_stat = self.getStats(q);
        first_stat.freq += @intCast(a_val);
        q.setSummFreq(q.getSummFreq() +% @as(u16, @intCast(a_val)));
        q.flags |= 0x04;

        return first_stat;
    }

    fn auxCutOff(self: *Model, p: *align(1) State, order: i32, max_order: i32) void {
        if (order < max_order) {
            prefetchData(self.getSucc(p));
            p.setSuccessor(self.cutOff(self.getSucc(p), order + 1, max_order));
        } else {
            p.setSuccessor(0);
        }
    }

    fn cutOff(self: *Model, q: *align(1) PpmContext, order: i32, max_order: i32) u32 {
        var i: i32 = 0;
        var tmp: i32 = 0;
        var esc_freq: i32 = 0;
        var scale: bool = false;
        var p: *align(1) State = undefined;
        var p0: *align(1) State = undefined;

        if (q.num_stats == 0) {
            var flag: bool = true;
            p = q.oneState();
            if (@intFromPtr(self.getSucc(p)) >= @intFromPtr(self.units_start)) {
                self.auxCutOff(p, order, max_order);
                if (p.getSuccessor() != 0 or order < O_BOUND) {
                    flag = false;
                }
            }
            if (flag) {
                self.freeUnit(q);
                return 0;
            }
        } else {
            tmp = (@as(i32, q.num_stats) + 2) >> 1;
            p0 = @ptrCast(@alignCast(self.moveUnitsUp(self.getStats(q), @intCast(tmp))));
            q.setIStats(self.ptr2Indx(p0));

            var i_idx = q.num_stats;
            var p_idx = q.num_stats;
            var p_many: [*]align(1) State = @ptrCast(p0);
            while (p_idx >= 0) : (p_idx -= 1) {
                const pi = &p_many[@intCast(p_idx)];
                if (@intFromPtr(self.getSucc(pi)) < @intFromPtr(self.units_start)) {
                    pi.setSuccessor(0);
                    self.swapState(pi, &p_many[@intCast(i_idx)]);
                    i_idx -= 1;
                } else {
                    self.auxCutOff(pi, order, max_order);
                }
                if (p_idx == 0) break;
            }
            i = @intCast(i_idx);

            if (i != q.num_stats and order > 0) {
                q.num_stats = @intCast(i);
                p = p0;
                if (i < 0) {
                    self.freeUnits(p, @intCast(tmp));
                    self.freeUnit(q);
                    return 0;
                }
                if (i == 0) {
                    const term = if (p.symbol >= 0x40) @as(u8, 0x08) else 0;
                    q.flags = (q.flags & 0x10) + term;
                    const num = 2 * (@as(i32, p.freq) - 1);
                    const den = @as(i32, q.getSummFreq()) - p.freq;
                    const new_f = 1 + @divTrunc(num, den);
                    p.freq = @intCast(new_f);
                    q.oneState().* = p.*;
                    self.freeUnits(p, @intCast(tmp));
                } else {
                    p = @ptrCast(@alignCast(self.shrinkUnits(p0, @intCast(tmp), @intCast((i + 2) >> 1))));
                    q.setIStats(self.ptr2Indx(p));
                    scale = q.getSummFreq() > 16 * i;
                    const scale_term = if (scale) @as(u8, 0x04) else 0;
                    q.flags = q.flags & (0x10 + scale_term);
                    var p_many_shrunk: [*]align(1) State = @ptrCast(p);
                    if (scale) {
                        esc_freq = q.getSummFreq();
                        q.setSummFreq(0);
                        var idx: i32 = 0;
                        while (idx <= q.num_stats) : (idx += 1) {
                            const pi = &p_many_shrunk[@intCast(idx)];
                            esc_freq -= pi.freq;
                            pi.freq = (pi.freq + 1) >> 1;
                            q.setSummFreq(q.getSummFreq() + pi.freq);
                            if (pi.symbol >= 0x40) q.flags |= 0x08;
                        }
                        esc_freq = (esc_freq + 1) >> 1;
                        q.setSummFreq(q.getSummFreq() + @as(u16, @intCast(esc_freq)));
                    } else {
                        var idx: i32 = 0;
                        while (idx <= q.num_stats) : (idx += 1) {
                            const pi = &p_many_shrunk[@intCast(idx)];
                            if (pi.symbol >= 0x40) q.flags |= 0x08;
                        }
                    }
                }
            }
        }

        if (@intFromPtr(q) == @intFromPtr(self.units_start)) {
            self.unitsCpy(self.aux_unit, q, 1);
            return self.ptr2Indx(self.aux_unit);
        } else {
            if (@intFromPtr(self.suff(q)) == @intFromPtr(self.units_start)) {
                q.setISuffix(self.ptr2Indx(self.aux_unit));
            }
        }

        return self.ptr2Indx(q);
    }

    fn startModelRare(self: *Model) void {
        @memset(&self.char_mask, 0);
        self.esc_count = 1;

        if (self.max_order < 2) {
            self.order_fall = self.max_order;
            var pc = self.max_context;
            while (pc.getISuffix() != 0) : (pc = self.suff(pc)) {
                self.order_fall -= 1;
            }
            return;
        }

        self.order_fall = self.max_order;

        self.initSubAllocator();

        self.init_rl = -if (self.max_order < 13) self.max_order else 13;
        self.run_length = self.init_rl;

        const ctx = self.allocContext() orelse return;
        self.max_context = @ptrCast(@alignCast(ctx));
        self.max_context.num_stats = 255;
        self.max_context.setSummFreq(255 + 2);
        const units = self.allocUnits(256 / 2) orelse return;
        self.max_context.setIStats(self.ptr2Indx(units));
        self.max_context.flags = 0;
        self.max_context.setISuffix(0);
        self.prev_success = 0;

        const stats = self.getStats(self.max_context);
        var stats_many: [*]align(1) State = @ptrCast(stats);
        var i: usize = 0;
        while (i < 256) : (i += 1) {
            stats_many[i].symbol = @intCast(i);
            stats_many[i].freq = 1;
            stats_many[i].setSuccessor(0);
        }

        var i2f = [_]u32{0} ** 25;
        var k: u32 = 0;
        i = 0;
        while (i < 25) {
            while (self.q_table[k] == i) {
                k += 1;
            }
            i2f[i] = k + 1;
            i += 1;
        }

        k = 0;
        while (k < 64) : (k += 1) {
            var s: i32 = 0;
            i = 0;
            while (i < 6) : (i += 1) {
                const index = 2 * i + ((k >> @intCast(i)) & 1);
                s += EscCoef[index];
            }
            s = 128 * self.clamp(s, 32, 256 - 32);
            i = 0;
            while (i < 25) : (i += 1) {
                const div_val = @divTrunc(s, @as(i32, @intCast(i2f[i])));
                self.bin_summ[i][k] = @intCast(BIN_SCALE - div_val);
            }
        }

        i = 0;
        while (i < 23) : (i += 1) {
            var k_idx: u32 = 0;
            while (k_idx < 32) : (k_idx += 1) {
                self.see2_cont[i][k_idx].init(8 * @as(u32, @intCast(i)) + 5);
            }
        }
    }

    fn restoreModelRare(self: *Model) void {
        self.p_text = self.heap_start;
        const pc = self.saved_pc;

        while (true) {
            if (self.max_context.num_stats == 1 and self.max_context != pc) {
                const p = self.getStats(self.max_context);
                const p_next = &(@as([*]align(1) State, @ptrCast(p)) + 1)[0];
                if (@intFromPtr(self.getSucc(p_next)) >= @intFromPtr(self.units_start)) {
                    break;
                }
                const term = if (p.symbol >= 0x40) @as(u8, 0x08) else 0;
                self.max_context.flags = (self.max_context.flags & 0x10) + term;
                p.freq = (p.freq + 1) >> 1;
                self.max_context.oneState().* = p.*;
                self.max_context.num_stats = 0;
                self.freeUnits(p, 1);
            } else {
                break;
            }
            self.max_context = self.suff(self.max_context);
        }

        while (self.max_context.getISuffix() != 0) {
            self.max_context = self.suff(self.max_context);
        }

        self.aux_unit = self.units_start;

        self.expandTextArea();

        while (true) {
            self.prepareTextArea();
            _ = self.cutOff(self.max_context, 0, self.max_order);
            self.expandTextArea();
            if (self.getUsedMemory() <= 3 * (self.sub_allocator_size >> 2)) {
                break;
            }
        }

        self.glue_count = 0;
        self.glue_count_1 = 0;
        self.order_fall = self.max_order;
    }

    fn updateModel(self: *Model, min_context: *align(1) PpmContext) ?*align(1) PpmContext {
        var flag: u8 = 0;
        var fsymbol: u8 = 0;
        var ns1: u32 = 0;
        var ns: u32 = 0;
        var cf: u32 = 0;
        var sf: u32 = 0;
        var s0: u32 = 0;
        var ffreq: u32 = 0;
        var i_successor: u32 = 0;
        var i_f_successor: u32 = 0;
        var pc: *align(1) PpmContext = undefined;
        var p: ?*align(1) State = null;

        const found = self.found_state.?;
        fsymbol = found.symbol;
        ffreq = found.freq;
        i_f_successor = found.getSuccessor();

        if (min_context.getISuffix() != 0) {
            pc = self.suff(min_context);
            if (pc.num_stats != 0) {
                p = self.getStats(pc);
                if (p.?.symbol != fsymbol) {
                    var p_many: [*]align(1) State = @ptrCast(p.?);
                    p_many += 1;
                    while (p_many[0].symbol != fsymbol) {
                        p_many += 1;
                    }
                    p = @ptrCast(@alignCast(p_many));
                    if (p_many[0].freq >= (p_many - 1)[0].freq) {
                        self.swapState(p.?, @ptrCast(@alignCast(p_many - 1)));
                        p = @ptrCast(@alignCast(p_many - 1));
                    }
                }
                if (p.?.freq < MAX_FREQ - 3) {
                    cf = 2 + @as(u32, if (ffreq < 28) 1 else 0);
                    p.?.freq += @intCast(cf);
                    pc.setSummFreq(pc.getSummFreq() +% @as(u16, @intCast(cf)));
                }
            } else {
                p = pc.oneState();
                p.?.freq += @as(u8, if (p.?.freq < 14) 1 else 0);
            }
        }

        if (self.order_fall == 0 and i_f_successor != 0) {
            found.setSuccessor(self.createSuccessors(1, p, min_context));
            if (found.getSuccessor() == 0) {
                self.saved_pc = pc;
                return null;
            }
            self.max_context = self.getSucc(found);
            return self.max_context;
        }

        self.p_text[0] = fsymbol;
        self.p_text += 1;
        i_successor = self.ptr2Indx(self.p_text);
        if (@intFromPtr(self.p_text) >= @intFromPtr(self.units_start)) {
            self.saved_pc = pc;
            return null;
        }

        if (i_f_successor != 0) {
            if (@intFromPtr(self.indx2Ptr(i_f_successor)) < @intFromPtr(self.units_start)) {
                i_f_successor = self.createSuccessors(0, p, min_context);
            } else {
                prefetchData(self.indx2Ptr(i_f_successor));
            }
        } else {
            i_f_successor = self.reduceOrder(p, min_context);
        }

        if (i_f_successor == 0) {
            self.saved_pc = pc;
            return null;
        }

        self.order_fall -= 1;
        if (self.order_fall == 0) {
            i_successor = i_f_successor;
            if (self.max_context != min_context) {
                self.p_text -= 1;
            }
        }

        s0 = min_context.getSummFreq() - ffreq;
        ns = min_context.num_stats;
        flag = if (fsymbol >= 0x40) @as(u8, 0x08) else 0;
        pc = self.max_context;
        while (pc != min_context) {
            ns1 = pc.num_stats;
            if (ns1 != 0) {
                if ((ns1 & 1) != 0) {
                    const expanded = self.expandUnits(self.getStats(pc), (ns1 + 1) >> 1);
                    if (expanded == null) {
                        self.saved_pc = pc;
                        return null;
                    }
                    pc.setIStats(self.ptr2Indx(expanded.?));
                    p = @ptrCast(@alignCast(expanded.?));
                }
                pc.setSummFreq(pc.getSummFreq() +% @as(u16, @intCast(self.q_table[ns + 4] >> 3)));
            } else {
                const allocated = self.allocUnits(1);
                if (allocated == null) {
                    self.saved_pc = pc;
                    return null;
                }
                p = @ptrCast(@alignCast(allocated.?));
                p.?.* = pc.oneState().*;
                pc.setIStats(self.ptr2Indx(p.?));
                p.?.freq = if (p.?.freq <= MAX_FREQ / 3) (2 * p.?.freq - 1) else (MAX_FREQ - 15);
                const exp_esc = ExpEscape[self.q_table[@intCast(self.b_summ >> 8)] & 0x0F];
                pc.setSummFreq(@as(u16, @intCast(p.?.freq)) +% (if (ns > 1) @as(u16, 1) else 0) +% exp_esc);
            }

            cf = (ffreq - 1) * (5 + pc.getSummFreq());
            sf = s0 + pc.getSummFreq();

            if (cf <= 3 * sf) {
                cf = 1 + (if (2 * cf > sf) @as(u32, 1) else 0) + (if (2 * cf > 3 * sf) @as(u32, 1) else 0);
                pc.setSummFreq(pc.getSummFreq() +% 4);
            } else {
                cf = 5 + (if (cf > 5 * sf) @as(u32, 1) else 0) + (if (cf > 6 * sf) @as(u32, 1) else 0) +
                    (if (cf > 8 * sf) @as(u32, 1) else 0) + (if (cf > 10 * sf) @as(u32, 1) else 0) +
                    (if (cf > 12 * sf) @as(u32, 1) else 0);
                pc.setSummFreq(pc.getSummFreq() +% @as(u16, @intCast(cf)));
            }

            pc.num_stats += 1;
            const stats = self.getStats(pc);
            const stats_many: [*]align(1) State = @ptrCast(stats);
            p = &stats_many[pc.num_stats];
            p.?.setSuccessor(i_successor);
            p.?.symbol = fsymbol;
            p.?.freq = @intCast(cf);
            pc.flags |= flag;

            pc = self.suff(pc);
        }

        self.max_context = @ptrCast(@alignCast(self.indx2Ptr(i_f_successor)));
        return self.max_context;
    }

    fn createSuccessors(self: *Model, skip: u32, p_in: ?*align(1) State, pc_in: *align(1) PpmContext) u32 {
        const p = p_in;
        var pc = pc_in;
        var ps: [MAX_O]*align(1) State = undefined;
        var pps_idx: usize = 0;

        const found = self.found_state.?;
        const sym = found.symbol;
        const i_up_branch = found.getSuccessor();

        if (skip == 0) {
            ps[pps_idx] = found;
            pps_idx += 1;
            if (pc.getISuffix() == 0) {
                return self.finishSuccessors(pps_idx, &ps, pc, sym, i_up_branch);
            }
        }

        if (p) |p_val| {
            pc = self.suff(pc);
            if (p_val.getSuccessor() != i_up_branch) {
                pc = self.getSucc(p_val);
            } else {
                ps[pps_idx] = p_val;
                pps_idx += 1;
                if (pc.getISuffix() != 0) {
                    pc = self.loopSuccessors(pps_idx, &ps, pc, sym, i_up_branch, &pps_idx);
                }
            }
        } else {
            pc = self.loopSuccessors(pps_idx, &ps, pc, sym, i_up_branch, &pps_idx);
        }

        return self.finishSuccessors(pps_idx, &ps, pc, sym, i_up_branch);
    }

    fn loopSuccessors(
        self: *Model,
        initial_pps_idx: usize,
        ps: *[MAX_O]*align(1) State,
        pc_in: *align(1) PpmContext,
        sym: u8,
        i_up_branch: u32,
        pps_idx_out: *usize,
    ) *align(1) PpmContext {
        var pc = pc_in;
        var pps_idx = initial_pps_idx;
        var p: *align(1) State = undefined;
        while (true) {
            pc = self.suff(pc);
            if (pc.num_stats != 0) {
                var p_many: [*]align(1) State = @ptrCast(self.getStats(pc));
                while (p_many[0].symbol != sym) {
                    p_many += 1;
                }
                p = @ptrCast(@alignCast(p_many));
                const tmp: u8 = if (p.freq < MAX_FREQ - 1) 2 else 0;
                p.freq += tmp;
                pc.setSummFreq(pc.getSummFreq() +% tmp);
            } else {
                p = pc.oneState();
                const term = if (self.suff(pc).num_stats == 0) @as(u32, 1) else 0;
                const cond = if (p.freq < 16) @as(u32, 1) else 0;
                p.freq += @intCast(term & cond);
            }

            if (p.getSuccessor() != i_up_branch) {
                pc = self.getSucc(p);
                break;
            }
            ps[pps_idx] = p;
            pps_idx += 1;

            if (pc.getISuffix() == 0) break;
        }
        pps_idx_out.* = pps_idx;
        return pc;
    }

    fn finishSuccessors(
        self: *Model,
        pps_idx: usize,
        ps: *[MAX_O]*align(1) State,
        pc_in: *align(1) PpmContext,
        sym_in: u8,
        i_up_branch: u32,
    ) u32 {
        var pc = pc_in;
        var sym = sym_in;
        if (pps_idx == 0) {
            return self.ptr2Indx(pc);
        }

        var ct: PpmContext = undefined;
        ct.num_stats = 0;
        ct.flags = if (sym >= 0x40) @as(u8, 0x10) else 0;
        const up_ptr: [*]align(1) u8 = @ptrCast(self.indx2Ptr(i_up_branch));
        sym = up_ptr[0];
        ct.oneState().setSuccessor(self.ptr2Indx(up_ptr + 1));
        ct.oneState().symbol = sym;
        ct.flags |= if (sym >= 0x40) @as(u8, 0x08) else 0;

        var p: *align(1) State = undefined;
        if (pc.num_stats != 0) {
            var p_many: [*]align(1) State = @ptrCast(self.getStats(pc));
            while (p_many[0].symbol != sym) {
                p_many += 1;
            }
            p = @ptrCast(@alignCast(p_many));
            const cf = @as(u32, p.freq) - 1;
            const s0 = @as(u32, pc.getSummFreq()) - pc.num_stats - cf;
            const term: u32 = if (2 * cf < s0) (if (12 * cf > s0) 1 else 0) else 2 + cf / s0;
            const cf_new = 1 + term;
            ct.oneState().freq = @intCast(self.minVal(@as(u32, 7), cf_new));
        } else {
            ct.oneState().freq = pc.oneState().freq;
        }

        var curr_pps_idx = pps_idx;
        while (curr_pps_idx > 0) {
            const allocated = self.allocContext();
            if (allocated == null) return 0;
            const pc1: *align(1) PpmContext = @ptrCast(@alignCast(allocated.?));
            pc1.num_stats = ct.num_stats;
            pc1.flags = ct.flags;
            pc1.oneState().* = ct.oneState().*;
            pc1.setISuffix(self.ptr2Indx(pc));
            pc = pc1;
            curr_pps_idx -= 1;
            ps[curr_pps_idx].setSuccessor(self.ptr2Indx(pc));
        }

        return self.ptr2Indx(pc);
    }

    fn reduceOrder(self: *Model, p_in: ?*align(1) State, pc_in: *align(1) PpmContext) u32 {
        var p = p_in;
        var pc = pc_in;
        const pc1 = pc;
        const found = self.found_state.?;
        found.setSuccessor(self.ptr2Indx(self.p_text));
        const sym = found.symbol;
        const i_up_branch = found.getSuccessor();
        self.order_fall += 1;

        if (p) |p_val| {
            pc = self.suff(pc);
            if (p_val.getSuccessor() == 0) {
                p_val.setSuccessor(i_up_branch);
                self.order_fall += 1;
                while (true) {
                    if (pc.getISuffix() == 0) return self.ptr2Indx(pc);
                    pc = self.suff(pc);
                    if (pc.num_stats != 0) {
                        var p_many: [*]align(1) State = @ptrCast(self.getStats(pc));
                        while (p_many[0].symbol != sym) {
                            p_many += 1;
                        }
                        p = @ptrCast(@alignCast(p_many));
                        const tmp: u8 = if (p.?.freq < MAX_FREQ - 3) 2 else 0;
                        p.?.freq += tmp;
                        pc.setSummFreq(pc.getSummFreq() +% tmp);
                    } else {
                        p = pc.oneState();
                        p.?.freq += if (p.?.freq < 11) @as(u8, 1) else 0;
                    }
                    if (p.?.getSuccessor() != 0) break;
                    p.?.setSuccessor(i_up_branch);
                    self.order_fall += 1;
                }
            } else {
                p = p_val;
            }
        } else {
            while (true) {
                if (pc.getISuffix() == 0) return self.ptr2Indx(pc);
                pc = self.suff(pc);
                if (pc.num_stats != 0) {
                    var p_many: [*]align(1) State = @ptrCast(self.getStats(pc));
                    while (p_many[0].symbol != sym) {
                        p_many += 1;
                    }
                    p = @ptrCast(@alignCast(p_many));
                    const tmp: u8 = if (p.?.freq < MAX_FREQ - 3) 2 else 0;
                    p.?.freq += tmp;
                    pc.setSummFreq(pc.getSummFreq() +% tmp);
                } else {
                    p = pc.oneState();
                    p.?.freq += if (p.?.freq < 11) @as(u8, 1) else 0;
                }
                if (p.?.getSuccessor() != 0) break;
                p.?.setSuccessor(i_up_branch);
                self.order_fall += 1;
            }
        }

        if (p.?.getSuccessor() <= i_up_branch) {
            const p1 = self.found_state;
            self.found_state = p;
            p.?.setSuccessor(self.createSuccessors(0, null, pc));
            self.found_state = p1;
        }

        if (self.order_fall == 1 and pc1 == self.max_context) {
            found.setSuccessor(p.?.getSuccessor());
            self.p_text -= 1;
        }

        return p.?.getSuccessor();
    }

    fn processBinSymbol(self: *Model, q: *align(1) PpmContext, symbol: i32) void {
        const rs = q.oneState();
        const run_len_term = (@as(u32, @bitCast(self.run_length)) >> 26) & 0x20;
        const i = self.ns_2_bs_indx[self.suff(q).num_stats] + @as(u32, @intCast(self.prev_success)) + q.flags + run_len_term;
        const freq_idx = self.q_table[rs.freq - 1];
        const bs_ref = &self.bin_summ[freq_idx][i];
        self.b_summ = bs_ref.*;
        bs_ref.* -%= @intCast((self.b_summ + 64) >> PERIOD_BITS);

        const flag = rs.symbol != symbol;
        if (flag) {
            self.char_mask[rs.symbol] = self.esc_count;
            self.num_masked = 0;
            self.prev_success = 0;
            self.found_state = null;
        } else {
            bs_ref.* +%= INTERVAL;
            rs.freq += if (rs.freq < 196) @as(u8, 1) else 0;
            self.run_length += 1;
            self.prev_success = 1;
            self.found_state = rs;
        }
    }

    fn processSymbol1(self: *Model, q: *align(1) PpmContext, symbol: i32) void {
        const stats = self.getStats(q);
        var p_many: [*]align(1) State = @ptrCast(stats);

        const cnum = q.num_stats;
        const first_symbol = p_many[0].symbol;
        var low: u32 = 0;
        var freq = p_many[0].freq;
        var flag = first_symbol == symbol;

        var p: ?*align(1) State = null;

        if (flag) {
            self.prev_success = 0;
            p_many[0].freq += 4;
            q.setSummFreq(q.getSummFreq() +% 4);
            p = @ptrCast(@alignCast(&p_many[0]));
        } else {
            self.prev_success = 0;
            low = freq;
            var idx: u32 = 1;
            while (idx <= cnum) : (idx += 1) {
                freq = p_many[idx].freq;
                flag = p_many[idx].symbol == symbol;
                if (flag) break;
                low += freq;
            }

            if (flag) {
                p_many[idx].freq += 4;
                q.setSummFreq(q.getSummFreq() +% 4);
                if (p_many[idx].freq > p_many[idx - 1].freq) {
                    self.swapState(&p_many[idx], &p_many[idx - 1]);
                    idx -= 1;
                }
                p = @ptrCast(@alignCast(&p_many[idx]));
            } else {
                if (q.getISuffix() != 0) {
                    prefetchData(self.suff(q));
                }
                self.num_masked = cnum;
                var j: u32 = 0;
                while (j <= cnum) : (j += 1) {
                    self.char_mask[p_many[j].symbol] = self.esc_count;
                }
                p = null;
            }
        }

        self.found_state = p;
        if (p) |p_val| {
            if (p_val.freq > MAX_FREQ) {
                self.found_state = self.rescale(q, self.order_fall, p_val);
            }
        }
    }

    fn processSymbol2(self: *Model, q: *align(1) PpmContext, symbol: i32) void {
        const stats = self.getStats(q);
        var p_many: [*]align(1) State = @ptrCast(stats);
        const cnum = q.num_stats;

        var psee2c: *See2Context = undefined;
        var see_freq: u32 = 0;

        if (cnum != 0xFF) {
            const q_idx = self.q_table[cnum + 3] - 4;
            var ptr = &self.see2_cont[q_idx];
            const cond1 = if (q.getSummFreq() > 10 * (@as(u32, cnum) + 1)) @as(usize, 1) else 0;
            const cond2 = if (2 * @as(u32, cnum) < @as(u32, self.suff(q).num_stats) + @as(u32, @intCast(self.num_masked))) @as(usize, 1) else 0;
            psee2c = &ptr[cond1 + 2 * cond2 + q.flags];
            see_freq = psee2c.getMean() + 1;
        } else {
            psee2c = &self.dummy_see2_cont;
            see_freq = 1;
        }

        var flag: bool = false;
        var j: usize = 0;
        var low: u32 = 0;

        var i: usize = 0;
        while (i <= cnum) : (i += 1) {
            const c = p_many[i].symbol;
            if (self.char_mask[c] != self.esc_count) {
                self.char_mask[c] = self.esc_count;
                low += p_many[i].freq;
                if (c == symbol) {
                    flag = true;
                    j = i;
                }
            }
        }

        const total = see_freq + low;

        if (flag) {
            const p = &p_many[j];
            if (see_freq > 2) {
                psee2c.summ -%= @intCast(see_freq);
            }
            psee2c.update();

            self.found_state = p;
            p.freq += 4;
            q.setSummFreq(q.getSummFreq() +% 4);
            if (p.freq > MAX_FREQ) {
                self.found_state = self.rescale(q, self.order_fall, p);
            }
            self.run_length = self.init_rl;
            self.esc_count += 1;
        } else {
            self.num_masked = cnum;
            psee2c.summ +%= @intCast(total - see_freq);
        }
    }

    fn processBinSymbolT(self: *Model, q: *align(1) PpmContext) void {
        const rs = q.oneState();
        const run_len_term = (@as(u32, @bitCast(self.run_length)) >> 26) & 0x20;
        const i = self.ns_2_bs_indx[self.suff(q).num_stats] + @as(u32, @intCast(self.prev_success)) + q.flags + run_len_term;
        const freq_idx = self.q_table[rs.freq - 1];
        self.b_summ = self.bin_summ[freq_idx][i];

        self.sq[self.sq_ptr].store(rs.symbol, @intCast(self.b_summ + self.b_summ), SCALE);
        self.sq_ptr += 1;
        self.sq[self.sq_ptr].store(256, SCALE - @as(u32, @intCast(self.b_summ + self.b_summ)), SCALE);
        self.sq_ptr += 1;

        self.char_mask[rs.symbol] = self.esc_count;
        self.num_masked = 0;
    }

    fn processSymbol1T(self: *Model, q: *align(1) PpmContext) void {
        const stats = self.getStats(q);
        const p_many: [*]align(1) State = @ptrCast(stats);
        const cnum = q.num_stats;
        var low: u32 = 0;
        const total = q.getSummFreq();

        var i: usize = 0;
        while (i <= cnum) : (i += 1) {
            const freq = p_many[i].freq;
            self.sq[self.sq_ptr].store(p_many[i].symbol, freq, total);
            self.sq_ptr += 1;
            low += freq;
        }

        if (q.getISuffix() != 0) {
            prefetchData(self.suff(q));
        }

        self.num_masked = cnum;
        i = 0;
        while (i <= cnum) : (i += 1) {
            self.char_mask[p_many[i].symbol] = self.esc_count;
        }

        self.sq[self.sq_ptr].store(256, total - low, total);
        self.sq_ptr += 1;
    }

    fn processSymbol2T(self: *Model, q: *align(1) PpmContext) void {
        const stats = self.getStats(q);
        const p_many: [*]align(1) State = @ptrCast(stats);
        const cnum = q.num_stats;

        var psee2c: *See2Context = undefined;
        var see_freq: u32 = 0;

        if (cnum != 0xFF) {
            const q_idx = self.q_table[cnum + 3] - 4;
            var ptr = &self.see2_cont[q_idx];
            const cond1 = if (q.getSummFreq() > 10 * (@as(u32, cnum) + 1)) @as(usize, 1) else 0;
            const cond2 = if (2 * @as(u32, cnum) < @as(u32, self.suff(q).num_stats) + @as(u32, @intCast(self.num_masked))) @as(usize, 1) else 0;
            psee2c = &ptr[cond1 + 2 * cond2 + q.flags];
            see_freq = psee2c.getMean() + 1;
        } else {
            psee2c = &self.dummy_see2_cont;
            see_freq = 1;
        }

        var low: u32 = 0;
        var i: usize = 0;
        while (i <= cnum) : (i += 1) {
            const c = p_many[i].symbol;
            if (self.char_mask[c] != self.esc_count) {
                low += p_many[i].freq;
            }
        }
        const total = see_freq + low;

        i = 0;
        while (i <= cnum) : (i += 1) {
            const c = p_many[i].symbol;
            if (self.char_mask[c] != self.esc_count) {
                self.sq[self.sq_ptr].store(c, p_many[i].freq, total);
                self.sq_ptr += 1;
                self.char_mask[c] = self.esc_count;
            }
        }

        self.sq[self.sq_ptr].store(256, see_freq, total);
        self.sq_ptr += 1;
        self.num_masked = cnum;
    }

    fn convertSQ(self: *Model) void {
        var cnum: u32 = 256;
        var cum: u32 = 0xFFFFFF00;

        @memset(&self.sqp, 0);
        @memset(&self.tr_f, 0);
        @memset(&self.tr_t, 0);

        var i: usize = 0;
        while (i < self.sq_ptr) : (i += 1) {
            const c = self.sq[i].sym;
            const freq = self.sq[i].freq;
            const total = self.sq[i].total;
            const prob = @as(u32, @intCast((@as(u64, cum) * freq) / total));
            if (c < 256) {
                self.sqp[c] = prob + 1;
                cnum -= 1;
            } else {
                cum = prob;
            }
        }

        var c: usize = 0;
        while (c < 256) : (c += 1) {
            i = 8;
            while (i != 0) {
                const j = (256 + c) >> @intCast(i);
                const b = (c >> @intCast(i - 1)) & 1;
                const term = if (b == 0) self.sqp[c] else 0;
                self.tr_f[j] += term;
                self.tr_t[j] += self.sqp[c];
                i -= 1;
            }
        }
    }

    pub fn ppmdPrepareByte(self: *Model) void {
        self.sq_ptr = 0;
        self.num_masked = 0;
        const old_order_fall = self.order_fall;

        var min_context = self.max_context;
        if (min_context.num_stats != 0) {
            self.processSymbol1T(min_context);
        } else {
            self.processBinSymbolT(min_context);
        }

        while (true) {
            while (true) {
                if (min_context.getISuffix() == 0) {
                    self.esc_count += 1;
                    self.num_masked = 0;
                    self.order_fall = old_order_fall;
                    self.convertSQ();
                    return;
                }
                self.order_fall += 1;
                min_context = self.suff(min_context);
                if (min_context.num_stats != self.num_masked) {
                    break;
                }
            }
            self.processSymbol2T(min_context);
        }
    }

    pub fn ppmdUpdateByte(self: *Model, c: u32) void {
        var min_context = self.max_context;
        if (min_context.num_stats != 0) {
            self.processSymbol1(min_context, @intCast(c));
        } else {
            self.processBinSymbol(min_context, @intCast(c));
        }

        while (self.found_state == null) {
            while (true) {
                self.order_fall += 1;
                min_context = self.suff(min_context);
                if (min_context.num_stats != self.num_masked) {
                    break;
                }
            }
            self.processSymbol2(min_context, @intCast(c));
        }

        var p: ?*align(1) PpmContext = null;
        const found = self.found_state.?;
        if (self.order_fall != 0 or @intFromPtr(self.getSucc(found)) < @intFromPtr(self.units_start)) {
            p = self.updateModel(min_context);
            if (p) |p_val| {
                self.max_context = p_val;
            }
        } else {
            self.max_context = self.getSucc(found);
            p = self.max_context;
        }

        if (p == null) {
            if (self.cutoff != 0) {
                self.restoreModelRare();
            } else {
                self.startModelRare();
            }
        }
    }
};

test "basic ppmd_shkarin initialization and update" {
    const allocator = std.testing.allocator;
    var model = try Model.init(allocator, 8, 8, 1, 0); // order=8, memory_mb=8, cutoff=1, filesize=0
    defer model.deinit();

    // Try a few updates and prepares
    model.ppmdUpdateByte(100);
    model.ppmdPrepareByte();

    // Verify that sq_ptr is set up
    try std.testing.expect(self_sq_ptr: {
        break :self_sq_ptr model.sq_ptr > 0;
    });
}
