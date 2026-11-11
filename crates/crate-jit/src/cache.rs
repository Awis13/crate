// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Translated code: one region of executable memory per CPU, the shared stubs
//! at its start, and the blocks after them. Each block's cold code (its slow
//! paths and the exits it takes until a chain is patched) goes at the far end
//! of the region instead, which fills downwards, so the code that runs is
//! packed together for the CPU's instruction cache.
//!
//! A block is found by the physical address its guest PC translated to and
//! the mode it was translated for (its exception level, and whether debug
//! made it a single instruction). Translated code computes every guest
//! virtual address it needs (PC-relative values, link values, branch
//! targets) from the run-time page of the block, so one translation serves
//! every virtual address its physical page is mapped at: the same library
//! code in every process, at whatever address each process's layout puts it.
//!
//! The cache also keeps, per 4 KiB physical page, the blocks translated from
//! it, so an instruction-cache invalidation of that page drops exactly those.
//! Blocks never cross a page, so a chain (a patched branch from one block to
//! another) never leaves its page, and dropping a page's blocks drops every
//! chain into them.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use crate_host::jit::JitRegion;

use crate::asm::*;
use crate::emit::{self, Stubs};
use crate::translate::{self, Ctx};

/// Identifies a translation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockKey {
    pub pa: u64,
    /// The exception level in bits 1:0, and bit 2 for single-instruction
    /// (debug) blocks.
    pub mode: u32,
}

/// A small, fast hasher for the cache's integer keys.
#[derive(Default)]
pub struct FastHasher(u64);

impl Hasher for FastHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    fn write_u32(&mut self, v: u32) {
        self.write_u64(v as u64);
    }
}

type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FastHasher>>;

/// How many guest instructions a block may hold before it is cut off.
const MAX_BLOCK: usize = 512;

/// The most of a region a cache uses. Blocks branch to the stubs at the
/// region's start with B and BL, which reach 128 MiB back.
pub const MAX_REGION: usize = 128 << 20;

/// Translated host code for one CPU.
pub struct CodeCache {
    region: JitRegion,
    /// How much of the region is used: all of it up to `MAX_REGION`.
    capacity: usize,
    /// Next free byte in the region for the code that runs, which grows up
    /// from the stubs.
    cursor: usize,
    /// Where the cold code (slow paths, exits taken once) starts: it grows
    /// down from the end of the capacity, so it stays out of the way of the
    /// code that runs and within reach of the stubs.
    cold: usize,
    /// Every live block, by key, to the offset of its entry.
    blocks: FastMap<BlockKey, usize>,
    /// The keys of the live blocks translated from each physical page.
    pages: FastMap<u64, Vec<BlockKey>>,
    stubs: Stubs,
    /// The memory helper the slow path calls.
    helper: u64,
    /// Fast paths on in the code this cache makes.
    pub fast: bool,
    /// Bumped whenever the cache is emptied, so a chain site recorded before
    /// it is known to be gone.
    epoch: u64,
}

/// A fault while translating: the region filled up.
#[derive(Debug)]
pub enum TranslateError {
    OutOfSpace,
}

impl CodeCache {
    /// A cache in `region`, whose slow path calls `helper`.
    pub fn with_region(region: JitRegion, helper: u64, fast: bool) -> Self {
        let mut cache = CodeCache {
            capacity: region.len().min(MAX_REGION),
            region,
            cursor: 0,
            cold: 0,
            blocks: FastMap::default(),
            pages: FastMap::default(),
            stubs: Stubs::default(),
            helper,
            fast,
            epoch: 0,
        };
        cache.cold = cache.capacity;
        cache.emit_stubs();
        cache
    }

    pub fn stubs(&self) -> &Stubs {
        &self.stubs
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    fn exec(&self, off: usize) -> u64 {
        self.region.exec_base() as u64 + off as u64
    }

    fn emit_stubs(&mut self) {
        // The save stub first: the others branch to it.
        let mut asm = Asm::new();
        emit::save_stub(&mut asm);
        self.stubs.save = self.place(asm);
        let mut asm = Asm::new();
        emit::mem_stub(&mut asm, self.helper);
        self.stubs.mem = self.place(asm);
        let stats = std::env::var_os("CRATE_JIT_STATS").is_some();
        for el in 0..2 {
            for ret in [false, true] {
                let mut asm = Asm::new();
                let base = self.exec(self.cursor);
                let (at, miss) = emit::jc_stub(&mut asm, el, self.stubs.save, stats, ret);
                self.place(asm);
                if ret {
                    self.stubs.jc_ret[el as usize] = base + at as u64;
                } else {
                    self.stubs.jc[el as usize] = base + at as u64;
                    self.stubs.jc_miss[el as usize] = base + miss as u64;
                }
            }
        }
    }

    /// Writes finished code at the cursor and returns its executable address.
    fn place(&mut self, asm: Asm) -> u64 {
        let base = self.exec(self.cursor);
        let words = asm.finish(base);
        self.put(&words);
        base
    }

    fn put(&mut self, words: &[u32]) {
        let bytes = std::mem::size_of_val(words);
        assert!(self.cursor + bytes <= self.cold, "code region full");
        let mut w = self.region.writer();
        w.write_words(self.cursor, words);
        drop(w);
        self.region.flush(self.cursor, bytes);
        self.cursor += bytes;
        // Keep blocks 16-byte aligned.
        self.cursor = self.cursor.next_multiple_of(16);
    }

    /// The entry of a cached block.
    pub fn lookup(&self, key: &BlockKey) -> Option<u64> {
        self.blocks.get(key).map(|&off| self.exec(off))
    }

    /// Translates the block for `key`, at guest `pc`, from `words` (the guest
    /// code from the PC to the end of its page) and returns its entry. On a
    /// full region the cache is emptied first (which the caller sees as a new
    /// epoch).
    pub fn translate(&mut self, key: BlockKey, pc: u64, words: &[u32], ctx: Ctx) -> u64 {
        let limit = if ctx.debug { 1 } else { MAX_BLOCK };
        let block = translate::translate(ctx, pc, words, limit);
        self.install(key, block, ctx.debug).unwrap_or_else(|_| {
            self.reset();
            let block = translate::translate(
                Ctx {
                    stubs: self.stubs,
                    ..ctx
                },
                pc,
                words,
                limit,
            );
            self.install(key, block, ctx.debug)
                .expect("a block in an empty cache")
        })
    }

    /// A block that only raises an instruction abort; run once, never cached.
    pub fn fetch_fault(&mut self, pc: u64, fsc: u8, ctx: Ctx) -> u64 {
        let block = translate::fetch_fault(ctx, pc, fsc);
        let key = BlockKey {
            pa: u64::MAX,
            mode: 0,
        };
        self.install(key, block, false).unwrap_or_else(|_| {
            self.reset();
            let block = translate::fetch_fault(
                Ctx {
                    stubs: self.stubs,
                    ..ctx
                },
                pc,
                fsc,
            );
            self.install(key, block, false)
                .expect("a block in an empty cache")
        })
    }

    fn install(
        &mut self,
        key: BlockKey,
        block: translate::Block,
        _debug: bool,
    ) -> Result<u64, TranslateError> {
        // Before resolving the block's branches: one to the stubs from past
        // the end of the capacity would not reach them.
        let cold = self.cold.saturating_sub(block.asm.cold_size()) & !15;
        if self.cursor + block.asm.size_bound() + 16 > cold {
            return Err(TranslateError::OutOfSpace);
        }
        let base = self.exec(self.cursor);
        let fetch_faulted = block.fetch_faulted;
        // The block's further entries (where its calls return to), each the
        // block for its own guest PC from now on.
        let entries: Vec<(BlockKey, usize)> = block
            .entries
            .iter()
            .map(|&(pc, label)| {
                let key = BlockKey {
                    pa: key.pa.wrapping_add(pc.wrapping_sub(block.start)),
                    mode: key.mode,
                };
                (key, block.asm.offset_of(label) as usize)
            })
            .collect();
        let (words, cold_words) = block.asm.finish_split(base, self.exec(cold));
        if !cold_words.is_empty() {
            let mut w = self.region.writer();
            w.write_words(cold, &cold_words);
            drop(w);
            self.region.flush(cold, cold_words.len() * 4);
            self.cold = cold;
        }
        let off = self.cursor;
        self.put(&words);
        if !fetch_faulted {
            let keys = self.pages.entry(key.pa >> 12).or_default();
            for (k, at) in std::iter::once((key, 0)).chain(entries) {
                if self.blocks.insert(k, off + at).is_none() {
                    keys.push(k);
                }
            }
        }
        Ok(self.exec(off))
    }

    /// Patches the branch at `site` (a host address in this region, recorded
    /// by a block's exit in `epoch`) to jump to the block entered at `entry`.
    pub fn chain(&mut self, site: u64, epoch: u64, entry: u64) {
        if epoch != self.epoch {
            return;
        }
        let base = self.region.exec_base() as u64;
        let end = base + self.cursor as u64;
        if site < base || site >= end {
            return;
        }
        let delta = entry.wrapping_sub(site) as i64;
        if !(-(1 << 27)..(1 << 27)).contains(&delta) {
            return;
        }
        let off = (site - base) as usize;
        let mut w = self.region.writer();
        w.patch_word(off, b(delta));
        drop(w);
        self.region.flush(off, 4);
    }

    /// Drops every block translated from the 4 KiB physical page at `pa`.
    /// Returns whether there were any.
    pub fn invalidate_page(&mut self, pa: u64) -> bool {
        match self.pages.remove(&(pa >> 12)) {
            Some(keys) => {
                for k in keys {
                    self.blocks.remove(&k);
                }
                true
            }
            None => false,
        }
    }

    /// Whether any block was translated from the 4 KiB physical page at `pa`.
    pub fn has_code(&self, pa: u64) -> bool {
        self.pages.contains_key(&(pa >> 12))
    }

    /// Empties the cache. Safe only between block runs.
    pub fn reset(&mut self) {
        self.cursor = 0;
        self.cold = self.capacity;
        self.blocks.clear();
        self.pages.clear();
        self.epoch += 1;
        self.emit_stubs();
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Bytes of code in use.
    pub fn used(&self) -> usize {
        self.cursor + (self.capacity - self.cold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_block_past_the_reach_of_the_stubs_starts_a_new_epoch() {
        let region = JitRegion::new_dual(MAX_REGION).unwrap();
        // The helper is never called: nothing here runs.
        let mut cache = CodeCache::with_region(region, 0x1000, true);
        // Just short of the end, where the block's branches back to the
        // stubs would be out of reach.
        cache.cursor = MAX_REGION - 16;
        let ctx = Ctx {
            el: 1,
            fast: true,
            debug: false,
            stubs: *cache.stubs(),
            mpidr: 0,
        };
        // add x0, x0, #1 (500 times); br x1: a long block, whose exit branch
        // to the stubs lies well past the cursor.
        let mut words = vec![0x9100_0400u32; 500];
        words.push(0xd61f_0020);
        let key = BlockKey {
            pa: 0x4000_0000,
            mode: 1,
        };
        let entry = cache.translate(key, 0x4000_0000, &words, ctx);
        assert_eq!(cache.epoch(), 1, "the cache was emptied");
        assert_eq!(cache.lookup(&key), Some(entry));
        assert!(cache.used() < 1 << 20);
    }

    #[test]
    fn a_call_returns_to_an_entry_of_its_block() {
        let region = JitRegion::new_dual(1 << 20).unwrap();
        let mut cache = CodeCache::with_region(region, 0x1000, true);
        let ctx = Ctx {
            el: 1,
            fast: true,
            debug: false,
            stubs: *cache.stubs(),
            mpidr: 0,
        };
        // add x0, x0, #1 ; bl +8 ; add x0, x0, #2 ; ret ; add x0, x0, #3
        let words = [
            0x9100_0400,
            0x9400_0002,
            0x9100_0800,
            0xd65f_03c0,
            0x9100_0c00,
        ];
        let key = BlockKey {
            pa: 0x4000_0000,
            mode: 1,
        };
        let entry = cache.translate(key, 0x4000_0000, &words, ctx);
        // The instruction after the call is an entry, after the call's BL.
        let after = BlockKey {
            pa: 0x4000_0008,
            mode: 1,
        };
        let ret = cache.lookup(&after).expect("an entry for the return point");
        assert!(ret > entry && ret < entry + 0x100);
        assert_eq!(cache.block_count(), 2);
        // Dropping the page drops both.
        assert!(cache.invalidate_page(0x4000_0000));
        assert_eq!(cache.lookup(&after), None);
        assert_eq!(cache.block_count(), 0);
    }
}
