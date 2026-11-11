// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Translated code: one region of executable memory per CPU, the shared stubs
//! at its start, and the blocks after them.
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
use crate::emit::{self, CHAIN_ENTRY, Stubs};
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
/// region's start with B and BL, which reach 128 MiB.
pub const MAX_REGION: usize = 128 << 20;

/// Translated host code for one CPU.
pub struct CodeCache {
    region: JitRegion,
    /// How much of the region is used: all of it up to `MAX_REGION`.
    capacity: usize,
    /// Next free byte in the region.
    cursor: usize,
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
            blocks: FastMap::default(),
            pages: FastMap::default(),
            stubs: Stubs::default(),
            helper,
            fast,
            epoch: 0,
        };
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
        for el in 0..2 {
            let mut asm = Asm::new();
            let base = self.exec(self.cursor);
            let stats = std::env::var_os("CRATE_JIT_STATS").is_some();
            let (at, miss) = emit::jc_stub(&mut asm, el, self.stubs.save, stats);
            self.place(asm);
            self.stubs.jc[el as usize] = base + at as u64;
            self.stubs.jc_miss[el as usize] = base + miss as u64;
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
        assert!(self.cursor + bytes <= self.capacity, "code region full");
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
        let base = self.exec(self.cursor);
        let fetch_faulted = block.fetch_faulted;
        let words = block.asm.finish(base);
        let bytes = std::mem::size_of_val(words.as_slice());
        if self.cursor + bytes + 16 > self.capacity {
            return Err(TranslateError::OutOfSpace);
        }
        let off = self.cursor;
        self.put(&words);
        if !fetch_faulted {
            self.blocks.insert(key, off);
            self.pages.entry(key.pa >> 12).or_default().push(key);
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
        let target = entry + CHAIN_ENTRY;
        let delta = target.wrapping_sub(site) as i64;
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
        self.cursor
    }
}
