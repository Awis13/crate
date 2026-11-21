// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! How fast the translator translates: blocks from a file of real arm64 code
//! (a kernel Image, say), at every 16th instruction of its first pages, into a
//! code cache. Ignored by default:
//!
//! ```text
//! CRATE_TRANSLATE_CORPUS=vmlinuz cargo test --release -p crate-jit \
//!     --test translate_speed -- --ignored --nocapture
//! ```

use std::time::Instant;

use crate_host::jit::JitRegion;
use crate_jit::cache::{BlockKey, CodeCache, MAX_REGION};
use crate_jit::translate::Ctx;

#[test]
#[ignore]
fn translation_speed() {
    let Some(path) = std::env::var_os("CRATE_TRANSLATE_CORPUS") else {
        eprintln!("set CRATE_TRANSLATE_CORPUS to a file of arm64 code");
        return;
    };
    let bytes = std::fs::read(path).expect("reading the corpus");
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    // Past an Image's header, its first 8 MiB: kernel text.
    let pages = (0x10000 / 4..words.len().min(0x80_0000 / 4)).step_by(1024);
    let region = JitRegion::new_dual(MAX_REGION).unwrap();
    // The helper is never called: nothing here runs.
    let mut cache = CodeCache::with_region(region, 0x1000, 0x2000, true);
    let (mut blocks, mut instructions) = (0u64, 0u64);
    let started = Instant::now();
    let rounds: usize = std::env::var("ROUNDS")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(3);
    for round in 0..rounds {
        for page in pages.clone() {
            let page_words = &words[page..(page + 1024).min(words.len())];
            for start in (0..page_words.len()).step_by(16) {
                let pc = 0x4000_0000 + (page * 4 + start * 4) as u64;
                let key = BlockKey {
                    pa: pc | (round as u64) << 40,
                    mode: 1,
                };
                let ctx = Ctx {
                    el: 1,
                    fast: true,
                    debug: false,
                    stubs: *cache.stubs(),
                    mpidr: 0,
                };
                cache.translate(key, pc, &page_words[start..], ctx);
                blocks += 1;
                instructions += (page_words.len() - start).min(16) as u64;
            }
        }
    }
    let took = started.elapsed();
    eprintln!(
        "{blocks} blocks in {:.3} s: {:.2} us a block ({instructions} instructions at least)",
        took.as_secs_f64(),
        took.as_secs_f64() * 1e6 / blocks as f64
    );
}
