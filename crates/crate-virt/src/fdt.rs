// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Flattened device trees (the Devicetree Specification, chapter 5): a writer
//! for the tree a machine hands its guest, and a reader to check it.

use std::collections::HashMap;

const MAGIC: u32 = 0xd00d_feed;
const VERSION: u32 = 17;
const LAST_COMPATIBLE_VERSION: u32 = 16;
const HEADER_SIZE: usize = 40;
const BEGIN_NODE: u32 = 1;
const END_NODE: u32 = 2;
const PROP: u32 = 3;
const NOP: u32 = 4;
const END: u32 = 9;

#[derive(Default)]
pub struct FdtWriter {
    structure: Vec<u8>,
    strings: Vec<u8>,
    string_offsets: HashMap<String, u32>,
    depth: usize,
}

impl FdtWriter {
    pub fn new() -> Self {
        Self::default()
    }

    fn token(&mut self, token: u32) {
        self.structure.extend_from_slice(&token.to_be_bytes());
    }

    fn pad(&mut self) {
        while self.structure.len() % 4 != 0 {
            self.structure.push(0);
        }
    }

    /// Opens a node; the root is the one named "".
    pub fn begin_node(&mut self, name: &str) {
        self.token(BEGIN_NODE);
        self.structure.extend_from_slice(name.as_bytes());
        self.structure.push(0);
        self.pad();
        self.depth += 1;
    }

    pub fn end_node(&mut self) {
        assert!(self.depth > 0, "end_node without begin_node");
        self.token(END_NODE);
        self.depth -= 1;
    }

    pub fn property(&mut self, name: &str, value: &[u8]) {
        let offset = match self.string_offsets.get(name) {
            Some(&o) => o,
            None => {
                let o = self.strings.len() as u32;
                self.strings.extend_from_slice(name.as_bytes());
                self.strings.push(0);
                self.string_offsets.insert(name.to_string(), o);
                o
            }
        };
        self.token(PROP);
        self.structure
            .extend_from_slice(&(value.len() as u32).to_be_bytes());
        self.structure.extend_from_slice(&offset.to_be_bytes());
        self.structure.extend_from_slice(value);
        self.pad();
    }

    pub fn property_empty(&mut self, name: &str) {
        self.property(name, &[]);
    }

    pub fn property_u32(&mut self, name: &str, value: u32) {
        self.property(name, &value.to_be_bytes());
    }

    pub fn property_u64(&mut self, name: &str, value: u64) {
        self.property(name, &value.to_be_bytes());
    }

    pub fn property_cells(&mut self, name: &str, cells: &[u32]) {
        let bytes: Vec<u8> = cells.iter().flat_map(|c| c.to_be_bytes()).collect();
        self.property(name, &bytes);
    }

    pub fn property_string(&mut self, name: &str, value: &str) {
        self.property_strings(name, &[value]);
    }

    pub fn property_strings(&mut self, name: &str, values: &[&str]) {
        let mut bytes = Vec::new();
        for v in values {
            bytes.extend_from_slice(v.as_bytes());
            bytes.push(0);
        }
        self.property(name, &bytes);
    }

    /// A `reg` of (address, size) pairs in two cells each.
    pub fn property_reg(&mut self, ranges: &[(u64, u64)]) {
        let cells: Vec<u32> = ranges
            .iter()
            .flat_map(|&(a, s)| [(a >> 32) as u32, a as u32, (s >> 32) as u32, s as u32])
            .collect();
        self.property_cells("reg", &cells);
    }

    pub fn finish(mut self, boot_cpuid: u32) -> Vec<u8> {
        assert_eq!(self.depth, 0, "unclosed nodes");
        self.token(END);
        // Header, an empty memory reservation block, structure, strings.
        let reserve_offset = HEADER_SIZE;
        let reserve_size = 16;
        let struct_offset = reserve_offset + reserve_size;
        let strings_offset = struct_offset + self.structure.len();
        let total = strings_offset + self.strings.len();
        let mut out = Vec::with_capacity(total);
        for field in [
            MAGIC,
            total as u32,
            struct_offset as u32,
            strings_offset as u32,
            reserve_offset as u32,
            VERSION,
            LAST_COMPATIBLE_VERSION,
            boot_cpuid,
            self.strings.len() as u32,
            self.structure.len() as u32,
        ] {
            out.extend_from_slice(&field.to_be_bytes());
        }
        out.extend_from_slice(&[0; 16]);
        out.extend_from_slice(&self.structure);
        out.extend_from_slice(&self.strings);
        out
    }
}

/// A node read back from a blob.
#[derive(Debug, Default)]
pub struct Node {
    pub name: String,
    pub properties: Vec<(String, Vec<u8>)>,
    pub children: Vec<Node>,
}

impl Node {
    pub fn property(&self, name: &str) -> Option<&[u8]> {
        self.properties
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, v)| v.as_slice())
    }

    pub fn cells(&self, name: &str) -> Option<Vec<u32>> {
        let v = self.property(name)?;
        (v.len() % 4 == 0).then(|| {
            v.chunks(4)
                .map(|c| u32::from_be_bytes(c.try_into().unwrap()))
                .collect()
        })
    }

    pub fn strings(&self, name: &str) -> Option<Vec<String>> {
        let v = self.property(name)?;
        let v = v.strip_suffix(&[0])?;
        Some(
            v.split(|&b| b == 0)
                .map(|s| String::from_utf8_lossy(s).into_owned())
                .collect(),
        )
    }

    pub fn child(&self, name: &str) -> Option<&Node> {
        self.children.iter().find(|c| c.name == name)
    }

    /// The node at a path like "/cpus/cpu@0".
    pub fn find(&self, path: &str) -> Option<&Node> {
        path.split('/')
            .filter(|p| !p.is_empty())
            .try_fold(self, |node, part| node.child(part))
    }

    /// A rough DTS rendering, for reading by eye.
    pub fn to_dts(&self) -> String {
        let mut out = String::from("/dts-v1/;\n\n");
        self.render(0, &mut out);
        out
    }

    fn render(&self, depth: usize, out: &mut String) {
        let indent = "\t".repeat(depth);
        let name = if self.name.is_empty() {
            "/"
        } else {
            &self.name
        };
        out.push_str(&format!("{indent}{name} {{\n"));
        for (n, v) in &self.properties {
            let printable = !v.is_empty()
                && v.last() == Some(&0)
                && v[..v.len() - 1]
                    .iter()
                    .all(|&b| b == 0 || (0x20..0x7f).contains(&b))
                && v.first() != Some(&0);
            if v.is_empty() {
                out.push_str(&format!("{indent}\t{n};\n"));
            } else if printable {
                let parts: Vec<String> = v[..v.len() - 1]
                    .split(|&b| b == 0)
                    .map(|s| format!("\"{}\"", String::from_utf8_lossy(s)))
                    .collect();
                out.push_str(&format!("{indent}\t{n} = {};\n", parts.join(", ")));
            } else if v.len() % 4 == 0 {
                let cells: Vec<String> = v
                    .chunks(4)
                    .map(|c| format!("{:#x}", u32::from_be_bytes(c.try_into().unwrap())))
                    .collect();
                out.push_str(&format!("{indent}\t{n} = <{}>;\n", cells.join(" ")));
            } else {
                let bytes: Vec<String> = v.iter().map(|b| format!("{b:02x}")).collect();
                out.push_str(&format!("{indent}\t{n} = [{}];\n", bytes.join(" ")));
            }
        }
        for c in &self.children {
            out.push('\n');
            c.render(depth + 1, out);
        }
        out.push_str(&format!("{indent}}};\n"));
    }
}

fn be32(blob: &[u8], at: usize) -> Result<u32, String> {
    blob.get(at..at + 4)
        .map(|b| u32::from_be_bytes(b.try_into().unwrap()))
        .ok_or_else(|| format!("truncated at {at:#x}"))
}

/// Parses a blob back into its tree, checking the header and structure.
pub fn parse(blob: &[u8]) -> Result<Node, String> {
    if be32(blob, 0)? != MAGIC {
        return Err("bad magic".into());
    }
    let total = be32(blob, 4)? as usize;
    let struct_offset = be32(blob, 8)? as usize;
    let strings_offset = be32(blob, 12)? as usize;
    if total != blob.len() {
        return Err(format!(
            "totalsize {total} but the blob is {} bytes",
            blob.len()
        ));
    }
    let strings_size = be32(blob, 32)? as usize;
    let struct_size = be32(blob, 36)? as usize;
    let strings = blob
        .get(strings_offset..strings_offset + strings_size)
        .ok_or("strings block out of range")?;
    let structure = blob
        .get(struct_offset..struct_offset + struct_size)
        .ok_or("structure block out of range")?;
    let mut at = 0;
    let mut stack: Vec<Node> = Vec::new();
    let mut root = None;
    loop {
        let token = be32(structure, at)?;
        at += 4;
        match token {
            BEGIN_NODE => {
                let end = structure[at..]
                    .iter()
                    .position(|&b| b == 0)
                    .ok_or("unterminated node name")?;
                let name = String::from_utf8_lossy(&structure[at..at + end]).into_owned();
                at = (at + end + 1).div_ceil(4) * 4;
                stack.push(Node {
                    name,
                    ..Node::default()
                });
            }
            END_NODE => {
                let node = stack.pop().ok_or("END_NODE without BEGIN_NODE")?;
                match stack.last_mut() {
                    Some(parent) => parent.children.push(node),
                    None => root = Some(node),
                }
            }
            PROP => {
                let len = be32(structure, at)? as usize;
                let name_offset = be32(structure, at + 4)? as usize;
                at += 8;
                let value = structure
                    .get(at..at + len)
                    .ok_or("property out of range")?
                    .to_vec();
                at = (at + len).div_ceil(4) * 4;
                let name_end = strings[name_offset..]
                    .iter()
                    .position(|&b| b == 0)
                    .ok_or("bad property name")?;
                let name = String::from_utf8_lossy(&strings[name_offset..name_offset + name_end])
                    .into_owned();
                stack
                    .last_mut()
                    .ok_or("property outside a node")?
                    .properties
                    .push((name, value));
            }
            NOP => {}
            END => break,
            t => return Err(format!("unknown token {t:#x}")),
        }
    }
    if !stack.is_empty() {
        return Err("unclosed nodes".into());
    }
    root.ok_or_else(|| "no root node".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let mut w = FdtWriter::new();
        w.begin_node("");
        w.property_u32("#address-cells", 2);
        w.property_strings("compatible", &["a,b", "c"]);
        w.begin_node("memory@40000000");
        w.property_string("device_type", "memory");
        w.property_reg(&[(0x4000_0000, 0x8000_0000)]);
        w.end_node();
        w.begin_node("x");
        w.property_empty("dma-coherent");
        w.property_u32("#address-cells", 1);
        w.end_node();
        w.end_node();
        let blob = w.finish(0);
        let root = parse(&blob).unwrap();
        assert_eq!(root.cells("#address-cells"), Some(vec![2]));
        assert_eq!(
            root.strings("compatible"),
            Some(vec!["a,b".into(), "c".into()])
        );
        let mem = root.find("/memory@40000000").unwrap();
        assert_eq!(mem.cells("reg"), Some(vec![0, 0x4000_0000, 0, 0x8000_0000]));
        assert_eq!(
            root.find("/x").unwrap().property("dma-coherent"),
            Some(&[][..])
        );
        let strings = &blob[be32(&blob, 12).unwrap() as usize..];
        assert_eq!(
            strings
                .windows(14)
                .filter(|w| w == b"#address-cells")
                .count(),
            1,
            "property names are stored once"
        );
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse(&[0; 64]).is_err());
        let mut w = FdtWriter::new();
        w.begin_node("");
        w.end_node();
        let mut blob = w.finish(0);
        blob.push(0);
        assert!(parse(&blob).is_err(), "totalsize mismatch");
    }
}
