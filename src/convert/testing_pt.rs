//! Zip and pickle writers for tests of the `head.pt` reader (feature `testing`). The library
//! never writes these formats: they exist to build valid files and to break them in one way.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    missing_docs
)]

/// An entry of a zip archive to write.
#[derive(Debug, Clone)]
pub struct ZipSpec {
    pub name: String,
    pub data: Vec<u8>,
    /// 0 = stored. Anything else writes the field but the data stays as it is (a lie, on purpose).
    pub method: u16,
    /// General purpose flags (bit 3 = sizes in a data descriptor, as `torch.save` writes).
    pub flags: u16,
    /// Bytes of padding in the local extra field (`torch.save` aligns data to 64 bytes).
    pub local_extra: usize,
}

impl ZipSpec {
    /// A stored entry written like `torch.save` does (data descriptor, aligned).
    pub fn stored(name: &str, data: &[u8]) -> ZipSpec {
        ZipSpec {
            name: name.to_string(),
            data: data.to_vec(),
            method: 0,
            flags: 0x0808,
            local_extra: 4,
        }
    }
}

/// Write a zip archive of `entries`.
pub fn zip_bytes(entries: &[ZipSpec]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut central = Vec::new();
    for e in entries {
        let offset = out.len() as u32;
        let descriptor = e.flags & 0x0008 != 0;
        out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
        out.extend_from_slice(&20u16.to_le_bytes());
        out.extend_from_slice(&e.flags.to_le_bytes());
        out.extend_from_slice(&e.method.to_le_bytes());
        out.extend_from_slice(&[0u8; 4]); // time, date
        out.extend_from_slice(&0u32.to_le_bytes()); // crc (not checked by the reader)
        let (c, u) = if descriptor {
            (0u32, 0u32)
        } else {
            (e.data.len() as u32, e.data.len() as u32)
        };
        out.extend_from_slice(&c.to_le_bytes());
        out.extend_from_slice(&u.to_le_bytes());
        out.extend_from_slice(&(e.name.len() as u16).to_le_bytes());
        out.extend_from_slice(&(e.local_extra as u16).to_le_bytes());
        out.extend_from_slice(e.name.as_bytes());
        out.extend(std::iter::repeat_n(0u8, e.local_extra));
        out.extend_from_slice(&e.data);
        if descriptor {
            out.extend_from_slice(&0x0807_4b50u32.to_le_bytes());
            out.extend_from_slice(&0u32.to_le_bytes());
            out.extend_from_slice(&(e.data.len() as u32).to_le_bytes());
            out.extend_from_slice(&(e.data.len() as u32).to_le_bytes());
        }
        central.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
        central.extend_from_slice(&20u16.to_le_bytes());
        central.extend_from_slice(&20u16.to_le_bytes());
        central.extend_from_slice(&e.flags.to_le_bytes());
        central.extend_from_slice(&e.method.to_le_bytes());
        central.extend_from_slice(&[0u8; 4]);
        central.extend_from_slice(&0u32.to_le_bytes());
        central.extend_from_slice(&(e.data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(e.data.len() as u32).to_le_bytes());
        central.extend_from_slice(&(e.name.len() as u16).to_le_bytes());
        central.extend_from_slice(&0u16.to_le_bytes()); // extra
        central.extend_from_slice(&0u16.to_le_bytes()); // comment
        central.extend_from_slice(&0u16.to_le_bytes()); // disk
        central.extend_from_slice(&0u16.to_le_bytes()); // internal attributes
        central.extend_from_slice(&0u32.to_le_bytes()); // external attributes
        central.extend_from_slice(&offset.to_le_bytes());
        central.extend_from_slice(e.name.as_bytes());
    }
    let cd_offset = out.len() as u32;
    out.extend_from_slice(&central);
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&[0u8; 4]);
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    out.extend_from_slice(&(central.len() as u32).to_le_bytes());
    out.extend_from_slice(&cd_offset.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}

/// A pickle (protocol 2) written opcode by opcode, to build valid and hostile ones.
#[derive(Debug, Clone, Default)]
pub struct Pickle(pub Vec<u8>);

impl Pickle {
    pub fn new() -> Pickle {
        Pickle(vec![0x80, 2])
    }
    pub fn raw(mut self, bytes: &[u8]) -> Pickle {
        self.0.extend_from_slice(bytes);
        self
    }
    pub fn mark(self) -> Pickle {
        self.raw(b"(")
    }
    pub fn empty_dict(self) -> Pickle {
        self.raw(b"}")
    }
    pub fn empty_tuple(self) -> Pickle {
        self.raw(b")")
    }
    pub fn tuple(self) -> Pickle {
        self.raw(b"t")
    }
    pub fn setitems(self) -> Pickle {
        self.raw(b"u")
    }
    pub fn reduce(self) -> Pickle {
        self.raw(b"R")
    }
    pub fn build(self) -> Pickle {
        self.raw(b"b")
    }
    pub fn persid(self) -> Pickle {
        self.raw(b"Q")
    }
    pub fn newfalse(self) -> Pickle {
        self.raw(&[0x89])
    }
    pub fn stop(self) -> Pickle {
        self.raw(b".")
    }
    pub fn string(self, s: &str) -> Pickle {
        let mut p = self.raw(b"X");
        p.0.extend_from_slice(&(s.len() as u32).to_le_bytes());
        p.raw(s.as_bytes())
    }
    pub fn int(self, n: i64) -> Pickle {
        if (0..256).contains(&n) {
            self.raw(&[b'K', n as u8])
        } else {
            let mut p = self.raw(b"J");
            p.0.extend_from_slice(&(n as i32).to_le_bytes());
            p
        }
    }
    pub fn float(self, f: f64) -> Pickle {
        let mut p = self.raw(b"G");
        p.0.extend_from_slice(&f.to_be_bytes());
        p
    }
    pub fn global(self, module: &str, name: &str) -> Pickle {
        self.raw(b"c")
            .raw(module.as_bytes())
            .raw(b"\n")
            .raw(name.as_bytes())
            .raw(b"\n")
    }
    pub fn ordered_dict(self) -> Pickle {
        self.global("collections", "OrderedDict")
            .empty_tuple()
            .reduce()
    }
    pub fn ints_tuple(self, v: &[u64]) -> Pickle {
        let mut p = self.mark();
        for n in v {
            p = p.int(*n as i64);
        }
        p.tuple()
    }
}

/// What to put in a `head.pt` (all the dials the tests turn).
#[derive(Debug, Clone)]
pub struct HeadPt {
    /// `(name, shape, values)`.
    pub tensors: Vec<(String, Vec<u64>, Vec<f32>)>,
    pub temperature: Option<f64>,
    pub head_dim: Option<u64>,
    pub base: Option<String>,
    pub base_revision: Option<String>,
    /// Folder name inside the archive (`torch.save` uses the file stem).
    pub prefix: String,
    /// Write these strides for the first tensor instead of the contiguous ones.
    pub stride_first: Option<Vec<u64>>,
    /// Write this storage offset for the first tensor.
    pub offset_first: i64,
    /// Write this number of elements for the storage of the first tensor.
    pub numel_first: Option<u64>,
    /// Make the second tensor use the storage of the first.
    pub share_storage: bool,
}

impl HeadPt {
    /// A small valid head: `d_model` 4, `proj_dim` 2.
    pub fn tiny() -> HeadPt {
        let w = |seed: f32| (0..8).map(|i| seed + i as f32 * 0.25).collect::<Vec<f32>>();
        HeadPt {
            tensors: vec![
                ("q.weight".into(), vec![2, 4], w(0.5)),
                ("q.bias".into(), vec![2], vec![0.125, -0.25]),
                ("k.weight".into(), vec![2, 4], w(-1.0)),
                ("k.bias".into(), vec![2], vec![0.5, 0.75]),
            ],
            temperature: Some(2.351),
            head_dim: Some(2),
            base: Some("Org/Base".into()),
            base_revision: Some("a".repeat(40)),
            prefix: "head".into(),
            stride_first: None,
            offset_first: 0,
            numel_first: None,
            share_storage: false,
        }
    }

    /// The pickle, in the shape `torch.save` writes it.
    pub fn pickle(&self) -> Pickle {
        let mut p = Pickle::new().empty_dict().mark();
        if let Some(b) = &self.base {
            p = p.string("base").string(b);
        }
        if let Some(r) = &self.base_revision {
            p = p.string("base_revision").string(r);
        }
        if let Some(h) = self.head_dim {
            p = p.string("head_dim").int(h as i64);
        }
        if let Some(t) = self.temperature {
            p = p.string("temperature").float(t);
        }
        p = p.string("head").ordered_dict().mark();
        for (i, (name, shape, _)) in self.tensors.iter().enumerate() {
            let mut numel: u64 = shape.iter().product();
            let mut stride = Vec::new();
            let mut acc = 1u64;
            for d in shape.iter().rev() {
                stride.push(acc);
                acc *= d;
            }
            stride.reverse();
            let mut offset = 0;
            if i == 0 {
                numel = self.numel_first.unwrap_or(numel);
                offset = self.offset_first;
                if let Some(s) = &self.stride_first {
                    stride = s.clone();
                }
            }
            let key = if self.share_storage && i == 1 { 0 } else { i };
            p = p
                .string(name)
                .global("torch._utils", "_rebuild_tensor_v2")
                .mark()
                .mark()
                .string("storage")
                .global("torch", "FloatStorage")
                .string(&key.to_string())
                .string("cpu")
                .int(numel as i64)
                .tuple()
                .persid()
                .int(offset)
                .ints_tuple(shape)
                .ints_tuple(&stride)
                .newfalse()
                .ordered_dict()
                .tuple()
                .reduce();
        }
        p.setitems()
            .empty_dict()
            .mark()
            .string("_metadata")
            .ordered_dict()
            .setitems()
            .build()
            .setitems()
            .stop()
    }

    /// The zip entries of a `torch.save` file for this head.
    pub fn entries(&self) -> Vec<ZipSpec> {
        let mut out = vec![ZipSpec::stored(
            &format!("{}/data.pkl", self.prefix),
            &self.pickle().0,
        )];
        out.push(ZipSpec::stored(
            &format!("{}/byteorder", self.prefix),
            b"little",
        ));
        for (i, (_, _, values)) in self.tensors.iter().enumerate() {
            let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            out.push(ZipSpec::stored(&format!("{}/data/{i}", self.prefix), &data));
        }
        out.push(ZipSpec::stored(&format!("{}/version", self.prefix), b"3\n"));
        out
    }

    /// The whole file.
    pub fn bytes(&self) -> Vec<u8> {
        zip_bytes(&self.entries())
    }
}
