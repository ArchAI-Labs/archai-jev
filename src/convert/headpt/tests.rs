//! Tests of the `head.pt` reader: a valid file, every way a file can be wrong, and hostile
//! pickles that must be refused without any effect.

use super::*;
use crate::convert::testing_pt::{HeadPt, Pickle, ZipSpec, zip_bytes};

fn read_bytes(bytes: &[u8]) -> Result<HeadTensors, Incompat> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("head.pt");
    std::fs::write(&path, bytes).unwrap();
    HeadPtReader.read(&path)
}

fn reason(r: Result<HeadTensors, Incompat>) -> String {
    match r {
        Err(Incompat::HeadWeights { detail }) => detail,
        Err(other) => panic!("not a head-weights error: {other}"),
        Ok(_) => panic!("the file was accepted"),
    }
}

/// A `head.pt` whose pickle is `pickle` (the rest of the archive is a valid one).
fn with_pickle(pickle: Vec<u8>) -> Vec<u8> {
    let mut entries = HeadPt::tiny().entries();
    entries[0].data = pickle;
    zip_bytes(&entries)
}

#[test]
fn a_valid_file_is_read_exactly() {
    let head = HeadPt::tiny();
    let got = read_bytes(&head.bytes()).unwrap();
    assert_eq!(got.tensors, head.tensors);
    assert_eq!(got.temperature, Some(2.351));
    assert_eq!(got.head_dim, Some(2));
    assert_eq!(got.d_model, Some(4));
    assert_eq!(got.base.as_deref(), Some("Org/Base"));
    assert_eq!(got.base_revision.as_deref(), Some("a".repeat(40).as_str()));
}

#[test]
fn metadata_is_optional_and_extra_keys_are_inert() {
    let mut head = HeadPt::tiny();
    head.temperature = None;
    head.base = None;
    head.base_revision = None;
    head.head_dim = None;
    let got = read_bytes(&head.bytes()).unwrap();
    assert_eq!(
        (got.temperature, got.head_dim, got.base, got.base_revision),
        (None, None, None, None)
    );
    // an entry the reader does not know does not matter
    let mut entries = HeadPt::tiny().entries();
    entries.push(ZipSpec::stored("head/.data/serialization_id", b"1234"));
    assert!(read_bytes(&zip_bytes(&entries)).is_ok());
}

// ---- row: the zip container ----

#[test]
fn what_is_not_an_acceptable_zip_is_refused() {
    assert!(reason(read_bytes(b"")).contains("too short"));
    assert!(reason(read_bytes(&[7u8; 4096])).contains("not a zip"));
    let good = HeadPt::tiny().bytes();
    assert!(reason(read_bytes(&good[..good.len() - 10])).contains("not a zip"));

    // compressed or encrypted entry
    let mut entries = HeadPt::tiny().entries();
    entries[0].method = 8;
    assert!(reason(read_bytes(&zip_bytes(&entries))).contains("compressed"));
    let mut entries = HeadPt::tiny().entries();
    entries[2].flags |= 0x0001;
    assert!(reason(read_bytes(&zip_bytes(&entries))).contains("encrypted"));

    // duplicate names
    let mut entries = HeadPt::tiny().entries();
    let again = entries[1].clone();
    entries.push(again);
    assert!(reason(read_bytes(&zip_bytes(&entries))).contains("twice"));

    // no data.pkl, two data.pkl, a storage without its data
    let mut entries = HeadPt::tiny().entries();
    entries.remove(0);
    assert!(reason(read_bytes(&zip_bytes(&entries))).contains("exactly one data.pkl"));
    let mut entries = HeadPt::tiny().entries();
    entries.push(ZipSpec::stored("other/data.pkl", b"x"));
    assert!(reason(read_bytes(&zip_bytes(&entries))).contains("exactly one data.pkl"));
    let mut entries = HeadPt::tiny().entries();
    entries.retain(|e| e.name != "head/data/2");
    assert!(reason(read_bytes(&zip_bytes(&entries))).contains("head/data/2"));

    // too many entries
    let mut entries = HeadPt::tiny().entries();
    for i in 0..70 {
        entries.push(ZipSpec::stored(&format!("head/x{i}"), b"."));
    }
    assert!(reason(read_bytes(&zip_bytes(&entries))).contains("entries"));
}

#[test]
fn a_damaged_directory_is_refused() {
    let good = HeadPt::tiny().bytes();
    let eocd = good.len() - 22;
    // directory offset beyond the file
    let mut bad = good.clone();
    bad[eocd + 16..eocd + 20].copy_from_slice(&(good.len() as u32 + 100).to_le_bytes());
    assert!(reason(read_bytes(&bad)).contains("outside"));
    // zip64 markers
    let mut bad = good.clone();
    bad[eocd + 10..eocd + 12].copy_from_slice(&0xFFFFu16.to_le_bytes());
    bad[eocd + 8..eocd + 10].copy_from_slice(&0xFFFFu16.to_le_bytes());
    assert!(reason(read_bytes(&bad)).contains("zip64"));
    // a split archive
    let mut bad = good.clone();
    bad[eocd + 4] = 1;
    assert!(reason(read_bytes(&bad)).contains("split"));
    // the first directory entry points at a place that is not a local header
    let cd = u32::from_le_bytes(good[eocd + 16..eocd + 20].try_into().unwrap()) as usize;
    let mut bad = good.clone();
    bad[cd + 42..cd + 46].copy_from_slice(&5u32.to_le_bytes());
    assert!(reason(read_bytes(&bad)).contains("local header"));
    // a directory entry whose size disagrees with its compressed size
    let mut bad = good.clone();
    bad[cd + 24] ^= 0x01;
    assert!(reason(read_bytes(&bad)).contains("inconsistent"));
    // a directory entry that claims more data than the file has
    let mut bad = good.clone();
    bad[cd + 20..cd + 24].copy_from_slice(&u32::MAX.to_le_bytes());
    bad[cd + 24..cd + 28].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(!reason(read_bytes(&bad)).is_empty());
}

// ---- row: opcodes outside the allow-list ----

#[test]
fn every_opcode_outside_the_allow_list_is_refused_with_its_offset() {
    for (name, op) in [
        ("INST", b'i'),
        ("OBJ", b'o'),
        ("NEWOBJ", 0x81),
        ("EXT1", 0x82),
        ("EXT2", 0x83),
        ("EXT4", 0x84),
        ("PERSID", b'P'),
        ("DUP", b'2'),
        ("POP", b'0'),
        ("STACK_GLOBAL", 0x93),
        ("NEWOBJ_EX", 0x92),
        ("MEMOIZE", 0x94),
        ("FRAME", 0x95),
        ("SHORT_BINBYTES", b'C'),
        ("BINBYTES", b'B'),
        ("GET", b'g'),
        ("PUT", b'p'),
        ("STRING", b'S'),
        ("UNICODE", b'V'),
        ("BYTEARRAY8", 0x96),
        ("PROTO-less junk", 0xFF),
    ] {
        let r = reason(read_bytes(&with_pickle(Pickle::new().raw(&[op]).0)));
        assert!(
            r.contains("allow-list") && r.contains("at byte 2"),
            "{name}: {r}"
        );
    }
    // a pickle of another protocol, or one that does not start with PROTO
    for proto in [0u8, 1, 3, 4, 5] {
        let r = reason(read_bytes(&with_pickle(vec![0x80, proto, b'.'])));
        assert!(r.contains("protocol"), "{r}");
    }
    let r = reason(read_bytes(&with_pickle(b"N.".to_vec())));
    assert!(r.contains("PROTO"), "{r}");
}

// ---- row: globals and calls outside the allow-list ----

#[test]
fn every_global_outside_the_allow_list_is_refused() {
    for (module, name) in [
        ("os", "system"),
        ("posix", "system"),
        ("nt", "system"),
        ("builtins", "eval"),
        ("builtins", "exec"),
        ("subprocess", "Popen"),
        ("collections", "defaultdict"),
        ("torch", "load"),
        ("torch", "HalfStorage"),
        ("torch", "BFloat16Storage"),
        ("torch._utils", "_rebuild_parameter"),
        ("torch._utils", "_rebuild_tensor"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("collections", "OrderedDict "),
    ] {
        let r = reason(read_bytes(&with_pickle(
            Pickle::new().global(module, name).stop().0,
        )));
        assert!(
            r.contains("allow-list") && r.contains("global"),
            "{module}.{name}: {r}"
        );
        assert!(r.contains(module), "{r}");
    }
}

#[test]
fn a_call_that_is_not_one_of_the_allowed_ones_is_refused() {
    // OrderedDict called with arguments
    let p = Pickle::new()
        .global("collections", "OrderedDict")
        .mark()
        .int(1)
        .tuple()
        .reduce()
        .stop();
    assert!(reason(read_bytes(&with_pickle(p.0))).contains("allowed call"));
    // calling the storage class
    let p = Pickle::new()
        .global("torch", "FloatStorage")
        .empty_tuple()
        .reduce()
        .stop();
    assert!(reason(read_bytes(&with_pickle(p.0))).contains("allowed call"));
    // calling something that is not a global at all
    let p = Pickle::new().int(1).empty_tuple().reduce().stop();
    assert!(reason(read_bytes(&with_pickle(p.0))).contains("allowed call"));
    // _rebuild_tensor_v2 with wrong arguments
    let p = Pickle::new()
        .global("torch._utils", "_rebuild_tensor_v2")
        .mark()
        .int(1)
        .tuple()
        .reduce()
        .stop();
    assert!(reason(read_bytes(&with_pickle(p.0))).contains("unexpected shape"));
    // BUILD on something that is not a dictionary, or with a state that is not one
    let p = Pickle::new().empty_tuple().empty_dict().build().stop();
    assert!(reason(read_bytes(&with_pickle(p.0))).contains("BUILD"));
    let p = Pickle::new().empty_dict().int(1).build().stop();
    assert!(reason(read_bytes(&with_pickle(p.0))).contains("BUILD"));
    // a persistent id of another shape
    let p = Pickle::new()
        .mark()
        .string("storage")
        .int(1)
        .tuple()
        .persid()
        .stop();
    assert!(reason(read_bytes(&with_pickle(p.0))).contains("persistent id"));
}

// ---- row: limits ----

#[test]
fn limits_are_enforced_before_anything_big_is_built() {
    // pickle over 1 MiB
    let big = vec![b'N'; pickle::MAX_PICKLE_BYTES + 1];
    assert!(reason(read_bytes(&with_pickle(big))).contains("maximum"));
    // a string that declares 4 GiB
    let p = Pickle::new()
        .raw(b"X")
        .raw(&u32::MAX.to_le_bytes())
        .raw(b"ab");
    assert!(reason(read_bytes(&with_pickle(p.0))).contains("limit"));
    // a stack that grows forever
    let mut p = Pickle::new();
    for _ in 0..5000 {
        p = p.mark();
    }
    assert!(reason(read_bytes(&with_pickle(p.0))).contains("stack"));
    // too many memo entries
    let mut p = Pickle::new().raw(b"N");
    for i in 0..5000u32 {
        p = p.raw(b"r").raw(&i.to_le_bytes());
    }
    assert!(reason(read_bytes(&with_pickle(p.0))).contains("memo"));
    // data that grows exponentially through the memo (a "pickle bomb")
    let mut p = Pickle::new().raw(b"]").raw(&[b'q', 0]);
    for _ in 0..40 {
        p = p.raw(&[b'h', 0, b'h', 0, 0x86, b'q', 0]);
    }
    assert!(reason(read_bytes(&with_pickle(p.0))).contains("too much data"));
    // deep nesting of the result
    let mut p = Pickle::new();
    for _ in 0..100 {
        p = p.mark();
    }
    for _ in 0..100 {
        p = p.tuple();
    }
    assert!(reason(read_bytes(&with_pickle(p.stop().0))).contains("nested"));
    // bytes after STOP
    assert!(
        reason(read_bytes(&with_pickle(b"\x80\x02N.junk".to_vec()))).contains("after the STOP")
    );
    // a pickle that ends in the middle of an opcode, or never stops
    assert!(
        reason(read_bytes(&with_pickle(
            b"\x80\x02X\x05\x00\x00\x00ab".to_vec()
        )))
        .contains("ends")
    );
    assert!(reason(read_bytes(&with_pickle(b"\x80\x02N".to_vec()))).contains("ends"));
    // a head with too many tensors, and a head that declares too much data
    let mut many = HeadPt::tiny();
    many.tensors = (0..17)
        .map(|i| (format!("t{i}"), vec![1], vec![0.0]))
        .collect();
    assert!(reason(read_bytes(&many.bytes())).contains("maximum"));
    let mut huge = HeadPt::tiny();
    huge.tensors = vec![("q.weight".into(), vec![1], vec![0.0])];
    huge.numel_first = Some(20_000_000);
    let mut entries = huge.entries();
    // make the pickle claim a 20M-element shape: rebuild it by hand
    entries[0].data = HeadPt {
        tensors: vec![("q.weight".into(), vec![20_000_000], vec![])],
        ..HeadPt::tiny()
    }
    .pickle()
    .0;
    assert!(reason(read_bytes(&zip_bytes(&entries))).contains("bytes of data"));
}

// ---- row: tensors ----

#[test]
fn tensor_defects_are_each_refused() {
    let mut h = HeadPt::tiny();
    h.stride_first = Some(vec![1, 2]);
    assert!(reason(read_bytes(&h.bytes())).contains("contiguous"));
    let mut h = HeadPt::tiny();
    h.offset_first = 4;
    assert!(reason(read_bytes(&h.bytes())).contains("offset"));
    let mut h = HeadPt::tiny();
    h.numel_first = Some(9);
    assert!(reason(read_bytes(&h.bytes())).contains("storage has 9"));
    let mut h = HeadPt::tiny();
    h.share_storage = true;
    assert!(reason(read_bytes(&h.bytes())).contains("share"));
    // data of the wrong length
    let mut entries = HeadPt::tiny().entries();
    entries[2].data.truncate(10);
    assert!(reason(read_bytes(&zip_bytes(&entries))).contains("needs 32"));
    // a non-finite value
    let mut h = HeadPt::tiny();
    h.tensors[1].2[0] = f32::NAN;
    assert!(reason(read_bytes(&h.bytes())).contains("non-finite"));
    let mut h = HeadPt::tiny();
    h.tensors[3].2[1] = f32::INFINITY;
    assert!(reason(read_bytes(&h.bytes())).contains("non-finite"));
    // shapes
    let mut h = HeadPt::tiny();
    h.tensors[0].1 = vec![2, 0, 4];
    assert!(reason(read_bytes(&h.bytes())).contains("shape"));
    let mut h = HeadPt::tiny();
    h.tensors[0].1 = vec![1, 1, 1, 1, 8];
    assert!(reason(read_bytes(&h.bytes())).contains("shape"));
}

// ---- row: what the file says about itself ----

#[test]
fn the_dictionary_and_its_metadata_are_checked() {
    // no 'head' dictionary at all
    let p = Pickle::new().empty_dict().stop();
    assert!(reason(read_bytes(&with_pickle(p.0))).contains("no 'head'"));
    // 'head' that is not a dictionary of tensors
    let p = Pickle::new()
        .empty_dict()
        .mark()
        .string("head")
        .empty_dict()
        .mark()
        .string("q.weight")
        .int(3)
        .setitems()
        .setitems()
        .stop();
    assert!(reason(read_bytes(&with_pickle(p.0))).contains("not a tensor"));
    // metadata of the wrong type
    for (key, value_pickle) in [
        ("temperature", Pickle::new().string("hot")),
        ("head_dim", Pickle::new().string("big")),
        ("base", Pickle::new().int(1)),
        ("base_revision", Pickle::new().int(1)),
        ("head_dim", Pickle::new().raw(&[0x8a, 1, 0xFF])), // -1
    ] {
        let mut p = Pickle::new()
            .empty_dict()
            .mark()
            .string("head")
            .empty_dict()
            .string(key);
        p.0.extend_from_slice(&value_pickle.0[2..]);
        let p = p.setitems().stop();
        let r = reason(read_bytes(&with_pickle(p.0)));
        assert!(r.contains(key), "{key}: {r}");
    }
    // a temperature that is not finite
    let mut h = HeadPt::tiny();
    h.temperature = Some(f64::INFINITY);
    assert!(reason(read_bytes(&h.bytes())).contains("temperature"));
}

#[test]
fn a_missing_file_is_a_head_weights_error() {
    let r = HeadPtReader.read(Path::new("/nonexistent/head.pt"));
    assert!(reason(r).contains("cannot open"));
}

// ---- the real file (level T1) ----

/// SHA-256 of the f32 bytes of each tensor of Kev-0.8B's head, from `torch.load(weights_only=True)`.
const KEV_TENSOR_DIGESTS: [(&str, [u64; 2], &str); 4] = [
    (
        "q.weight",
        [256, 1024],
        "8b0c5514125d8177b17c8cb63418e1b4efb102626bb4cebbd44388a318fe6ad9",
    ),
    (
        "q.bias",
        [256, 0],
        "6f8bea894a8a5347a6ebe334a9afb64ee1d5ba48f90efd80f5de65c1a31a0e9a",
    ),
    (
        "k.weight",
        [256, 1024],
        "e43aa59692a4a2d09cd8c0beb9ae7ea9c18b2bc5a44a37a60aa4323b53da1b0d",
    ),
    (
        "k.bias",
        [256, 0],
        "a02d492fd4372da9cb0053e5ed450607323b660bad899acae9df6df7e82d6169",
    ),
];

#[test]
fn the_real_kev_head_matches_torch_load_bit_for_bit() {
    let Some(path) = crate::models::testing::assets::kev_head() else {
        return;
    };
    let got = HeadPtReader.read(&path).unwrap();
    assert_eq!(got.tensors.len(), 4);
    for ((name, shape, values), (want_name, want_shape, want_sha)) in
        got.tensors.iter().zip(KEV_TENSOR_DIGESTS)
    {
        assert_eq!(name, want_name);
        let want: Vec<u64> = want_shape.iter().copied().filter(|d| *d != 0).collect();
        assert_eq!(*shape, want, "{name}");
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(crate::models::hash::sha256_hex(&bytes), want_sha, "{name}");
    }
    assert_eq!(got.temperature, Some(2.351_095_812_567_217_4));
    assert_eq!(got.head_dim, Some(256));
    assert_eq!(got.d_model, Some(1024));
    assert_eq!(got.base.as_deref(), Some("Qwen/Qwen3.5-0.8B-Base"));
    assert_eq!(
        got.base_revision.as_deref(),
        Some("dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
    );
}

// ---- robustness ----

#[test]
fn random_damage_never_panics() {
    // splitmix64, fixed seed: a deterministic stream of changes to a valid file
    let mut state = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let good = HeadPt::tiny().bytes();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("head.pt");
    for case in 0..3000 {
        let mut bytes = good.clone();
        for _ in 0..=(next() % 4) {
            let at = (next() as usize) % bytes.len();
            bytes[at] = next() as u8;
        }
        if case % 7 == 0 {
            bytes.truncate((next() as usize) % bytes.len());
        }
        std::fs::write(&path, &bytes).unwrap();
        // the outcome is Ok or a head-weights error: reaching the end is the assertion
        if let Err(e) = HeadPtReader.read(&path) {
            assert!(matches!(e, Incompat::HeadWeights { .. }), "{e}");
        }
    }
    // the same for the pickle alone
    let pickle = HeadPt::tiny().pickle().0;
    for _ in 0..3000 {
        let mut bytes = pickle.clone();
        for _ in 0..=(next() % 4) {
            let at = (next() as usize) % bytes.len();
            bytes[at] = next() as u8;
        }
        if next() % 5 == 0 {
            bytes.truncate((next() as usize) % bytes.len());
        }
        let _ = pickle::interpret(&bytes);
    }
}

#[test]
fn a_hostile_pickle_has_no_effect() {
    // os.system("...") would create this file if anything executed it
    let dir = tempfile::tempdir().unwrap();
    let marker = dir.path().join("marker");
    let command = format!("echo x > {}", marker.display());
    let p = Pickle::new()
        .global("os", "system")
        .mark()
        .string(&command)
        .tuple()
        .reduce()
        .stop();
    let r = reason(read_bytes(&with_pickle(p.0)));
    assert!(r.contains("os.system"), "{r}");
    assert!(!marker.exists());
}
