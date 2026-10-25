//! A pickle reader that **never executes anything** (spec 018, section 4, decision D26).
//!
//! A pickle is a program for a small stack machine, and the machine can be told to call any
//! function (`os.system`, ...). So we do not run it: we *interpret* it symbolically. Every
//! opcode builds plain data (numbers, strings, lists, dictionaries) or, for the three names the
//! `torch.save` format uses, an inert marker. The set of accepted opcodes and the set of accepted
//! globals are closed lists; anything else stops the reading with an error naming the opcode or
//! the global and its offset. The amount of data a pickle can make us build is limited too.

use std::collections::HashMap;

/// Largest pickle we read.
pub const MAX_PICKLE_BYTES: usize = 1024 * 1024;
const MAX_STRING_BYTES: usize = 1024 * 1024;
const MAX_STACK: usize = 4096;
const MAX_MEMO: usize = 4096;
const MAX_NODES: usize = 200_000;
const MAX_DEPTH: usize = 64;

/// The only three globals `torch.save` writes for a plain state dictionary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Global {
    /// `collections.OrderedDict`
    OrderedDict,
    /// `torch._utils._rebuild_tensor_v2`
    RebuildTensorV2,
    /// `torch.FloatStorage`
    FloatStorage,
}

/// A value the machine built.
#[derive(Debug, Clone, PartialEq)]
pub enum Val {
    /// `None`
    None,
    /// A boolean.
    Bool(bool),
    /// An integer (64 bits).
    Int(i64),
    /// A float.
    Float(f64),
    /// A string.
    Str(String),
    /// A list.
    List(Vec<Val>),
    /// A tuple.
    Tuple(Vec<Val>),
    /// A dictionary (also an `OrderedDict`), in insertion order.
    Dict(Vec<(Val, Val)>),
    /// One of the allowed globals, not called.
    Global(Global),
    /// A reference to a storage (`persistent_load`): key in the archive and number of elements.
    Storage {
        /// Key of the data entry.
        key: String,
        /// Number of `f32` elements.
        numel: u64,
    },
    /// A tensor over a storage.
    Tensor {
        /// Key of the data entry.
        key: String,
        /// Elements in the storage.
        numel: u64,
        /// Offset into the storage, in elements.
        offset: u64,
        /// Dimensions.
        shape: Vec<u64>,
        /// Strides, in elements.
        stride: Vec<u64>,
    },
    /// A stack mark.
    Mark,
}

impl Val {
    fn weight(&self) -> usize {
        match self {
            Val::List(v) | Val::Tuple(v) => 1 + v.iter().map(Val::weight).sum::<usize>(),
            Val::Dict(v) => {
                1 + v
                    .iter()
                    .map(|(k, x)| k.weight() + x.weight())
                    .sum::<usize>()
            }
            Val::Str(s) => 1 + s.len() / 64,
            Val::Tensor { shape, stride, .. } => 1 + shape.len() + stride.len(),
            _ => 1,
        }
    }

    /// The string, if this is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Val::Str(s) => Some(s),
            _ => None,
        }
    }

    /// The entry `key` of a dictionary.
    pub fn get(&self, key: &str) -> Option<&Val> {
        match self {
            Val::Dict(pairs) => pairs
                .iter()
                .find(|(k, _)| k.as_str() == Some(key))
                .map(|(_, v)| v),
            _ => None,
        }
    }
}

fn fail<T>(offset: usize, what: impl Into<String>) -> Result<T, String> {
    Err(format!("at byte {offset}: {}", what.into()))
}

struct Machine<'a> {
    data: &'a [u8],
    at: usize,
    stack: Vec<Val>,
    memo: HashMap<u32, Val>,
    nodes: usize,
}

impl<'a> Machine<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let start = self.at;
        let end = start.checked_add(n);
        match end.and_then(|e| self.data.get(start..e)) {
            Some(s) => {
                self.at += n;
                Ok(s)
            }
            None => fail(
                start,
                format!("the pickle ends inside an opcode (needs {n} bytes)"),
            ),
        }
    }

    fn byte(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?.first().copied().unwrap_or(0))
    }

    fn le_u32(&mut self) -> Result<u32, String> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([
            b.first().copied().unwrap_or(0),
            b.get(1).copied().unwrap_or(0),
            b.get(2).copied().unwrap_or(0),
            b.get(3).copied().unwrap_or(0),
        ]))
    }

    fn push(&mut self, v: Val, op_at: usize) -> Result<(), String> {
        if self.stack.len() >= MAX_STACK {
            return fail(op_at, format!("the stack is deeper than {MAX_STACK}"));
        }
        self.nodes = self.nodes.saturating_add(v.weight());
        if self.nodes > MAX_NODES {
            return fail(op_at, "the pickle builds too much data");
        }
        self.stack.push(v);
        Ok(())
    }

    fn pop(&mut self, op_at: usize) -> Result<Val, String> {
        self.stack
            .pop()
            .map_or_else(|| fail(op_at, "the stack is empty"), Ok)
    }

    /// The items above the nearest mark (removing the mark), in order.
    fn pop_to_mark(&mut self, op_at: usize) -> Result<Vec<Val>, String> {
        let Some(pos) = self.stack.iter().rposition(|v| matches!(v, Val::Mark)) else {
            return fail(op_at, "there is no mark on the stack");
        };
        let items = self.stack.split_off(pos + 1);
        self.stack.pop();
        Ok(items)
    }

    fn string(&mut self, n: usize, op_at: usize) -> Result<Val, String> {
        if n > MAX_STRING_BYTES {
            return fail(op_at, format!("a string of {n} bytes is over the limit"));
        }
        let bytes = self.take(n)?;
        match std::str::from_utf8(bytes) {
            Ok(s) => Ok(Val::Str(s.to_string())),
            Err(_) => fail(op_at, "a string is not valid UTF-8"),
        }
    }

    fn put(&mut self, idx: u32, op_at: usize) -> Result<(), String> {
        let Some(top) = self.stack.last().cloned() else {
            return fail(op_at, "the stack is empty");
        };
        if self.memo.len() >= MAX_MEMO && !self.memo.contains_key(&idx) {
            return fail(op_at, format!("more than {MAX_MEMO} memo entries"));
        }
        self.nodes = self.nodes.saturating_add(top.weight());
        if self.nodes > MAX_NODES {
            return fail(op_at, "the pickle builds too much data");
        }
        self.memo.insert(idx, top);
        Ok(())
    }

    fn get(&mut self, idx: u32, op_at: usize) -> Result<(), String> {
        let Some(v) = self.memo.get(&idx).cloned() else {
            return fail(op_at, format!("memo entry {idx} was never stored"));
        };
        self.push(v, op_at)
    }
}

fn global_of(module: &str, name: &str) -> Option<Global> {
    match (module, name) {
        ("collections", "OrderedDict") => Some(Global::OrderedDict),
        ("torch._utils", "_rebuild_tensor_v2") => Some(Global::RebuildTensorV2),
        ("torch", "FloatStorage") => Some(Global::FloatStorage),
        _ => None,
    }
}

fn depth(v: &Val) -> usize {
    match v {
        Val::List(x) | Val::Tuple(x) => 1 + x.iter().map(depth).max().unwrap_or(0),
        Val::Dict(x) => {
            1 + x
                .iter()
                .map(|(k, v)| depth(k).max(depth(v)))
                .max()
                .unwrap_or(0)
        }
        _ => 0,
    }
}

fn to_u64(v: &Val) -> Option<u64> {
    match v {
        Val::Int(n) => u64::try_from(*n).ok(),
        _ => None,
    }
}

fn dims(v: &Val) -> Option<Vec<u64>> {
    match v {
        Val::Tuple(items) | Val::List(items) => items.iter().map(to_u64).collect(),
        _ => None,
    }
}

fn persistent_id(pid: Val, at: usize) -> Result<Val, String> {
    // ('storage', torch.FloatStorage, key, location, numel)
    if let Val::Tuple(items) = &pid
        && let [
            Val::Str(kind),
            Val::Global(Global::FloatStorage),
            Val::Str(key),
            Val::Str(_),
            numel,
        ] = items.as_slice()
        && kind == "storage"
        && let Some(numel) = to_u64(numel)
    {
        return Ok(Val::Storage {
            key: key.clone(),
            numel,
        });
    }
    fail(
        at,
        "a persistent id is not ('storage', torch.FloatStorage, key, location, numel)",
    )
}

fn rebuild_tensor(args: Vec<Val>, at: usize) -> Result<Val, String> {
    // (storage, storage_offset, size, stride, requires_grad, backward_hooks[, metadata])
    let ok_tail = match args.get(6) {
        None | Some(Val::None) => args.len() <= 7,
        _ => false,
    };
    if let (
        Some(Val::Storage { key, numel }),
        Some(offset),
        Some(size),
        Some(stride),
        Some(Val::Bool(_)),
        Some(Val::Dict(hooks)),
    ) = (
        args.first(),
        args.get(1).and_then(to_u64),
        args.get(2).and_then(dims),
        args.get(3).and_then(dims),
        args.get(4),
        args.get(5),
    ) && ok_tail
        && hooks.is_empty()
    {
        return Ok(Val::Tensor {
            key: key.clone(),
            numel: *numel,
            offset,
            shape: size,
            stride,
        });
    }
    fail(
        at,
        "_rebuild_tensor_v2 was called with arguments of an unexpected shape",
    )
}

/// Interpret `data` and return the value the pickle produces.
///
/// # Errors
/// A description (with the byte offset) of the first thing that is not on the allow-list or is
/// malformed; nothing is ever executed.
pub fn interpret(data: &[u8]) -> Result<Val, String> {
    if data.len() > MAX_PICKLE_BYTES {
        return Err(format!(
            "the pickle is {} bytes; the maximum is {MAX_PICKLE_BYTES}",
            data.len()
        ));
    }
    let mut m = Machine {
        data,
        at: 0,
        stack: Vec::new(),
        memo: HashMap::new(),
        nodes: 0,
    };
    let mut first = true;
    loop {
        let op_at = m.at;
        let op = m.byte()?;
        if first && op != 0x80 {
            return fail(op_at, "the pickle does not start with a PROTO opcode");
        }
        first = false;
        match op {
            0x80 => {
                // PROTO
                let version = m.byte()?;
                if version != 2 {
                    return fail(
                        op_at,
                        format!("pickle protocol {version} is not accepted (only 2)"),
                    );
                }
            }
            b'.' => {
                // STOP
                let result = m.pop(op_at)?;
                if m.at != data.len() {
                    return fail(m.at, "there are bytes after the STOP opcode");
                }
                if !m.stack.is_empty() {
                    return fail(op_at, "the stack is not empty at STOP");
                }
                if depth(&result) > MAX_DEPTH {
                    return fail(
                        op_at,
                        format!("the result is nested deeper than {MAX_DEPTH}"),
                    );
                }
                return Ok(result);
            }
            b'(' => m.push(Val::Mark, op_at)?,
            b'}' => m.push(Val::Dict(Vec::new()), op_at)?,
            b']' => m.push(Val::List(Vec::new()), op_at)?,
            b')' => m.push(Val::Tuple(Vec::new()), op_at)?,
            b'N' => m.push(Val::None, op_at)?,
            0x88 => m.push(Val::Bool(true), op_at)?,
            0x89 => m.push(Val::Bool(false), op_at)?,
            b'J' => {
                // BININT
                let v = i64::from(m.le_u32()? as i32);
                m.push(Val::Int(v), op_at)?;
            }
            b'K' => {
                let v = i64::from(m.byte()?);
                m.push(Val::Int(v), op_at)?;
            }
            b'M' => {
                let b = m.take(2)?;
                let v = i64::from(u16::from_le_bytes([
                    b.first().copied().unwrap_or(0),
                    b.get(1).copied().unwrap_or(0),
                ]));
                m.push(Val::Int(v), op_at)?;
            }
            0x8a => {
                // LONG1: up to 8 bytes, little endian, two's complement
                let n = usize::from(m.byte()?);
                if n > 8 {
                    return fail(op_at, "an integer longer than 8 bytes");
                }
                let b = m.take(n)?;
                let mut v: i128 = 0;
                for (i, x) in b.iter().enumerate() {
                    v |= i128::from(*x) << (8 * i);
                }
                if n > 0 && b.last().is_some_and(|x| x & 0x80 != 0) {
                    v -= 1i128 << (8 * n);
                }
                let v =
                    i64::try_from(v).map_or_else(|_| fail(op_at, "an integer out of range"), Ok)?;
                m.push(Val::Int(v), op_at)?;
            }
            b'G' => {
                // BINFLOAT, big endian
                let b = m.take(8)?;
                let mut arr = [0u8; 8];
                for (i, x) in b.iter().enumerate() {
                    if let Some(slot) = arr.get_mut(i) {
                        *slot = *x;
                    }
                }
                m.push(Val::Float(f64::from_be_bytes(arr)), op_at)?;
            }
            b'X' => {
                let n = m.le_u32()? as usize;
                let v = m.string(n, op_at)?;
                m.push(v, op_at)?;
            }
            0x8c => {
                // SHORT_BINUNICODE
                let n = usize::from(m.byte()?);
                let v = m.string(n, op_at)?;
                m.push(v, op_at)?;
            }
            b'q' => {
                let idx = u32::from(m.byte()?);
                m.put(idx, op_at)?;
            }
            b'r' => {
                let idx = m.le_u32()?;
                m.put(idx, op_at)?;
            }
            b'h' => {
                let idx = u32::from(m.byte()?);
                m.get(idx, op_at)?;
            }
            b'j' => {
                let idx = m.le_u32()?;
                m.get(idx, op_at)?;
            }
            b't' => {
                let items = m.pop_to_mark(op_at)?;
                m.push(Val::Tuple(items), op_at)?;
            }
            0x85..=0x87 => {
                // TUPLE1, TUPLE2, TUPLE3
                let n = usize::from(op - 0x84);
                if m.stack.len() < n {
                    return fail(op_at, "the stack has too few items for a tuple");
                }
                let items = m.stack.split_off(m.stack.len() - n);
                m.push(Val::Tuple(items), op_at)?;
            }
            b'a' | b'e' => {
                // APPEND, APPENDS
                let items = if op == b'e' {
                    m.pop_to_mark(op_at)?
                } else {
                    vec![m.pop(op_at)?]
                };
                match m.stack.last_mut() {
                    Some(Val::List(list)) => list.extend(items),
                    _ => return fail(op_at, "APPEND(S) needs a list below"),
                }
            }
            b's' | b'u' => {
                // SETITEM, SETITEMS
                let items = if op == b'u' {
                    m.pop_to_mark(op_at)?
                } else {
                    let value = m.pop(op_at)?;
                    let key = m.pop(op_at)?;
                    vec![key, value]
                };
                if items.len() % 2 != 0 {
                    return fail(op_at, "SETITEMS has an odd number of items");
                }
                let Some(Val::Dict(dict)) = m.stack.last_mut() else {
                    return fail(op_at, "SETITEM(S) needs a dictionary below");
                };
                let mut it = items.into_iter();
                while let (Some(k), Some(v)) = (it.next(), it.next()) {
                    if !matches!(k, Val::Str(_) | Val::Int(_)) {
                        return fail(op_at, "a dictionary key is not a string or an integer");
                    }
                    match dict.iter_mut().find(|(ek, _)| *ek == k) {
                        Some(slot) => slot.1 = v,
                        None => dict.push((k, v)),
                    }
                }
            }
            b'c' => {
                // GLOBAL: "module\nname\n"
                let start = m.at;
                let rest = data.get(start..).unwrap_or(&[]);
                let end = rest
                    .iter()
                    .enumerate()
                    .filter(|(_, b)| **b == b'\n')
                    .map(|(i, _)| i)
                    .nth(1);
                let Some(end) = end else {
                    return fail(op_at, "a GLOBAL without its two lines");
                };
                if end > 256 {
                    return fail(op_at, "a GLOBAL name is too long");
                }
                let text = std::str::from_utf8(rest.get(..end).unwrap_or(&[]))
                    .map_err(|_| format!("at byte {op_at}: a GLOBAL name is not UTF-8"))?;
                m.at = start + end + 1;
                let mut parts = text.splitn(2, '\n');
                let (module, name) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
                match global_of(module, name) {
                    Some(g) => m.push(Val::Global(g), op_at)?,
                    None => {
                        return fail(
                            op_at,
                            format!("the global {module}.{name} is not on the allow-list"),
                        );
                    }
                }
            }
            b'R' => {
                // REDUCE: only the two callables the allow-list knows
                let args = m.pop(op_at)?;
                let callable = m.pop(op_at)?;
                let Val::Tuple(args) = args else {
                    return fail(op_at, "REDUCE needs a tuple of arguments");
                };
                let built = match callable {
                    Val::Global(Global::OrderedDict) if args.is_empty() => Val::Dict(Vec::new()),
                    Val::Global(Global::RebuildTensorV2) => rebuild_tensor(args, op_at)?,
                    _ => return fail(op_at, "REDUCE on something that is not an allowed call"),
                };
                m.push(built, op_at)?;
            }
            b'Q' => {
                // BINPERSID
                let pid = m.pop(op_at)?;
                let v = persistent_id(pid, op_at)?;
                m.push(v, op_at)?;
            }
            b'b' => {
                // BUILD: the state of an OrderedDict (torch writes its `_metadata` this way)
                let state = m.pop(op_at)?;
                if !matches!(state, Val::Dict(_)) {
                    return fail(op_at, "BUILD with a state that is not a dictionary");
                }
                if !matches!(m.stack.last(), Some(Val::Dict(_))) {
                    return fail(op_at, "BUILD on something that is not a dictionary");
                }
            }
            other => {
                return fail(
                    op_at,
                    format!("the opcode {other:#04x} is not on the allow-list"),
                );
            }
        }
    }
}
