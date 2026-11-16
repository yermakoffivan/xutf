//! [`UnitSet`]: SIMD membership scans over small sets of code-unit values.
//!
//! A set holds up to [`UnitSet::MAX_RANGES`] disjoint inclusive ranges of
//! values in `0..=0xFF`. Each 64-unit block is classified with one wrapping
//! `(v - lo) <= (hi - lo)` compare per range on byte lanes (UTF-16/32 units
//! saturate to `0xFF` first, so the classifier always runs on four NEON / two
//! AVX2 registers), packed into a `u64`, and walked with `trailing_zeros` /
//! `leading_zeros`. [`Fields`] keeps the current block's mask and finds field
//! boundaries from its bit transitions instead of restarting a search per
//! field.

use core::{
	iter::FusedIterator,
	ops::RangeInclusive,
	simd::{
		Mask, Simd,
		cmp::{SimdOrd, SimdPartialOrd},
		num::SimdUint,
	},
};

use crate::{
	encoding::{Encoding, Kind},
	simd::{BLOCK, low_bits, window},
	unit::{Unit, as_units},
	utf8::Utf8,
};

const MAX: usize = 8;

/// Inputs shorter than this are classified with a scalar bitmap probe per
/// unit: below one 16-byte register the vector setup costs more than the
/// loop.
const SCALAR_BELOW: usize = 16;

/// Units a search tests one at a time before its first vector probe.
const PROBE: usize = 4;

/// Units [`UnitSet::trim`] tests one at a time from each end before
/// switching to blocks: edges rarely hold more than a few members.
const TRIM_SCALAR: usize = 8;

/// A set of code-unit values in `0..=0xFF`, scanned with SIMD.
///
/// Use it for the byte-class loops of tokenizers and parsers: skipping
/// whitespace, finding the next delimiter, trimming, splitting on a separator
/// class. Values above `0xFF` are never members, so any UTF-16/UTF-32 unit
/// outside Latin-1 is a non-member. UTF-8 input is matched as raw bytes: a
/// member `>= 0x80` matches lead or continuation bytes with that value.
/// Malformed input is scanned as-is and never panics.
///
/// # Example
/// ```
/// use xutf::{UnitSet, Utf8, Utf16};
///
/// const DELIMS: UnitSet = UnitSet::new(b",;:");
/// assert_eq!(DELIMS.find::<Utf8>(b"key: value"), Some(3));
///
/// let wide: Vec<u16> = "a\u{3000}b;c".encode_utf16().collect();
/// assert_eq!(DELIMS.find::<Utf16>(&wide), Some(3));
///
/// let ws = UnitSet::ASCII_WHITESPACE;
/// assert_eq!(ws.trim_str("\t hello world \n"), "hello world");
/// let words: Vec<_> = ws.fields_str("  ab  cd e").map(|(_, w)| w).collect();
/// assert_eq!(words, ["ab", "cd", "e"]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitSet {
	/// Membership bitmap of `0..=0xFF`, for scalar probes.
	map:  [u64; 4],
	/// Range starts, ascending; unused slots are zero.
	lo:   [u8; MAX],
	/// Range widths (`hi - lo`); unused slots are zero.
	span: [u8; MAX],
	/// Number of live ranges.
	len:  u8,
}

impl UnitSet {
	/// C `isspace` / Python `bytes.isspace` set: `\t \n \v \f \r` and space
	/// (note: includes `\v`, unlike [`u8::is_ascii_whitespace`]).
	pub const ASCII_WHITESPACE: Self = Self::new(b"\t\n\x0b\x0c\r ");
	/// Most disjoint ranges a set can hold.
	pub const MAX_RANGES: usize = MAX;

	/// Builds the set of `members`; adjacent values merge into ranges.
	///
	/// # Panics
	/// If the members form more than [`Self::MAX_RANGES`] ranges (a compile
	/// error in const context).
	///
	/// # Example
	/// ```
	/// use xutf::UnitSet;
	///
	/// const QUOTES: UnitSet = UnitSet::new(b"\"'`");
	/// assert!(QUOTES.contains(u32::from(b'`')));
	/// assert!(!QUOTES.contains(u32::from(b'a')));
	/// ```
	pub const fn new(members: &[u8]) -> Self {
		let mut map = [0u64; 4];
		let mut i = 0;
		while i < members.len() {
			let b = members[i] as usize;
			map[b >> 6] |= 1 << (b & 63);
			i += 1;
		}
		Self::from_bitmap(map)
	}

	/// Builds the set covering `ranges`; overlapping or adjacent ranges merge
	/// and empty ranges (`start > end`) are ignored.
	///
	/// # Panics
	/// If the merged ranges number more than [`Self::MAX_RANGES`] (a compile
	/// error in const context).
	///
	/// # Example
	/// ```
	/// use xutf::{UnitSet, Utf8};
	///
	/// const IDENT: UnitSet =
	/// 	UnitSet::from_ranges(&[b'0'..=b'9', b'A'..=b'Z', b'_'..=b'_', b'a'..=b'z']);
	/// assert_eq!(IDENT.find_not::<Utf8>(b"snake_case2 = 1"), Some(11));
	/// ```
	pub const fn from_ranges(ranges: &[RangeInclusive<u8>]) -> Self {
		let mut map = [0u64; 4];
		let mut i = 0;
		while i < ranges.len() {
			let mut b = *ranges[i].start() as usize;
			let end = *ranges[i].end() as usize;
			while b <= end {
				map[b >> 6] |= 1 << (b & 63);
				b += 1;
			}
			i += 1;
		}
		Self::from_bitmap(map)
	}

	const fn from_bitmap(map: [u64; 4]) -> Self {
		let mut set = Self { map, lo: [0; MAX], span: [0; MAX], len: 0 };
		let mut b = 0;
		while b < 256 {
			if (map[b >> 6] >> (b & 63)) & 1 == 0 {
				b += 1;
				continue;
			}
			let start = b;
			while b < 256 && (map[b >> 6] >> (b & 63)) & 1 != 0 {
				b += 1;
			}
			assert!((set.len as usize) < MAX, "UnitSet holds at most MAX_RANGES (8) disjoint ranges");
			set.lo[set.len as usize] = start as u8;
			set.span[set.len as usize] = (b - 1 - start) as u8;
			set.len += 1;
		}
		set
	}

	/// Whether the native-order unit value `unit` is a member.
	pub const fn contains(&self, unit: u32) -> bool {
		unit < 256 && (self.map[(unit >> 6) as usize] >> (unit & 63)) & 1 != 0
	}

	/// Whether every member is below `0x80`, so splitting UTF-8 at members
	/// keeps char boundaries (required by the `_str` methods).
	pub const fn is_ascii(&self) -> bool {
		self.map[2] == 0 && self.map[3] == 0
	}

	/// Index of the first member unit in `input`.
	///
	/// # Example
	/// ```
	/// use xutf::{UnitSet, Utf8};
	///
	/// const JSON_ESCAPE: UnitSet = UnitSet::from_ranges(&[0..=0x1f, b'"'..=b'"', b'\\'..=b'\\']);
	/// assert_eq!(JSON_ESCAPE.find::<Utf8>(b"plain text \"quoted\""), Some(11));
	/// assert_eq!(JSON_ESCAPE.find::<Utf8>(b"plain text"), None);
	/// ```
	pub fn find<E: Encoding>(&self, input: &[E::Unit]) -> Option<usize> {
		self.first::<E, false>(input)
	}

	/// Index of the first non-member unit in `input`; the leading member span
	/// is `find_not(input).unwrap_or(input.len())` units long.
	pub fn find_not<E: Encoding>(&self, input: &[E::Unit]) -> Option<usize> {
		self.first::<E, true>(input)
	}

	/// Index of the last member unit in `input`.
	pub fn rfind<E: Encoding>(&self, input: &[E::Unit]) -> Option<usize> {
		self.last::<E, false>(input)
	}

	/// Index of the last non-member unit in `input`.
	pub fn rfind_not<E: Encoding>(&self, input: &[E::Unit]) -> Option<usize> {
		self.last::<E, true>(input)
	}

	/// Number of member units in `input`.
	///
	/// # Example
	/// ```
	/// use xutf::{UnitSet, Utf16};
	///
	/// let text: Vec<u16> = "a b\tc\n".encode_utf16().collect();
	/// assert_eq!(UnitSet::ASCII_WHITESPACE.count::<Utf16>(&text), 3);
	/// ```
	pub fn count<E: Encoding>(&self, input: &[E::Unit]) -> usize {
		let n = input.len();
		if n < SCALAR_BELOW {
			return input.iter().filter(|&&u| self.has::<E>(u)).count();
		}
		(0..n)
			.step_by(BLOCK)
			.map(|at| self.bits::<E>(input, at).count_ones() as usize)
			.sum()
	}

	/// `input` without leading and trailing members.
	///
	/// # Example
	/// ```
	/// use xutf::{UnitSet, Utf8};
	///
	/// assert_eq!(UnitSet::ASCII_WHITESPACE.trim::<Utf8>(b" \x0bkey\r\n"), b"key");
	/// ```
	pub fn trim<'a, E: Encoding>(&self, input: &'a [E::Unit]) -> &'a [E::Unit] {
		self.trim_end::<E>(self.trim_start::<E>(input))
	}

	/// `input` without leading members.
	pub fn trim_start<'a, E: Encoding>(&self, input: &'a [E::Unit]) -> &'a [E::Unit] {
		&input[self.leading::<E>(input)..]
	}

	/// `input` without trailing members.
	pub fn trim_end<'a, E: Encoding>(&self, input: &'a [E::Unit]) -> &'a [E::Unit] {
		&input[..input.len() - self.trailing::<E>(input)]
	}

	/// Maximal runs of non-members with their unit offsets, like awk's default
	/// field splitting or [`str::split_whitespace`] with this set as the
	/// separator class.
	///
	/// # Example
	/// ```
	/// use xutf::{UnitSet, Utf8};
	///
	/// let request = b" GET  /index HTTP/1.1\r\n";
	/// let fields: Vec<_> = UnitSet::ASCII_WHITESPACE.fields::<Utf8>(request).collect();
	/// assert_eq!(fields, [(1, &b"GET"[..]), (6, &b"/index"[..]), (13, &b"HTTP/1.1"[..])]);
	/// ```
	pub const fn fields<'a, E: Encoding>(&self, input: &'a [E::Unit]) -> Fields<'a, E> {
		Fields { set: *self, input, next: 0, base: 0, starts: 0, ends: 0, carry: 0 }
	}

	/// [`UnitSet::trim`] over UTF-8 `str`.
	///
	/// # Panics
	/// Unless [`UnitSet::is_ascii`], since trimming a non-ASCII member could
	/// split a char.
	pub fn trim_str<'a>(&self, input: &'a str) -> &'a str {
		assert!(self.is_ascii(), "UnitSet::trim_str needs an ASCII-only set");
		let bytes = input.as_bytes();
		let start = self.leading::<Utf8>(bytes);
		let end = bytes.len() - self.trailing::<Utf8>(&bytes[start..]);
		// SAFETY: members are ASCII, so every non-member run starts and ends at
		// a char boundary (an ASCII byte, or the ends of `input`).
		unsafe { input.get_unchecked(start..end) }
	}

	/// [`UnitSet::fields`] over UTF-8 `str`, yielding byte offsets.
	///
	/// # Panics
	/// Unless [`UnitSet::is_ascii`], since splitting at a non-ASCII member
	/// could split a char.
	///
	/// # Example
	/// ```
	/// use xutf::UnitSet;
	///
	/// let words: Vec<_> = UnitSet::ASCII_WHITESPACE
	/// 	.fields_str("状态  café\tok")
	/// 	.collect();
	/// assert_eq!(words, [(0, "状态"), (8, "café"), (14, "ok")]);
	/// ```
	pub const fn fields_str<'a>(&self, input: &'a str) -> StrFields<'a> {
		assert!(self.is_ascii(), "UnitSet::fields_str needs an ASCII-only set");
		StrFields { input, inner: self.fields::<Utf8>(input.as_bytes()) }
	}

	/// Membership of a raw unit of encoding `E`.
	#[inline(always)]
	fn has<E: Encoding>(&self, unit: E::Unit) -> bool {
		let unit = if E::FOREIGN { unit.swap_bytes() } else { unit };
		self.contains(unit.to_u32())
	}

	/// Member bits of byte lanes.
	#[inline(always)]
	fn hits(&self, q: Simd<u8, 16>) -> Mask<i8, 16> {
		let mut hit = Mask::splat(false);
		for (&lo, &span) in self.lo.iter().zip(&self.span).take(self.len as usize) {
			hit |= (q - Simd::splat(lo)).simd_le(Simd::splat(span));
		}
		hit
	}

	/// Member lanes of 16 UTF-16/32 units: lanes saturate to bytes before the
	/// range compares, and lanes above `0xFF` are excluded separately only
	/// when `0xFF` is a member.
	#[inline(always)]
	fn hits_wide<U: Wide>(&self, q: Simd<U, 16>, foreign: bool) -> Mask<i8, 16> {
		let q = if foreign { U::swap(q) } else { q };
		let mut hit = self.hits(U::saturate(q));
		if self.contains(0xff) {
			hit &= !U::above_byte(q);
		}
		hit
	}

	/// Member bits of the block at `at`, bit `i` for unit `at + i`; bits at or
	/// past the end are clear.
	#[inline(always)]
	fn bits<E: Encoding>(&self, data: &[E::Unit], at: usize) -> u64 {
		macro_rules! classify {
			($unit:ty, $pred:expr) => {{
				// SAFETY: the sealed `Encoding` maps each `Kind` to one unit type.
				let data: &[$unit] = unsafe { as_units(data) };
				let (window, origin) = window(data, at);
				window.bits($pred) >> (at - origin)
			}};
		}
		match E::KIND {
			Kind::Utf8 => classify!(u8, |q| self.hits(q)),
			Kind::Utf16 => classify!(u16, |q| self.hits_wide(q, E::FOREIGN)),
			Kind::Utf32 => classify!(u32, |q| self.hits_wide(q, E::FOREIGN)),
		}
	}

	/// [`Self::bits`] for the members (`NOT = false`) or non-members.
	#[inline(always)]
	fn select<E: Encoding, const NOT: bool>(&self, data: &[E::Unit], at: usize) -> u64 {
		let bits = self.bits::<E>(data, at);
		if NOT {
			!bits & low_bits(data.len() - at)
		} else {
			bits
		}
	}

	fn first<E: Encoding, const NOT: bool>(&self, data: &[E::Unit]) -> Option<usize> {
		let n = data.len();
		if n < SCALAR_BELOW {
			return data.iter().position(|&u| self.has::<E>(u) != NOT);
		}
		// A scanner restarting after each hit usually finds the next one within
		// a few units: probe those one at a time, then 16 units, before a whole
		// block would pay off.
		if let Some(i) = data[..PROBE].iter().position(|&u| self.has::<E>(u) != NOT) {
			return Some(i);
		}
		let head = self.select::<E, NOT>(&data[..16], 0);
		if head != 0 {
			return Some(head.trailing_zeros() as usize);
		}
		(16..n).step_by(BLOCK).find_map(|at| {
			let bits = self.select::<E, NOT>(data, at);
			(bits != 0).then(|| at + bits.trailing_zeros() as usize)
		})
	}

	fn last<E: Encoding, const NOT: bool>(&self, data: &[E::Unit]) -> Option<usize> {
		let n = data.len();
		if n < SCALAR_BELOW {
			return data.iter().rposition(|&u| self.has::<E>(u) != NOT);
		}
		let tail = self.select::<E, NOT>(data, n - 16);
		if tail != 0 {
			return Some(n - 16 + top(tail));
		}
		let mut end = n - 16;
		while end > 0 {
			let at = end.saturating_sub(BLOCK);
			let bits = self.select::<E, NOT>(&data[..end], at);
			if bits != 0 {
				return Some(at + top(bits));
			}
			end = at;
		}
		None
	}

	/// Number of leading members: a few scalar probes, then blocks.
	#[inline]
	fn leading<E: Encoding>(&self, data: &[E::Unit]) -> usize {
		let head = data.len().min(TRIM_SCALAR);
		match data[..head].iter().position(|&u| !self.has::<E>(u)) {
			Some(i) => i,
			None if head == data.len() => head,
			None => {
				head
					+ self
						.first::<E, true>(&data[head..])
						.unwrap_or(data.len() - head)
			},
		}
	}

	/// Number of trailing members: a few scalar probes, then blocks.
	#[inline]
	fn trailing<E: Encoding>(&self, data: &[E::Unit]) -> usize {
		let body = data.len() - data.len().min(TRIM_SCALAR);
		match data[body..].iter().rposition(|&u| !self.has::<E>(u)) {
			Some(i) => data.len() - body - i - 1,
			None if body == 0 => data.len(),
			None => data.len() - self.last::<E, true>(&data[..body]).map_or(0, |i| i + 1),
		}
	}
}

/// 16-lane operations [`UnitSet`] needs on UTF-16/32 units.
trait Wide: Unit {
	fn swap(q: Simd<Self, 16>) -> Simd<Self, 16>;
	/// Lanes clamped to `0xFF` and narrowed to bytes.
	fn saturate(q: Simd<Self, 16>) -> Simd<u8, 16>;
	/// Lanes above `0xFF`.
	fn above_byte(q: Simd<Self, 16>) -> Mask<i8, 16>;
}

macro_rules! impl_wide {
	($unit:ty) => {
		impl Wide for $unit {
			#[inline(always)]
			fn swap(q: Simd<Self, 16>) -> Simd<Self, 16> {
				q.swap_bytes()
			}

			#[inline(always)]
			fn saturate(q: Simd<Self, 16>) -> Simd<u8, 16> {
				q.simd_min(Simd::splat(0xff)).cast()
			}

			#[inline(always)]
			fn above_byte(q: Simd<Self, 16>) -> Mask<i8, 16> {
				q.simd_gt(Simd::splat(0xff)).cast()
			}
		}
	};
}

impl_wide!(u16);
impl_wide!(u32);

/// Index of the highest set bit of a nonzero `bits`.
#[inline(always)]
const fn top(bits: u64) -> usize {
	63 - bits.leading_zeros() as usize
}

/// Iterator over the maximal non-member runs of a unit slice, from
/// [`UnitSet::fields`]; yields `(unit offset, field)`.
pub struct Fields<'a, E: Encoding> {
	set:    UnitSet,
	input:  &'a [E::Unit],
	/// Offset of the next block to classify.
	next:   usize,
	/// Offset of the block `starts` and `ends` cover.
	base:   usize,
	/// Unvisited field starts in that block.
	starts: u64,
	/// Unvisited field ends (one past a field's last unit) in that block.
	ends:   u64,
	/// Bit 0 set when the block's last unit is a field unit.
	carry:  u64,
}

impl<E: Encoding> Clone for Fields<'_, E> {
	fn clone(&self) -> Self {
		Self { ..*self }
	}
}

impl<E: Encoding> core::fmt::Debug for Fields<'_, E> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		f.debug_struct("Fields")
			.field("set", &self.set)
			.field("len", &self.input.len())
			.field("next", &self.next)
			.finish_non_exhaustive()
	}
}

impl<E: Encoding> Fields<'_, E> {
	/// Classifies the next block into `starts`/`ends`; `false` past the end.
	///
	/// Lanes past the end count as members, so a field reaching the end of a
	/// partial block ends there; one filling the last whole block has no end
	/// bit and ends with the input.
	#[inline(always)]
	fn advance(&mut self) -> bool {
		let (at, n) = (self.next, self.input.len());
		if at >= n {
			return false;
		}
		let field = !self.set.bits::<E>(self.input, at) & low_bits(n - at);
		let before = (field << 1) | self.carry;
		self.starts = field & !before;
		self.ends = !field & before;
		self.carry = field >> 63;
		self.base = at;
		self.next = at + BLOCK;
		true
	}
}

impl<'a, E: Encoding> Iterator for Fields<'a, E> {
	type Item = (usize, &'a [E::Unit]);

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		while self.starts == 0 {
			if !self.advance() {
				return None;
			}
		}
		let start = self.base + self.starts.trailing_zeros() as usize;
		self.starts &= self.starts - 1;
		// Starts and ends alternate, so the lowest unvisited end closes this
		// field; until a block has one, the field runs on.
		while self.ends == 0 {
			if !self.advance() {
				return Some((start, &self.input[start..]));
			}
		}
		let end = self.base + self.ends.trailing_zeros() as usize;
		self.ends &= self.ends - 1;
		Some((start, &self.input[start..end]))
	}
}

impl<E: Encoding> FusedIterator for Fields<'_, E> {}

/// Iterator over the maximal non-member runs of a `str`, from
/// [`UnitSet::fields_str`]; yields `(byte offset, field)`.
#[derive(Clone, Debug)]
pub struct StrFields<'a> {
	input: &'a str,
	inner: Fields<'a, Utf8>,
}

impl<'a> Iterator for StrFields<'a> {
	type Item = (usize, &'a str);

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		let (start, field) = self.inner.next()?;
		// SAFETY: `fields_str` checked the set is ASCII, so fields start and end
		// at ASCII bytes or the ends of `input`: both char boundaries.
		Some((start, unsafe { self.input.get_unchecked(start..start + field.len()) }))
	}
}

impl FusedIterator for StrFields<'_> {}
