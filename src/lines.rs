//! Line splitting over UTF-8, UTF-16, and UTF-32, 64 code units per step.
//!
//! Each 64-unit block is classified into a `u64` of line-end candidates once;
//! the iterator then pops one bit per line, so a short line costs a
//! `trailing_zeros` and a few scalar compares rather than a fresh vector
//! search, and a long line costs one block test per 64 units. A final partial
//! block re-reads the last whole block; input shorter than a block is read 16
//! units at a time (see [`crate::simd::window`]). Counting UTF-8 lines skips
//! the bitmasks and sums compare results in byte lanes.
//!
//! Which units end a line is a [`LineEnds`] choice: `\n` alone (with an
//! optional `\r` before it), universal newlines, or Python's full Unicode
//! set. Lines never include their terminator in [`Line::text`]; a trailing
//! terminator does not start an extra empty line.

use core::{
	fmt,
	iter::FusedIterator,
	marker::PhantomData,
	ops::{Index, RangeFrom, RangeTo},
	simd::{
		Simd,
		cmp::{SimdPartialEq, SimdPartialOrd},
		num::{SimdInt, SimdUint},
	},
};

use crate::{
	encoding::{Encoding, Kind},
	simd::{BLOCK, Window, window},
	unit::{Unit, as_units},
	utf8::Utf8,
};

const LF: u32 = 0x0a;
const CR: u32 = 0x0d;

/// The code points that end a line.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LineEnds {
	/// LF, with a CR directly before it joining the terminator; a lone CR is
	/// text. Matches [`str::lines`], `BufRead::lines`, and Git.
	Lf,
	/// LF, CRLF, and lone CR. Matches Python's `bytes.splitlines`, universal
	/// newlines, and terminal hard breaks.
	Universal,
	/// [`Universal`](LineEnds::Universal) plus VT, FF, FS, GS, RS, NEL
	/// (U+0085), LS (U+2028), and PS (U+2029). Matches Python's
	/// `str.splitlines`.
	Unicode,
}

/// One line from [`lines`] or [`lines_str`]: its text, its terminator, and
/// where it starts.
///
/// `T` is `str` for [`lines_str`] and the code-unit slice type for
/// [`lines`].
#[derive(Debug, PartialEq, Eq)]
pub struct Line<'a, T: ?Sized> {
	/// Text followed by its terminator.
	full:     &'a T,
	/// Units of `full` before the terminator.
	text_len: usize,
	/// Offset of `full` in the input.
	at:       usize,
}

impl<T: ?Sized> Clone for Line<'_, T> {
	#[inline(always)]
	fn clone(&self) -> Self {
		*self
	}
}

impl<T: ?Sized> Copy for Line<'_, T> {}

impl<'a, T> Line<'a, T>
where
	T: ?Sized + Index<RangeTo<usize>, Output = T> + Index<RangeFrom<usize>, Output = T>,
{
	/// The line without its terminator.
	#[inline]
	pub fn text(&self) -> &'a T {
		&self.full[..self.text_len]
	}

	/// The terminator (`"\n"`, `"\r\n"`, ...): empty only for a final line the
	/// input ends without terminating, which a stream reader should hold
	/// until more input arrives.
	#[inline]
	pub fn end(&self) -> &'a T {
		&self.full[self.text_len..]
	}

	/// The line with its terminator, as Python's `splitlines(keepends=True)`
	/// returns it.
	#[inline]
	pub const fn with_end(&self) -> &'a T {
		self.full
	}

	/// Offset of the line's first unit in the input: bytes for
	/// [`lines_str`], code units for [`lines`].
	#[inline]
	pub const fn at(&self) -> usize {
		self.at
	}
}

/// Iterator over the lines of encoded text, created by [`lines`].
pub struct Lines<'a, E: Encoding> {
	input:     &'a [E::Unit],
	ends:      LineEnds,
	/// Offset where the next line starts.
	start:     usize,
	/// Offset of the block `bits` was classified from.
	base:      usize,
	/// Line-end candidates in that block not yet visited, bit `i` for unit
	/// `base + i`.
	bits:      u64,
	_encoding: PhantomData<E>,
}

impl<E: Encoding> Clone for Lines<'_, E> {
	#[inline]
	fn clone(&self) -> Self {
		Self { ..*self }
	}
}

impl<E: Encoding> fmt::Debug for Lines<'_, E> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("Lines")
			.field("ends", &self.ends)
			.field("start", &self.start)
			.field("len", &self.input.len())
			.finish_non_exhaustive()
	}
}

/// Splits encoded text into lines ending at `ends`.
///
/// Offsets are code units. Decoding is permissive: units inside malformed
/// sequences that equal a line end still end a line. Use [`lines_str`] for
/// `&str` lines.
///
/// # Example
/// ```
/// use xutf::{LineEnds, Utf16, lines};
///
/// let text: Vec<u16> = "one\r\ntwo\rthree".encode_utf16().collect();
/// let mut it = lines::<Utf16>(&text, LineEnds::Universal);
/// let first = it.next().unwrap();
/// assert_eq!(String::from_utf16_lossy(first.text()), "one");
/// assert_eq!(first.end(), [b'\r' as u16, b'\n' as u16]);
/// assert_eq!(it.map(|line| line.at()).collect::<Vec<_>>(), [5, 9]);
/// ```
#[inline]
pub fn lines<E: Encoding>(input: &[E::Unit], ends: LineEnds) -> Lines<'_, E> {
	let bits = if input.is_empty() {
		0
	} else {
		classify::<E>(input, 0, ends)
	};
	Lines { input, ends, start: 0, base: 0, bits, _encoding: PhantomData }
}

impl<'a, E: Encoding> Lines<'a, E> {
	/// Classifies the next block into `bits`, or returns `false` past the end.
	#[inline(always)]
	fn advance(&mut self) -> bool {
		let next = self.base + BLOCK;
		if next >= self.input.len() {
			return false;
		}
		self.base = next;
		self.bits = classify::<E>(self.input, next, self.ends);
		true
	}

	/// Resolves the candidate at `at` to `(text_end, next_start)`, or `None`
	/// for a CR that the LF after it terminates as CRLF.
	#[inline(always)]
	fn resolve(&self, at: usize) -> Option<(usize, usize)> {
		let unit = value::<E>(self.input[at]);
		let text_end = match unit {
			LF if at > self.start && value::<E>(self.input[at - 1]) == CR => at - 1,
			CR if self
				.input
				.get(at + 1)
				.is_some_and(|&next| value::<E>(next) == LF) =>
			{
				return None;
			},
			// Multi-byte UTF-8 ends are marked on their last byte.
			0x85 if E::KIND == Kind::Utf8 => at - 1,
			0xa8 | 0xa9 if E::KIND == Kind::Utf8 => at - 2,
			_ => at,
		};
		Some((text_end, at + 1))
	}

	/// The unterminated final line, if any input remains.
	#[inline]
	fn finish(&mut self) -> Option<Line<'a, [E::Unit]>> {
		let len = self.input.len();
		if self.start == len {
			return None;
		}
		let line = Line {
			full:     &self.input[self.start..],
			text_len: len - self.start,
			at:       self.start,
		};
		self.start = len;
		Some(line)
	}
}

impl<'a, E: Encoding> Iterator for Lines<'a, E> {
	type Item = Line<'a, [E::Unit]>;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		loop {
			while self.bits == 0 {
				if !self.advance() {
					return self.finish();
				}
			}
			let at = self.base + self.bits.trailing_zeros() as usize;
			self.bits &= self.bits - 1;
			if let Some((text_end, next)) = self.resolve(at) {
				let line = Line {
					full:     &self.input[self.start..next],
					text_len: text_end - self.start,
					at:       self.start,
				};
				self.start = next;
				return Some(line);
			}
		}
	}

	#[inline]
	fn size_hint(&self) -> (usize, Option<usize>) {
		let rest = self.input.len() - self.start;
		(usize::from(rest != 0), Some(rest))
	}

	/// Counts lines without building them: UTF-8 LF and universal line ends
	/// are summed in byte lanes; elsewhere under [`LineEnds::Lf`] every
	/// candidate is a terminator, so whole blocks count by popcount.
	fn count(mut self) -> usize {
		if E::KIND == Kind::Utf8 && self.ends != LineEnds::Unicode {
			// SAFETY: Utf8 is the only sealed encoding with Kind::Utf8.
			let rest: &[u8] = unsafe { as_units(&self.input[self.start..]) };
			let universal = self.ends == LineEnds::Universal;
			let terminated = rest
				.last()
				.is_none_or(|&last| last == LF as u8 || (universal && last == CR as u8));
			return count_ends_u8(rest, universal) + usize::from(!terminated);
		}
		let mut count = 0;
		// End of the last terminator seen; a final unterminated line follows
		// it unless it reaches the end of the input.
		let mut last = self.start;
		loop {
			if self.bits != 0 {
				if self.ends == LineEnds::Lf {
					count += self.bits.count_ones() as usize;
					last = self.base + (BLOCK - self.bits.leading_zeros() as usize);
				} else {
					while self.bits != 0 {
						let at = self.base + self.bits.trailing_zeros() as usize;
						self.bits &= self.bits - 1;
						if let Some((_, next)) = self.resolve(at) {
							count += 1;
							last = next;
						}
					}
				}
			}
			if !self.advance() {
				return count + usize::from(last < self.input.len());
			}
		}
	}
}

impl<E: Encoding> FusedIterator for Lines<'_, E> {}

/// Iterator over the lines of a `&str`, created by [`lines_str`].
#[derive(Clone, Debug)]
pub struct StrLines<'a>(Lines<'a, Utf8>);

/// Splits a string into lines ending at `ends`; offsets are bytes.
///
/// # Example
/// ```
/// use xutf::{LineEnds, lines_str};
///
/// let text = "fn main() {\r\n\tprintln!();\n}";
/// let lines: Vec<&str> = lines_str(text, LineEnds::Lf)
/// 	.map(|line| line.text())
/// 	.collect();
/// assert_eq!(lines, ["fn main() {", "\tprintln!();", "}"]);
///
/// // Python's `str.splitlines(keepends=True)`.
/// let kept: Vec<&str> = lines_str("a\u{2028}b\x0cc", LineEnds::Unicode)
/// 	.map(|line| line.with_end())
/// 	.collect();
/// assert_eq!(kept, ["a\u{2028}", "b\x0c", "c"]);
/// assert_eq!(lines_str("a\nb\n", LineEnds::Lf).count(), 2);
/// ```
#[inline]
pub fn lines_str(input: &str, ends: LineEnds) -> StrLines<'_> {
	StrLines(lines::<Utf8>(input.as_bytes(), ends))
}

impl<'a> Iterator for StrLines<'a> {
	type Item = Line<'a, str>;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		self.0.next().map(|line| Line {
			// SAFETY: the input is a `str`, and every terminator is a whole
			// character (ASCII, or a complete NEL/LS/PS sequence), so lines
			// start and end on character boundaries.
			full:     unsafe { core::str::from_utf8_unchecked(line.full) },
			text_len: line.text_len,
			at:       line.at,
		})
	}

	#[inline]
	fn size_hint(&self) -> (usize, Option<usize>) {
		self.0.size_hint()
	}

	#[inline]
	fn count(self) -> usize {
		self.0.count()
	}
}

impl FusedIterator for StrLines<'_> {}

/// Native value of `unit`.
#[inline(always)]
fn value<E: Encoding>(unit: E::Unit) -> u32 {
	if E::FOREIGN {
		unit.swap_bytes().to_u32()
	} else {
		unit.to_u32()
	}
}

/// Classifies a UTF-16 or UTF-32 window, where every line end is a single
/// unit; foreign units are swapped per 16-lane piece.
macro_rules! classify_wide {
	($window:expr, $ends:expr, $foreign:expr) => {{
		let window = $window;
		let native = |v: Simd<_, 16>| if $foreign { SimdUint::swap_bytes(v) } else { v };
		match $ends {
			LineEnds::Lf => window.bits(|v| native(v).simd_eq(Simd::splat(LF as _))),
			LineEnds::Universal => window.bits(|v| {
				let v = native(v);
				v.simd_eq(Simd::splat(LF as _)) | v.simd_eq(Simd::splat(CR as _))
			}),
			LineEnds::Unicode => window.bits(|v| {
				let v = native(v);
				// LF VT FF CR, FS GS RS, NEL, LS PS.
				(v - Simd::splat(0x0a)).simd_le(Simd::splat(3))
					| (v - Simd::splat(0x1c)).simd_le(Simd::splat(2))
					| v.simd_eq(Simd::splat(0x85))
					| (v - Simd::splat(0x2028)).simd_le(Simd::splat(1))
			}),
		}
	}};
}

/// Line-end candidates of the block at `at`, bit `i` for unit `at + i`; bits
/// past the end are clear.
#[inline(always)]
fn classify<E: Encoding>(input: &[E::Unit], at: usize, ends: LineEnds) -> u64 {
	match E::KIND {
		Kind::Utf8 => {
			// SAFETY: Utf8 is the only sealed encoding with Kind::Utf8.
			let input: &[u8] = unsafe { as_units(input) };
			let (window, origin) = window(input, at);
			let bits = match ends {
				LineEnds::Lf => window.bits(|v| v.simd_eq(Simd::splat(LF as u8))),
				LineEnds::Universal => {
					window.bits(|v| v.simd_eq(Simd::splat(LF as u8)) | v.simd_eq(Simd::splat(CR as u8)))
				},
				LineEnds::Unicode => unicode_u8(&window, input, origin),
			};
			bits >> (at - origin)
		},
		Kind::Utf16 => {
			// SAFETY: Utf16 is the only sealed encoding with Kind::Utf16.
			let input: &[u16] = unsafe { as_units(input) };
			let (window, origin) = window(input, at);
			classify_wide!(window, ends, E::FOREIGN) >> (at - origin)
		},
		Kind::Utf32 => {
			// SAFETY: Utf32 is the only sealed encoding with Kind::Utf32.
			let input: &[u32] = unsafe { as_units(input) };
			let (window, origin) = window(input, at);
			classify_wide!(window, ends, E::FOREIGN) >> (at - origin)
		},
	}
}

/// [`LineEnds::Unicode`] candidates of the UTF-8 window starting at
/// `input[origin]`.
///
/// The ASCII ends are single bytes. NEL (`C2 85`), LS (`E2 80 A8`), and PS
/// (`E2 80 A9`) are marked on their last byte, and only where the whole
/// sequence is present, so a lead byte never needs a lookahead past the
/// window; a sequence that starts before it is completed from the two bytes
/// preceding `origin`.
#[inline(always)]
fn unicode_u8(window: &Window<'_, u8>, input: &[u8], origin: usize) -> u64 {
	let ascii = window.bits(|v| {
		(v - Simd::splat(0x0a)).simd_le(Simd::splat(3))
			| (v - Simd::splat(0x1c)).simd_le(Simd::splat(2))
	});
	if !window.any(|v| v.simd_ge(Simd::splat(0x80))) {
		return ascii;
	}
	let equal = |x: u8| window.bits(|v| v.simd_eq(Simd::splat(x)));
	let (c2, e2, x80, x85) = (equal(0xc2), equal(0xe2), equal(0x80), equal(0x85));
	let a8_a9 = window.bits(|v| (v & Simd::splat(0xfe)).simd_eq(Simd::splat(0xa8)));
	let before = |k: usize| origin.checked_sub(k).map_or(0, |i| input[i]);
	let (b1, b2) = (before(1), before(2));
	let e2_80 = ((e2 << 1) | u64::from(b1 == 0xe2)) & x80;
	let nel = ((c2 << 1) | u64::from(b1 == 0xc2)) & x85;
	let ls_ps = ((e2_80 << 1) | u64::from(b2 == 0xe2 && b1 == 0x80)) & a8_a9;
	ascii | nel | ls_ps
}

/// Line terminators in UTF-8 `data`: LFs, plus lone CRs and CRLFs when
/// `universal` (counted as LF + CR - CRLF).
///
/// Compare results (`-1`/`0` bytes) are subtracted into byte-lane counters,
/// which each gain at most one per block, so 255 blocks run before the lanes
/// are widened and summed: no bitmask, no popcount.
fn count_ends_u8(data: &[u8], universal: bool) -> usize {
	type Lanes = Simd<u8, 16>;
	let lf = Simd::splat(LF as u8);
	let cr = Simd::splat(CR as u8);
	// A CRLF check reads one byte past each block.
	let reach = BLOCK + usize::from(universal);
	let mut total = 0;
	let mut at = 0;
	while data.len() - at >= reach {
		let mut sums = [Lanes::splat(0); 4];
		let mut blocks = 0;
		while blocks < 255 && data.len() - at >= reach {
			for (k, sum) in sums.iter_mut().enumerate() {
				let v = Lanes::from_slice(&data[at + k * 16..]);
				let mut ends = v.simd_eq(lf).to_simd();
				if universal {
					let after = Lanes::from_slice(&data[at + k * 16 + 1..]);
					let crlf = v.simd_eq(cr) & after.simd_eq(lf);
					ends |= (v.simd_eq(cr) & !crlf).to_simd();
				}
				*sum -= ends.cast::<u8>();
			}
			at += BLOCK;
			blocks += 1;
		}
		total += sums
			.iter()
			.map(|sum| sum.cast::<u16>().reduce_sum() as usize)
			.sum::<usize>();
	}
	if at == data.len() {
		return total;
	}
	let (window, origin) = window(data, at);
	let shift = at - origin;
	let lfs = window.bits(|v| v.simd_eq(lf)) >> shift;
	if !universal {
		return total + lfs.count_ones() as usize;
	}
	// The window ends with the data, so a CR at its last bit has no LF after.
	let crs = window.bits(|v| v.simd_eq(cr)) >> shift;
	total + (lfs | (crs & !(lfs >> 1))).count_ones() as usize
}
