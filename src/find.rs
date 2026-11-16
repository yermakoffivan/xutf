//! Code-unit substring search ([`find`], [`rfind`], [`find_iter`]) with an
//! optional ASCII-only case fold.
//!
//! The scanner is Wojciech Muła's generic SIMD filter: for each block of
//! [`BLOCK`] candidate positions it compares the haystack at `p` against the
//! needle's first unit and the haystack at `p + k` against needle unit `k` (the
//! last unit that differs from the first), ANDs the two lane masks into a `u64`
//! and verifies only the surviving positions. Two probes reject almost every
//! position of real text, so the cost is one pair of vector compares per 64
//! units plus a verification per true candidate. [`FindIter`] keeps the
//! unverified candidates of the current block, so successive matches resume
//! from the bitmask instead of rescanning.
//!
//! Matching works on raw code units: no decoding, no normalization, and a match
//! may start or end inside a malformed sequence exactly where the units agree.
//! Since needle units are compared in full, a match found in well-formed text
//! always starts and ends on codepoint boundaries. The caseless variants fold
//! `A-Z` to `a-z` and compare every other unit exactly; that is sound for UTF-8
//! because folding never touches bytes `>= 0x80`. Foreign-endian units are
//! swapped before folding.
//!
//! # Complexity
//! `O(n + c * m)` for a haystack of `n` units, a needle of `m` units and `c`
//! filter candidates. Adversarial inputs where every position passes the filter
//! (`"aaaa…"` searched for `"aa…ab"`) degrade to `O(n * m)`; there is no
//! Two-Way fallback.

use core::{
	iter::FusedIterator,
	marker::PhantomData,
	simd::{
		Simd,
		cmp::{SimdOrd, SimdPartialEq},
		num::SimdUint,
	},
};

use crate::{
	encoding::{Encoding, Kind},
	simd::{BLOCK, bits64, low_bits, raw_eq_ignore_ascii_case},
	unit::{Unit, as_units, fold_case_scalar},
};

/// Concrete per-width SIMD operations the generic scanner needs.
trait Lane: Unit {
	/// Candidate bits of a block: lane `i` set when `x[i]` is one of `a` and,
	/// unless `single`, `y[i]` is one of `b`.
	fn pair_bits<const FOLD: bool>(
		x: Simd<Self, BLOCK>,
		a: [Self; 2],
		y: Simd<Self, BLOCK>,
		b: [Self; 2],
		single: bool,
	) -> u64;
	/// Reverses the byte order of every lane.
	fn swap_lanes<const N: usize>(v: Simd<Self, N>) -> Simd<Self, N>;
}

macro_rules! impl_lane {
	($($t:ty),*) => {$(
		impl Lane for $t {
			#[inline(always)]
			fn pair_bits<const FOLD: bool>(
				x: Simd<Self, BLOCK>,
				a: [Self; 2],
				y: Simd<Self, BLOCK>,
				b: [Self; 2],
				single: bool,
			) -> u64 {
				// Lane-wise distances instead of 64-lane masks: a lane is zero
				// exactly when it equals one of the accepted units, so the
				// candidates are the zero lanes of `first | last`.
				let distance = |v: Simd<Self, BLOCK>, units: [Self; 2]| {
					let d = v ^ Simd::splat(units[0]);
					if FOLD {
						d.simd_min(v ^ Simd::splat(units[1]))
					} else {
						d
					}
				};
				let mut misses = distance(x, a);
				if !single {
					misses |= distance(y, b);
				}
				bits64(misses, |v| v.simd_eq(Simd::splat(0)))
			}

			#[inline(always)]
			fn swap_lanes<const N: usize>(v: Simd<Self, N>) -> Simd<Self, N> {
				v.swap_bytes()
			}
		}
	)*};
}

impl_lane!(u8, u16, u32);

/// The two filter probes of a needle, as raw (possibly foreign) unit values.
#[derive(Clone, Copy, Debug)]
struct Probe {
	/// Offset of the second probe; `0` for a one-unit needle.
	k:     usize,
	/// Raw units accepted at offset `0` (both equal unless folding a letter).
	a:     [u32; 2],
	/// Raw units accepted at offset `k`.
	b:     [u32; 2],
	/// The probes cover every needle unit, so a candidate is a match.
	exact: bool,
}

/// Value of `unit` in native order.
#[inline(always)]
fn value<L: Lane, const FOREIGN: bool>(unit: L) -> u32 {
	if FOREIGN {
		unit.swap_bytes().to_u32()
	} else {
		unit.to_u32()
	}
}

/// Raw units that equal `unit` under the search's comparison.
#[inline(always)]
fn alternatives<L: Lane, const FOREIGN: bool, const FOLD: bool>(unit: L) -> [u32; 2] {
	if !FOLD {
		return [unit.to_u32(); 2];
	}
	let v = value::<L, FOREIGN>(unit);
	let raw = |x: u32| {
		let x = L::from_u32(x);
		(if FOREIGN { x.swap_bytes() } else { x }).to_u32()
	};
	[raw(fold_case_scalar(v, 'A' as u32)), raw(fold_case_scalar(v, 'a' as u32))]
}

/// Picks the probes for a non-empty `needle`.
#[inline(always)]
fn probe<L: Lane, const FOREIGN: bool, const FOLD: bool>(needle: &[L]) -> Probe {
	let key = |u: L| {
		if FOLD {
			fold_case_scalar(value::<L, FOREIGN>(u), 'A' as u32)
		} else {
			u.to_u32()
		}
	};
	let m = needle.len();
	let first = key(needle[0]);
	// The last unit that differs from the first: a repeated unit makes the
	// second probe redundant (`"aa"`, `"=="`), a distinct one roughly squares
	// the filter's rejection rate.
	let k = (1..m)
		.rev()
		.find(|&j| key(needle[j]) != first)
		.unwrap_or(m - 1);
	Probe {
		k,
		a: alternatives::<L, FOREIGN, FOLD>(needle[0]),
		b: alternatives::<L, FOREIGN, FOLD>(needle[k]),
		exact: m <= 2,
	}
}

/// Candidate bits of the full block of positions `at..at + BLOCK`.
///
/// # Safety
/// `at + probe.k + BLOCK <= hay.len()`.
#[inline(always)]
unsafe fn block<L: Lane, const FOLD: bool>(hay: &[L], probe: &Probe, at: usize) -> u64 {
	let base = hay.as_ptr();
	// SAFETY: the caller guarantees `at + k + BLOCK <= hay.len()`, so both
	// unaligned 64-unit reads stay in bounds.
	let (x, y) = unsafe {
		(
			base.add(at).cast::<Simd<L, BLOCK>>().read_unaligned(),
			base
				.add(at + probe.k)
				.cast::<Simd<L, BLOCK>>()
				.read_unaligned(),
		)
	};
	L::pair_bits::<FOLD>(x, lanes(probe.a), y, lanes(probe.b), probe.k == 0)
}

/// Raw probe values as units.
#[inline(always)]
fn lanes<L: Lane>(raw: [u32; 2]) -> [L; 2] {
	[L::from_u32(raw[0]), L::from_u32(raw[1])]
}

/// The [`BLOCK`] units of `hay` from `at`, zero-filled past the end.
#[inline(always)]
fn padded<L: Lane>(hay: &[L], at: usize) -> Simd<L, BLOCK> {
	let rest = &hay[at..hay.len().min(at + BLOCK)];
	let mut block = [L::default(); BLOCK];
	block[..rest.len()].copy_from_slice(rest);
	Simd::from_array(block)
}

/// Candidate bits of positions `at..at + BLOCK` clipped to the `npos` valid
/// start positions (`at < npos`); bit `i` stands for position `at + i`.
#[inline(always)]
fn window<L: Lane, const FOLD: bool>(hay: &[L], npos: usize, probe: &Probe, at: usize) -> u64 {
	if at + BLOCK <= npos {
		// SAFETY: `npos = len - m + 1` and `k < m`, so `at + BLOCK <= npos`
		// gives `at + k + BLOCK <= len`.
		unsafe { block::<L, FOLD>(hay, probe, at) }
	} else if npos >= BLOCK {
		// Overlapped final block, shifted so bit 0 is position `at` again.
		let base = npos - BLOCK;
		// SAFETY: `base + BLOCK == npos`; see above.
		unsafe { block::<L, FOLD>(hay, probe, base) >> (at - base) }
	} else {
		short_window::<L, FOLD>(hay, npos, probe, at)
	}
}

/// Start positions up to which [`short_window`] tests units one at a time:
/// below this a short loop is cheaper than zero-filling and copying two
/// padded blocks.
const SCALAR_POSITIONS: usize = 32;

/// [`window`] for haystacks with fewer than [`BLOCK`] start positions.
#[inline(never)]
fn short_window<L: Lane, const FOLD: bool>(
	hay: &[L],
	npos: usize,
	probe: &Probe,
	at: usize,
) -> u64 {
	let count = npos - at;
	let [a0, a1] = lanes::<L>(probe.a);
	let [b0, b1] = lanes::<L>(probe.b);
	if count <= SCALAR_POSITIONS {
		let xs = &hay[at..at + count];
		let ys = &hay[at + probe.k..at + probe.k + count];
		let mut bits = 0;
		for (i, (&x, &y)) in xs.iter().zip(ys).enumerate() {
			#[allow(clippy::needless_bitwise_bool, reason = "keeps the loop branch-free")]
			let hit = ((x == a0) | (FOLD & (x == a1))) & ((y == b0) | (FOLD & (y == b1)));
			bits |= u64::from(hit) << i;
		}
		return bits;
	}
	let x = padded(hay, at);
	let y = if probe.k == 0 {
		x
	} else {
		padded(hay, at + probe.k)
	};
	// Zero padding can match a NUL probe; clip to the real positions.
	L::pair_bits::<FOLD>(x, [a0, a1], y, [b0, b1], probe.k == 0) & low_bits(count)
}

/// ASCII-caseless equality of two equal-length unit runs.
#[inline(always)]
fn eq_fold<L: Lane, const FOREIGN: bool>(a: &[L], b: &[L]) -> bool {
	const N: usize = 16;
	if !FOREIGN {
		return raw_eq_ignore_ascii_case::<L, N>(a, b);
	}
	let len = a.len();
	if len < N {
		return a.iter().zip(b).all(|(&x, &y)| {
			fold_case_scalar(value::<L, true>(x), 'A' as u32)
				== fold_case_scalar(value::<L, true>(y), 'A' as u32)
		});
	}
	let fold = |s: &[L], i: usize| {
		L::fold_case(L::swap_lanes(Simd::<L, N>::from_slice(&s[i..])), 'A' as u32)
	};
	let mut i = 0;
	loop {
		if fold(a, i) != fold(b, i) {
			return false;
		}
		i += N;
		if i + N > len {
			if i == len {
				return true;
			}
			// Overlapped tail window; re-checking units is harmless.
			i = len - N;
		}
	}
}

/// Whether the needle occurs at candidate position `pos`.
#[inline(always)]
fn verify<L: Lane, const FOREIGN: bool, const FOLD: bool>(
	hay: &[L],
	needle: &[L],
	probe: &Probe,
	pos: usize,
) -> bool {
	if probe.exact {
		return true;
	}
	let at = &hay[pos..pos + needle.len()];
	if FOLD {
		eq_fold::<L, FOREIGN>(at, needle)
	} else if needle.len() <= SHORT_NEEDLE {
		// The probes already matched, so a mismatch usually shows up within
		// the first units; an inline loop beats a `bcmp` call here.
		at.iter().zip(needle).all(|(x, y)| x == y)
	} else {
		at == needle
	}
}

/// Needle length up to which [`verify`] compares unit by unit.
const SHORT_NEEDLE: usize = 16;

/// Number of valid start positions: `len - m + 1`, or `0` when the needle is
/// longer than the haystack.
#[inline(always)]
const fn positions(len: usize, m: usize) -> usize {
	(len + 1).saturating_sub(m)
}

fn find_impl<L: Lane, const FOREIGN: bool, const FOLD: bool>(
	hay: &[L],
	needle: &[L],
) -> Option<usize> {
	if needle.is_empty() {
		return Some(0);
	}
	let npos = positions(hay.len(), needle.len());
	let probe = probe::<L, FOREIGN, FOLD>(needle);
	let mut at = 0;
	while at < npos {
		let mut bits = window::<L, FOLD>(hay, npos, &probe, at);
		while bits != 0 {
			let pos = at + bits.trailing_zeros() as usize;
			if verify::<L, FOREIGN, FOLD>(hay, needle, &probe, pos) {
				return Some(pos);
			}
			bits &= bits - 1;
		}
		at += BLOCK;
	}
	None
}

fn rfind_impl<L: Lane, const FOREIGN: bool, const FOLD: bool>(
	hay: &[L],
	needle: &[L],
) -> Option<usize> {
	if needle.is_empty() {
		return Some(hay.len());
	}
	let npos = positions(hay.len(), needle.len());
	let probe = probe::<L, FOREIGN, FOLD>(needle);
	let mut hi = npos;
	while hi > 0 {
		let base = hi.saturating_sub(BLOCK);
		let mut bits = window::<L, FOLD>(hay, npos, &probe, base) & low_bits(hi - base);
		while bits != 0 {
			let top = 63 - bits.leading_zeros() as usize;
			let pos = base + top;
			if verify::<L, FOREIGN, FOLD>(hay, needle, &probe, pos) {
				return Some(pos);
			}
			bits ^= 1 << top;
		}
		hi = base;
	}
	None
}

/// Whether a search over `E` must swap units: only the fold looks at values.
#[inline(always)]
const fn swapped<E: Encoding>(fold: bool) -> bool {
	fold && E::FOREIGN
}

/// Calls `$f::<unit, FOREIGN, $fold>` on the haystack and needle reinterpreted
/// as `E`'s concrete unit type.
macro_rules! dispatch {
	($E:ty, $f:ident:: < $fold:literal > ($hay:expr, $needle:expr $(, $rest:expr)*)) => {
		// SAFETY: `Encoding` is sealed and each `Kind` has exactly one unit
		// type: u8 for UTF-8, u16 for UTF-16 and u32 for UTF-32.
		unsafe {
			match <$E>::KIND {
				Kind::Utf8 => $f::<u8, false, $fold>(as_units($hay), as_units($needle) $(, $rest)*),
				// Exact search compares raw units, so byte order only matters
				// to the fold.
				Kind::Utf16 if swapped::<$E>($fold) => {
					$f::<u16, true, $fold>(as_units($hay), as_units($needle) $(, $rest)*)
				},
				Kind::Utf16 => $f::<u16, false, $fold>(as_units($hay), as_units($needle) $(, $rest)*),
				Kind::Utf32 if swapped::<$E>($fold) => {
					$f::<u32, true, $fold>(as_units($hay), as_units($needle) $(, $rest)*)
				},
				Kind::Utf32 => $f::<u32, false, $fold>(as_units($hay), as_units($needle) $(, $rest)*),
			}
		}
	};
}

/// Offset of the first occurrence of `needle` in `haystack`, comparing raw code
/// units.
///
/// The SIMD replacement for `windows(n).position(..)` or a restart-per-match
/// loop over `u16`/`u32` text; for `&str` and bytes it is a drop-in
/// alternative to `str::find`. An empty needle matches at `0`.
///
/// # Example
/// ```
/// use xutf::{Utf8, Utf16, find};
///
/// assert_eq!(find::<Utf8>(b"let x = y;", b"= y"), Some(6));
/// let wide: Vec<u16> = "日本語のテキスト".encode_utf16().collect();
/// let needle: Vec<u16> = "テキ".encode_utf16().collect();
/// assert_eq!(find::<Utf16>(&wide, &needle), Some(4));
/// assert_eq!(find::<Utf8>(b"abc", b""), Some(0));
/// ```
#[inline]
pub fn find<E: Encoding>(haystack: &[E::Unit], needle: &[E::Unit]) -> Option<usize> {
	dispatch!(E, find_impl::<false>(haystack, needle))
}

/// Offset of the last occurrence of `needle` in `haystack`, comparing raw code
/// units.
///
/// Scans from the end, so a match near the back costs only the units after it.
/// An empty needle matches at `haystack.len()`.
///
/// # Example
/// ```
/// use xutf::{Utf8, rfind};
///
/// assert_eq!(rfind::<Utf8>(b"a/b/c.rs", b"/"), Some(3));
/// assert_eq!(rfind::<Utf8>(b"abc", b""), Some(3));
/// ```
#[inline]
pub fn rfind<E: Encoding>(haystack: &[E::Unit], needle: &[E::Unit]) -> Option<usize> {
	dispatch!(E, rfind_impl::<false>(haystack, needle))
}

/// Offset of the first occurrence of `needle` in `haystack`, folding ASCII
/// `A-Z` to `a-z` on both sides and comparing all other units exactly.
///
/// Use it for case-insensitive keyword, header or identifier lookups without
/// lowercasing a copy of the haystack.
///
/// # Example
/// ```
/// use xutf::{Utf8, Utf16Be, find_ignore_ascii_case};
///
/// assert_eq!(find_ignore_ascii_case::<Utf8>(b"Content-TYPE: text", b"content-type"), Some(0));
/// // Non-ASCII units compare exactly: no Unicode case folding.
/// assert_eq!(find_ignore_ascii_case::<Utf8>("ÉCOLE".as_bytes(), "école".as_bytes()), None);
/// // Foreign-endian units are swapped before folding.
/// let be = |s: &str| s.encode_utf16().map(u16::to_be).collect::<Vec<_>>();
/// assert_eq!(find_ignore_ascii_case::<Utf16Be>(&be("Hello World"), &be("WORLD")), Some(6));
/// ```
#[inline]
pub fn find_ignore_ascii_case<E: Encoding>(
	haystack: &[E::Unit],
	needle: &[E::Unit],
) -> Option<usize> {
	dispatch!(E, find_impl::<true>(haystack, needle))
}

/// Offset of the last occurrence of `needle` in `haystack` under the ASCII
/// case fold of [`find_ignore_ascii_case`].
///
/// # Example
/// ```
/// use xutf::{Utf8, rfind_ignore_ascii_case};
///
/// assert_eq!(rfind_ignore_ascii_case::<Utf8>(b"Foo foo FOO", b"foo"), Some(8));
/// ```
#[inline]
pub fn rfind_ignore_ascii_case<E: Encoding>(
	haystack: &[E::Unit],
	needle: &[E::Unit],
) -> Option<usize> {
	dispatch!(E, rfind_impl::<true>(haystack, needle))
}

/// Byte offset of the first ASCII-caseless occurrence of `needle` in
/// `haystack`; the `&str` form of [`find_ignore_ascii_case`].
///
/// The result is always a char boundary of `haystack`, so it can slice it.
///
/// # Example
/// ```
/// use xutf::find_ignore_ascii_case_str;
///
/// let line = "2024-01-01 WARN disk almost full";
/// let at = find_ignore_ascii_case_str(line, "warn").unwrap();
/// assert_eq!(&line[at..at + 4], "WARN");
/// ```
#[inline]
pub fn find_ignore_ascii_case_str(haystack: &str, needle: &str) -> Option<usize> {
	find_impl::<u8, false, true>(haystack.as_bytes(), needle.as_bytes())
}

/// Byte offset of the last ASCII-caseless occurrence of `needle` in
/// `haystack`; the `&str` form of [`rfind_ignore_ascii_case`].
///
/// The result is always a char boundary of `haystack`.
#[inline]
pub fn rfind_ignore_ascii_case_str(haystack: &str, needle: &str) -> Option<usize> {
	rfind_impl::<u8, false, true>(haystack.as_bytes(), needle.as_bytes())
}

/// Iterator over the unit offsets of non-overlapping occurrences of a needle,
/// left to right.
///
/// Created by [`find_iter`] and [`find_iter_ignore_ascii_case`]. It keeps the
/// current block's unverified candidates, so each call resumes from the bitmask
/// instead of rescanning.
pub struct FindIter<'a, E: Encoding> {
	haystack:  &'a [E::Unit],
	needle:    &'a [E::Unit],
	fold:      bool,
	cursor:    Cursor,
	_encoding: PhantomData<E>,
}

/// Scan state of a [`FindIter`], independent of the unit type.
#[derive(Clone, Copy)]
struct Cursor {
	probe: Probe,
	/// Valid start positions: `haystack.len() - needle.len() + 1`, or `0`.
	npos:  usize,
	/// Position of bit 0 of `bits`.
	block: usize,
	/// Unverified candidates of the current block.
	bits:  u64,
	/// First position not yet classified (the next offset for an empty
	/// needle).
	next:  usize,
}

impl<E: Encoding> Clone for FindIter<'_, E> {
	#[inline]
	fn clone(&self) -> Self {
		Self { ..*self }
	}
}

impl<E: Encoding> core::fmt::Debug for FindIter<'_, E> {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		f.debug_struct("FindIter")
			.field("needle_len", &self.needle.len())
			.field("fold", &self.fold)
			.field("next", &self.cursor.next)
			.finish_non_exhaustive()
	}
}

impl<'a, E: Encoding> FindIter<'a, E> {
	fn new(haystack: &'a [E::Unit], needle: &'a [E::Unit], fold: bool) -> Self {
		let probe = if needle.is_empty() {
			Probe { k: 0, a: [0; 2], b: [0; 2], exact: true }
		} else if fold {
			dispatch!(E, probe_of::<true>(haystack, needle))
		} else {
			dispatch!(E, probe_of::<false>(haystack, needle))
		};
		Self {
			haystack,
			needle,
			fold,
			cursor: Cursor {
				probe,
				npos: positions(haystack.len(), needle.len()),
				block: 0,
				bits: 0,
				next: 0,
			},
			_encoding: PhantomData,
		}
	}
}

/// [`probe`] in the `dispatch!` calling shape.
#[inline(always)]
fn probe_of<L: Lane, const FOREIGN: bool, const FOLD: bool>(_: &[L], needle: &[L]) -> Probe {
	probe::<L, FOREIGN, FOLD>(needle)
}

/// Advances a non-empty-needle [`FindIter`] state to its next match.
#[inline(always)]
fn iter_next<L: Lane, const FOREIGN: bool, const FOLD: bool>(
	hay: &[L],
	needle: &[L],
	c: &mut Cursor,
) -> Option<usize> {
	loop {
		while c.bits != 0 {
			let pos = c.block + c.bits.trailing_zeros() as usize;
			c.bits &= c.bits - 1;
			if verify::<L, FOREIGN, FOLD>(hay, needle, &c.probe, pos) {
				// Non-overlapping: drop candidates inside this match.
				let skip = pos + needle.len() - c.block;
				if skip >= BLOCK {
					c.bits = 0;
					c.next = pos + needle.len();
				} else {
					c.bits &= u64::MAX << skip;
				}
				return Some(pos);
			}
		}
		if c.next >= c.npos {
			return None;
		}
		c.block = c.next;
		c.bits = window::<L, FOLD>(hay, c.npos, &c.probe, c.next);
		c.next += BLOCK;
	}
}

impl<E: Encoding> Iterator for FindIter<'_, E> {
	type Item = usize;

	#[inline]
	fn next(&mut self) -> Option<usize> {
		let c = &mut self.cursor;
		if self.needle.is_empty() {
			if c.next >= c.npos {
				return None;
			}
			c.next += 1;
			return Some(c.next - 1);
		}
		if self.fold {
			dispatch!(E, iter_next::<true>(self.haystack, self.needle, c))
		} else {
			dispatch!(E, iter_next::<false>(self.haystack, self.needle, c))
		}
	}

	#[inline]
	fn size_hint(&self) -> (usize, Option<usize>) {
		let c = &self.cursor;
		let rest = c.npos.saturating_sub(c.next);
		if self.needle.is_empty() {
			(rest, Some(rest))
		} else {
			(0, Some(rest + c.bits.count_ones() as usize))
		}
	}
}

impl<E: Encoding> FusedIterator for FindIter<'_, E> {}

/// Iterates the unit offsets of the non-overlapping occurrences of `needle` in
/// `haystack`, left to right.
///
/// Use it to count or replace every occurrence; `"aaaa"` searched for `"aa"`
/// yields `0` and `2`. An empty needle yields every offset `0..=len`.
///
/// # Example
/// ```
/// use xutf::{Utf8, find_iter};
///
/// let hits: Vec<usize> = find_iter::<Utf8>(b"a,b,,c", b",").collect();
/// assert_eq!(hits, [1, 3, 4]);
/// assert_eq!(find_iter::<Utf8>(b"aaaa", b"aa").collect::<Vec<_>>(), [0, 2]);
/// ```
#[inline]
pub fn find_iter<'a, E: Encoding>(
	haystack: &'a [E::Unit],
	needle: &'a [E::Unit],
) -> FindIter<'a, E> {
	FindIter::new(haystack, needle, false)
}

/// Iterates the unit offsets of the non-overlapping ASCII-caseless occurrences
/// of `needle` in `haystack`, left to right.
///
/// The iterator form of [`find_ignore_ascii_case`].
///
/// # Example
/// ```
/// use xutf::{Utf8, find_iter_ignore_ascii_case};
///
/// let text = b"TODO: x; todo: y; ToDo: z";
/// assert_eq!(find_iter_ignore_ascii_case::<Utf8>(text, b"todo").count(), 3);
/// ```
#[inline]
pub fn find_iter_ignore_ascii_case<'a, E: Encoding>(
	haystack: &'a [E::Unit],
	needle: &'a [E::Unit],
) -> FindIter<'a, E> {
	FindIter::new(haystack, needle, true)
}
