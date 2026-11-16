//! Differential tests for `find`/`rfind`/`find_iter` and their ASCII-caseless
//! variants against a naive scalar reference, in every encoding and byte order.

use xutf::{
	Encoding, Unit, Utf8, Utf16, Utf32, find, find_ignore_ascii_case, find_ignore_ascii_case_str,
	find_iter, find_iter_ignore_ascii_case, rfind, rfind_ignore_ascii_case,
	rfind_ignore_ascii_case_str,
};

/// Fixed-seed xorshift64 generator.
struct Rng(u64);

impl Rng {
	const fn step(&mut self) -> u64 {
		self.0 ^= self.0 << 13;
		self.0 ^= self.0 >> 7;
		self.0 ^= self.0 << 17;
		self.0
	}

	const fn below(&mut self, n: usize) -> usize {
		(self.step() % n as u64) as usize
	}
}

const fn key(v: u32, fold: bool) -> u32 {
	if fold && v.wrapping_sub('A' as u32) <= 25 {
		v ^ 0x20
	} else {
		v
	}
}

fn matches_at(hay: &[u32], needle: &[u32], pos: usize, fold: bool) -> bool {
	hay[pos..pos + needle.len()]
		.iter()
		.zip(needle)
		.all(|(&h, &n)| key(h, fold) == key(n, fold))
}

fn ref_find(hay: &[u32], needle: &[u32], fold: bool) -> Option<usize> {
	if needle.len() > hay.len() {
		return None;
	}
	(0..=hay.len() - needle.len()).find(|&p| matches_at(hay, needle, p, fold))
}

fn ref_rfind(hay: &[u32], needle: &[u32], fold: bool) -> Option<usize> {
	if needle.len() > hay.len() {
		return None;
	}
	(0..=hay.len() - needle.len())
		.rev()
		.find(|&p| matches_at(hay, needle, p, fold))
}

fn ref_iter(hay: &[u32], needle: &[u32], fold: bool) -> Vec<usize> {
	if needle.is_empty() {
		return (0..=hay.len()).collect();
	}
	let mut out = Vec::new();
	let mut p = 0;
	while p + needle.len() <= hay.len() {
		if matches_at(hay, needle, p, fold) {
			out.push(p);
			p += needle.len();
		} else {
			p += 1;
		}
	}
	out
}

/// Encodes codepoint-domain unit values as `E`'s raw (possibly swapped) units.
fn units<E: Encoding>(values: &[u32]) -> Vec<E::Unit> {
	values
		.iter()
		.map(|&v| {
			let u = E::Unit::from_u32(v);
			if E::FOREIGN { u.swap_bytes() } else { u }
		})
		.collect()
}

/// Unit values for `E`: ASCII letters in both cases, the bytes around the fold
/// range, NUL (the padding value), non-ASCII and, for wide units, values whose
/// byte swap is an ASCII letter.
fn alphabet<E: Encoding>() -> Vec<u32> {
	let mut values = vec![
		b'a' as u32,
		b'A' as u32,
		b'b' as u32,
		b'B' as u32,
		b'z' as u32,
		b'Z' as u32,
		b'@' as u32,
		b'[' as u32,
		b'`' as u32,
		b'{' as u32,
		0,
		0xc1,
		0xe1,
	];
	if size_of::<E::Unit>() >= 2 {
		values.extend([0x4100, 0x6100, 0x0141, 0x3042]);
	}
	if size_of::<E::Unit>() == 4 {
		values.extend([0x4100_0000, 0x1_f600]);
	}
	values
}

fn check_case<E: Encoding>(hay_v: &[u32], needle_v: &[u32]) {
	let hay = units::<E>(hay_v);
	let needle = units::<E>(needle_v);
	let ctx = || format!("{:?} hay={hay_v:x?} needle={needle_v:x?}", E::KIND);
	for fold in [false, true] {
		let (f, r, mut it) = if fold {
			(
				find_ignore_ascii_case::<E>(&hay, &needle),
				rfind_ignore_ascii_case::<E>(&hay, &needle),
				find_iter_ignore_ascii_case::<E>(&hay, &needle),
			)
		} else {
			(find::<E>(&hay, &needle), rfind::<E>(&hay, &needle), find_iter::<E>(&hay, &needle))
		};
		assert_eq!(f, ref_find(hay_v, needle_v, fold), "find fold={fold} {}", ctx());
		assert_eq!(r, ref_rfind(hay_v, needle_v, fold), "rfind fold={fold} {}", ctx());
		let expected = ref_iter(hay_v, needle_v, fold);
		let (lo, hi) = it.size_hint();
		let mut got = Vec::new();
		// Clone midway: the copy must resume at the same point.
		if let Some(first) = it.next() {
			got.push(first);
			let rest: Vec<usize> = it.clone().collect();
			got.extend(&rest);
			assert_eq!(it.by_ref().collect::<Vec<_>>(), rest, "clone fold={fold} {}", ctx());
		}
		assert_eq!(it.next(), None, "fused fold={fold} {}", ctx());
		assert_eq!(it.next(), None, "fused fold={fold} {}", ctx());
		assert_eq!(got, expected, "find_iter fold={fold} {}", ctx());
		assert!(lo <= got.len() && hi.is_none_or(|hi| got.len() <= hi), "size_hint {}", ctx());
	}
}

fn random_differential<E: Encoding>(seed: u64) {
	let mut rng = Rng(seed);
	let full = alphabet::<E>();
	for len in 0..=200 {
		for round in 0..6 {
			// Tiny alphabets make dense partial matches; the full one mixes in
			// the near-miss values.
			let size = [2, 3, 4, full.len()][round % 4].min(full.len());
			let start = rng.below(full.len() - size + 1);
			let pool = &full[start..start + size];
			let hay: Vec<u32> = (0..len).map(|_| pool[rng.below(pool.len())]).collect();
			for _ in 0..4 {
				// A substring of the haystack (always matches), possibly with
				// its letters' case flipped, or a random needle.
				let needle: Vec<u32> = if len > 0 && rng.below(3) != 0 {
					let at = rng.below(len);
					let m = 1 + rng.below((len - at).min(70));
					hay[at..at + m]
						.iter()
						.map(|&v| if rng.below(2) == 0 { key(v, true) } else { v })
						.collect()
				} else {
					let m = rng.below(8);
					(0..m).map(|_| pool[rng.below(pool.len())]).collect()
				};
				check_case::<E>(&hay, &needle);
			}
			check_case::<E>(&hay, &[]);
			check_case::<E>(&hay, &hay);
			let mut longer = hay.clone();
			longer.push(pool[0]);
			check_case::<E>(&hay, &longer);
		}
	}
}

/// Plants a needle at every position of a haystack that contains no other
/// occurrence, so the first, last and block-straddling offsets all get hit.
fn planted<E: Encoding>() {
	let filler = b'x' as u32;
	let needles: [&[u32]; 7] = [
		&[b'q' as u32],
		&[b'q' as u32, b'Q' as u32],
		&[b'a' as u32, b'x' as u32, b'B' as u32],
		&[0, b'Y' as u32],
		&[b'y' as u32, b'y' as u32, b'y' as u32, b'Z' as u32],
		&[b'k' as u32; 64],
		&[b'k' as u32; 65],
	];
	for needle in needles {
		for len in needle.len()..=200 {
			for pos in 0..=len - needle.len() {
				let mut hay = vec![filler; len];
				// Plant the case-flipped needle: exact search must miss the
				// letters, caseless search must find it.
				for (i, &v) in needle.iter().enumerate() {
					let letter = u8::try_from(v).is_ok_and(|b| b.is_ascii_alphabetic());
					hay[pos + i] = if letter { v ^ 0x20 } else { v };
				}
				check_case::<E>(&hay, needle);
				hay[pos..pos + needle.len()].copy_from_slice(needle);
				check_case::<E>(&hay, needle);
			}
		}
	}
}

/// Non-overlapping semantics on fully periodic input across block boundaries.
fn periodic<E: Encoding>() {
	for len in 0..=200 {
		let hay = vec![b'a' as u32; len];
		for m in 1..=5 {
			check_case::<E>(&hay, &vec![b'a' as u32; m]);
			check_case::<E>(&hay, &vec![b'A' as u32; m]);
		}
		let ab: Vec<u32> = (0..len).map(|i| u32::from(b"ab"[i % 2])).collect();
		check_case::<E>(&ab, &[b'a' as u32, b'b' as u32, b'a' as u32]);
		check_case::<E>(&ab, &[b'B' as u32, b'A' as u32]);
	}
}

macro_rules! encoding_tests {
	($($name:ident: $enc:ty, $seed:expr;)*) => {
		mod random {
			use super::*;
			$(#[test] fn $name() { random_differential::<$enc>($seed); })*
		}
		mod planted {
			use super::*;
			$(#[test] fn $name() { planted::<$enc>(); })*
		}
		mod periodic {
			use super::*;
			$(#[test] fn $name() { periodic::<$enc>(); })*
		}
	};
}

encoding_tests! {
	utf8: Utf8, 0x9e37_79b9_7f4a_7c15;
	utf16: Utf16<false>, 0xd1b5_4a32_d192_ed03;
	utf16_foreign: Utf16<true>, 0x8cb9_2ba7_2f3d_8dd7;
	utf32: Utf32<false>, 0xa076_1d64_78bd_642f;
	utf32_foreign: Utf32<true>, 0xe703_7ed1_a0b4_28db;
}

#[test]
fn empty_needle_and_haystack() {
	assert_eq!(find::<Utf8>(b"", b""), Some(0));
	assert_eq!(rfind::<Utf8>(b"", b""), Some(0));
	assert_eq!(find::<Utf8>(b"", b"a"), None);
	assert_eq!(rfind::<Utf8>(b"ab", b"abc"), None);
	assert_eq!(find_iter::<Utf8>(b"ab", b"").collect::<Vec<_>>(), [0, 1, 2]);
	assert_eq!(find_iter::<Utf8>(b"", b"").collect::<Vec<_>>(), [0]);
	assert_eq!(find_iter::<Utf8>(b"ab", b"").size_hint(), (3, Some(3)));
}

#[test]
fn str_variants_return_char_boundaries() {
	let pieces = ["a", "A", "é", "É", "日", "ß", "K", "k", "\u{212a}", " "];
	let mut rng = Rng(0x2545_f491_4f6c_dd1d);
	for len in 0..=80 {
		let hay: String = (0..len).map(|_| pieces[rng.below(pieces.len())]).collect();
		for _ in 0..8 {
			let needle: String = (0..rng.below(4))
				.map(|_| pieces[rng.below(pieces.len())])
				.collect();
			let hb: Vec<u32> = hay.bytes().map(u32::from).collect();
			let nb: Vec<u32> = needle.bytes().map(u32::from).collect();
			let f = find_ignore_ascii_case_str(&hay, &needle);
			let r = rfind_ignore_ascii_case_str(&hay, &needle);
			assert_eq!(f, ref_find(&hb, &nb, true), "{hay:?} {needle:?}");
			assert_eq!(r, ref_rfind(&hb, &nb, true), "{hay:?} {needle:?}");
			for at in [f, r].into_iter().flatten() {
				assert!(hay.is_char_boundary(at) && hay.is_char_boundary(at + needle.len()));
				assert!(hay[at..at + needle.len()].eq_ignore_ascii_case(&needle));
			}
		}
	}
}

#[test]
fn non_ascii_units_compare_exactly() {
	// Only ASCII letters fold: Kelvin sign, sharp s and accented capitals don't.
	assert_eq!(find_ignore_ascii_case_str("\u{212a}elvin", "kelvin"), None);
	assert_eq!(find_ignore_ascii_case_str("STRASSE", "straße"), None);
	assert_eq!(find_ignore_ascii_case_str("CAFÉ café", "CAFé"), Some(6));
	assert_eq!(find_ignore_ascii_case_str("CAFÉ", "café"), None);
	assert_eq!(find_ignore_ascii_case_str("CAFÉ café", "CAFÉ"), Some(0));
	assert_eq!(find_ignore_ascii_case_str("xx CAFé", "café"), Some(3));
}
