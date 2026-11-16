//! Differential tests of `UnitSet` against scalar references over every
//! encoding, including foreign-endian UTF-16/UTF-32.
#![allow(
	clippy::single_range_in_vec_init,
	reason = "single-range sets exercise `UnitSet::from_ranges`"
)]

use xutf::{Encoding, Unit, UnitSet, Utf8, Utf16, Utf32};

struct Rng(u64);

impl Rng {
	fn step(&mut self) -> u64 {
		self.0 ^= self.0 << 13;
		self.0 ^= self.0 >> 7;
		self.0 ^= self.0 << 17;
		self.0
	}

	fn pick<T: Copy>(&mut self, items: &[T]) -> T {
		items[(self.step() % items.len() as u64) as usize]
	}
}

/// Raw units of encoding `E` holding the logical `values` (swapped when
/// foreign).
fn encode<E: Encoding>(values: &[u32]) -> Vec<E::Unit> {
	values
		.iter()
		.map(|&v| {
			let unit = <E::Unit as Unit>::from_u32(v);
			if E::FOREIGN { unit.swap_bytes() } else { unit }
		})
		.collect()
}

/// Membership reference built from an explicit member list.
fn member_table(members: &[u8]) -> impl Fn(u32) -> bool + '_ {
	move |v| v < 256 && members.contains(&(v as u8))
}

fn ref_fields(values: &[u32], is: &dyn Fn(u32) -> bool) -> Vec<(usize, usize)> {
	let mut out = Vec::new();
	let mut start = None;
	for (i, &v) in values.iter().enumerate() {
		match (is(v), start) {
			(false, None) => start = Some(i),
			(true, Some(s)) => {
				out.push((s, i - s));
				start = None;
			},
			_ => {},
		}
	}
	if let Some(s) = start {
		out.push((s, values.len() - s));
	}
	out
}

fn offset_of<T>(whole: &[T], part: &[T]) -> usize {
	(part.as_ptr() as usize - whole.as_ptr() as usize) / size_of::<T>()
}

/// Checks every scan of `set` on `values` encoded as `E` against scalar
/// references using `is` for membership.
fn check_encoding<E: Encoding>(set: &UnitSet, values: &[u32], is: &dyn Fn(u32) -> bool) {
	let units = encode::<E>(values);
	let ctx = || format!("{:?} {values:x?}", core::any::type_name::<E>());
	let n = values.len();

	assert_eq!(set.find::<E>(&units), values.iter().position(|&v| is(v)), "find {}", ctx());
	assert_eq!(set.find_not::<E>(&units), values.iter().position(|&v| !is(v)), "find_not {}", ctx());
	assert_eq!(set.rfind::<E>(&units), values.iter().rposition(|&v| is(v)), "rfind {}", ctx());
	assert_eq!(
		set.rfind_not::<E>(&units),
		values.iter().rposition(|&v| !is(v)),
		"rfind_not {}",
		ctx()
	);
	assert_eq!(set.count::<E>(&units), values.iter().filter(|&&v| is(v)).count(), "count {}", ctx());

	let start = values.iter().position(|&v| !is(v)).unwrap_or(n);
	let end = values.iter().rposition(|&v| !is(v)).map_or(0, |i| i + 1);
	let span = |part: &[E::Unit]| (offset_of(&units, part), part.len());
	assert_eq!(span(set.trim_start::<E>(&units)), (start, n - start), "trim_start {}", ctx());
	assert_eq!(span(set.trim_end::<E>(&units)), (0, end), "trim_end {}", ctx());
	let trimmed = span(set.trim::<E>(&units));
	let expected = if start < end {
		(start, end - start)
	} else {
		(trimmed.0, 0)
	};
	assert_eq!(trimmed, expected, "trim {}", ctx());

	let fields: Vec<_> = set
		.fields::<E>(&units)
		.map(|(at, field)| {
			assert_eq!(offset_of(&units, field), at, "field slice offset {}", ctx());
			(at, field.len())
		})
		.collect();
	assert_eq!(fields, ref_fields(values, is), "fields {}", ctx());
}

/// Runs `check_encoding` over every encoding that can carry `values`.
fn check_all(set: &UnitSet, values: &[u32], is: &dyn Fn(u32) -> bool) {
	if values.iter().all(|&v| v < 0x100) {
		check_encoding::<Utf8>(set, values, is);
	}
	if values.iter().all(|&v| v < 0x1_0000) {
		check_encoding::<Utf16<false>>(set, values, is);
		check_encoding::<Utf16<true>>(set, values, is);
	}
	check_encoding::<Utf32<false>>(set, values, is);
	check_encoding::<Utf32<true>>(set, values, is);
}

/// Non-members that stress the saturating narrow, foreign swaps and the
/// zero padding: `0x0A00` swaps to `\n`, `0x100` saturates next to `0xFF`.
const WIDE_NOISE: &[u32] = &[0x100, 0x1ff, 0x0a00, 0x2000, 0x3000, 0xff00, 0xffff];
const HUGE_NOISE: &[u32] = &[0x10_ffff, 0x0a00_0000, 0xffff_ffff, 0x1_0020];

struct Case {
	set:     UnitSet,
	members: Vec<u8>,
}

fn cases() -> Vec<Case> {
	let range = |lo: u8, hi: u8| (lo..=hi).collect::<Vec<u8>>();
	let mut json = range(0, 0x1f);
	json.extend(b"\"\\");
	let mut high = range(0x80, 0x9f);
	high.extend([0xc3, 0xff, b'a']);
	vec![
		Case { set: UnitSet::ASCII_WHITESPACE, members: b"\t\n\x0b\x0c\r ".to_vec() },
		Case { set: UnitSet::new(b"<>&"), members: b"<>&".to_vec() },
		Case {
			set:     UnitSet::from_ranges(&[0..=0x1f, b'"'..=b'"', b'\\'..=b'\\']),
			members: json,
		},
		Case { set: UnitSet::new(&high), members: high.clone() },
		Case { set: UnitSet::from_ranges(&[0..=0xff]), members: range(0, 0xff) },
		Case { set: UnitSet::new(&[]), members: Vec::new() },
		Case {
			set:     UnitSet::new(b"\x00\x02\x04\x06\x08\x0a\x0c\xfe"),
			members: b"\x00\x02\x04\x06\x08\x0a\x0c\xfe".to_vec(),
		},
	]
}

/// Alphabet for `case`: every member plus byte neighbours and wide noise.
fn alphabet(case: &Case) -> Vec<u32> {
	let mut out: Vec<u32> = case.members.iter().map(|&m| u32::from(m)).collect();
	for &m in &case.members {
		out.push(u32::from(m.wrapping_add(1)));
		out.push(u32::from(m.wrapping_sub(1)));
	}
	out.extend([0, b'x'.into(), 0x7f, 0x80, 0xfe, 0xff]);
	out
}

#[test]
fn random_inputs_match_reference() {
	let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
	for case in cases() {
		let is = member_table(&case.members);
		let narrow = alphabet(&case);
		let mut wide = narrow.clone();
		wide.extend(WIDE_NOISE);
		let mut huge = wide.clone();
		huge.extend(HUGE_NOISE);
		for len in 0..=200 {
			for alpha in [&narrow, &wide, &huge] {
				for _ in 0..3 {
					let values: Vec<u32> = (0..len).map(|_| rng.pick(alpha)).collect();
					check_all(&case.set, &values, &is);
				}
			}
		}
		// Mostly-member and mostly-non-member mixes give long runs that cross
		// block boundaries.
		let (member, other) = (narrow[0], 0x3000);
		for len in [63, 64, 65, 127, 128, 129, 191, 192, 193, 300] {
			for bias in [2, 30] {
				let values: Vec<u32> = (0..len)
					.map(|_| {
						if rng.step() % 32 < bias {
							member
						} else {
							other
						}
					})
					.collect();
				check_all(&case.set, &values, &is);
			}
		}
	}
}

#[test]
fn single_hit_at_every_position() {
	for case in cases() {
		let is = member_table(&case.members);
		let Some(&member) = case.members.first() else {
			continue;
		};
		let Some(other) = (0..=0xffu32).find(|&v| !is(v)) else {
			continue;
		};
		let member = u32::from(member);
		for len in [1, 2, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200] {
			for at in 0..len {
				let mut values = vec![other; len];
				values[at] = member;
				check_all(&case.set, &values, &is);
				let mut values = vec![member; len];
				values[at] = other;
				check_all(&case.set, &values, &is);
				let mut values = vec![0x0a00; len];
				values[at] = member;
				check_all(&case.set, &values, &is);
			}
		}
	}
}

#[test]
fn constructors_merge_ranges() {
	assert_eq!(UnitSet::new(b"cab"), UnitSet::from_ranges(&[b'a'..=b'c']));
	let overlapping = UnitSet::from_ranges(&[b'a'..=b'f', b'c'..=b'k', b'l'..=b'm']);
	assert_eq!(overlapping, UnitSet::from_ranges(&[b'a'..=b'm']));
	#[allow(clippy::reversed_empty_ranges, reason = "empty ranges are ignored")]
	let with_empty = UnitSet::from_ranges(&[b'z'..=b'a', b'q'..=b'q']);
	assert_eq!(with_empty, UnitSet::new(b"q"));
	assert_eq!(UnitSet::new(b"\t\n\x0b\x0c\r "), UnitSet::ASCII_WHITESPACE);
	assert_eq!(UnitSet::new(&[]), UnitSet::from_ranges(&[]));
	// Eight disjoint ranges fit, built from many adjacent members.
	let members: Vec<u8> = (0..8u8)
		.flat_map(|r| [r * 32, r * 32 + 1, r * 32 + 2])
		.collect();
	let set = UnitSet::new(&members);
	for v in 0..0x400u32 {
		assert_eq!(set.contains(v), v < 256 && members.contains(&(v as u8)), "{v:#x}");
	}
	assert_eq!(UnitSet::MAX_RANGES, 8);
}

#[test]
#[should_panic(expected = "MAX_RANGES")]
fn nine_ranges_panic() {
	let members: Vec<u8> = (0..9u8).map(|r| r * 2).collect();
	let _ = UnitSet::new(&members);
}

#[test]
fn contains_and_is_ascii() {
	let ws = UnitSet::ASCII_WHITESPACE;
	for v in 0..0x300u32 {
		assert_eq!(ws.contains(v), matches!(v, 9..=13 | 32), "{v:#x}");
	}
	assert!(!ws.contains(u32::MAX));
	assert!(ws.is_ascii());
	assert!(UnitSet::new(&[]).is_ascii());
	assert!(UnitSet::new(&[0x7f]).is_ascii());
	assert!(!UnitSet::new(&[0x80]).is_ascii());
	assert!(!UnitSet::from_ranges(&[0..=0xff]).contains(0x100));
	assert!(UnitSet::from_ranges(&[0..=0xff]).contains(0xff));
}

#[test]
fn utf8_high_members_match_raw_bytes() {
	// 'é' is C3 A9 in UTF-8: a member 0xC3 matches its lead byte.
	let set = UnitSet::new(&[0xc3]);
	assert_eq!(set.find::<Utf8>("café".as_bytes()), Some(3));
	// In UTF-16 'é' is the single unit 0xE9; 0xC3 is 'Ã'.
	let wide: Vec<u16> = "caféÃ".encode_utf16().collect();
	assert_eq!(set.find::<Utf16>(&wide), Some(4));
}

const CHARS: &[char] =
	&[' ', '\t', '\n', '\x0b', '\r', 'a', 'Z', '-', 'é', '状', '🦀', '\u{3000}', '\u{85}', '\u{a0}'];

fn random_string(rng: &mut Rng, len: usize) -> String {
	(0..len).map(|_| rng.pick(CHARS)).collect()
}

#[test]
fn str_variants_match_reference() {
	let mut rng = Rng(0x2545_f491_4f6c_dd1d);
	for set in [UnitSet::ASCII_WHITESPACE, UnitSet::new(b"-a"), UnitSet::new(&[])] {
		let is = |c: char| (c as u32) < 0x80 && set.contains(c as u32);
		for len in 0..=120 {
			for _ in 0..4 {
				let s = random_string(&mut rng, len);
				assert_eq!(set.trim_str(&s), s.trim_matches(is), "{s:?}");
				let expected: Vec<(usize, &str)> = s
					.split(is)
					.filter(|f| !f.is_empty())
					.map(|f| (f.as_ptr() as usize - s.as_ptr() as usize, f))
					.collect();
				assert_eq!(set.fields_str(&s).collect::<Vec<_>>(), expected, "{s:?}");
			}
		}
	}
	let s = " \x0bfoo bar\x0c";
	let split: Vec<_> = UnitSet::ASCII_WHITESPACE
		.fields_str(s)
		.map(|(_, f)| f)
		.collect();
	assert_eq!(split, ["foo", "bar"]);
	assert_eq!(UnitSet::ASCII_WHITESPACE.trim_str(s), "foo bar");
}

#[test]
#[should_panic(expected = "ASCII-only")]
fn trim_str_rejects_non_ascii_set() {
	let _ = UnitSet::new(&[0xa0]).trim_str("x");
}

#[test]
#[should_panic(expected = "ASCII-only")]
fn fields_str_rejects_non_ascii_set() {
	let _ = UnitSet::new(&[0xa0]).fields_str("x");
}

#[test]
fn fields_iterator_is_clonable_and_fused() {
	let text: Vec<u8> = b"  alpha beta  gamma ".repeat(9);
	let set = UnitSet::ASCII_WHITESPACE;
	let mut fields = set.fields::<Utf8>(&text);
	let _ = fields.next();
	let rest: Vec<_> = fields.clone().collect();
	assert_eq!(rest.len(), 26);
	assert_eq!(fields.by_ref().collect::<Vec<_>>(), rest);
	assert_eq!(fields.next(), None);
	assert_eq!(fields.next(), None);
	let mut words = set.fields_str("a b");
	assert_eq!(words.by_ref().count(), 2);
	assert_eq!(words.next(), None);
}
