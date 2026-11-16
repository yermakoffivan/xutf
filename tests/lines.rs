use xutf::{Encoding, LineEnds, Utf8, Utf16, Utf32, lines, lines_str};

const MODES: [LineEnds; 3] = [LineEnds::Lf, LineEnds::Universal, LineEnds::Unicode];

struct Rng(u64);

impl Rng {
	fn next(&mut self) -> u64 {
		self.0 ^= self.0 << 13;
		self.0 ^= self.0 >> 7;
		self.0 ^= self.0 << 17;
		self.0
	}

	fn pick<T: Copy>(&mut self, items: &[T]) -> T {
		items[self.next() as usize % items.len()]
	}
}

/// `(text_start, text_end, next_start)` per line, scanning native unit values
/// one at a time; `utf8` selects the multi-byte NEL/LS/PS forms.
fn reference(values: &[u32], utf8: bool, ends: LineEnds) -> Vec<(usize, usize, usize)> {
	let unicode = ends == LineEnds::Unicode;
	let mut out = Vec::new();
	let (mut start, mut i) = (0, 0);
	while i < values.len() {
		let next = |k: usize| values.get(i + k).copied();
		let terminator = match values[i] {
			0x0a => 1,
			0x0d if next(1) == Some(0x0a) => 2,
			0x0d if ends != LineEnds::Lf => 1,
			0x0b | 0x0c | 0x1c..=0x1e if unicode => 1,
			0x85 | 0x2028 | 0x2029 if unicode && !utf8 => 1,
			0xc2 if unicode && utf8 && next(1) == Some(0x85) => 2,
			0xe2
				if unicode && utf8 && next(1) == Some(0x80) && matches!(next(2), Some(0xa8 | 0xa9)) =>
			{
				3
			},
			_ => 0,
		};
		if terminator == 0 {
			i += 1;
		} else {
			out.push((start, i, i + terminator));
			i += terminator;
			start = i;
		}
	}
	if start < values.len() {
		out.push((start, values.len(), values.len()));
	}
	out
}

fn check<E: Encoding>(units: &[E::Unit], values: &[u32], ends: LineEnds) {
	let expected = reference(values, size_of::<E::Unit>() == 1, ends);
	let actual: Vec<_> = lines::<E>(units, ends)
		.map(|line| {
			let at = line.at();
			assert_eq!(line.text().len() + line.end().len(), line.with_end().len());
			(at, at + line.text().len(), at + line.with_end().len())
		})
		.collect();
	assert_eq!(actual, expected, "{ends:?} over {values:x?}");
	assert_eq!(lines::<E>(units, ends).count(), expected.len(), "{ends:?} count over {values:x?}");
	// `count` resumes mid-stream, including mid-block.
	for skip in [1, expected.len() / 2, expected.len().saturating_sub(1)] {
		let mut it = lines::<E>(units, ends);
		it.by_ref().take(skip).for_each(drop);
		assert_eq!(it.count(), expected.len().saturating_sub(skip), "{ends:?} skip {skip}");
	}
}

/// Raw units drawn from line ends, their byte-swapped forms, and the bytes of
/// near-miss sequences (`C2 A8`, `E2 80 A6`, `C5 85`) that must not match.
#[test]
fn matches_reference_on_dense_raw_units() {
	let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
	let bytes = [
		0x0a, 0x0d, 0x0b, 0x0c, 0x1c, 0x1e, 0x1f, 0x09, 0xc2, 0x85, 0xe2, 0x80, 0xa8, 0xa9, 0xa6,
		0xc5, b'a', 0,
	];
	let wide =
		[0x0a, 0x0d, 0x0b, 0x1d, 0x85, 0x2028, 0x2029, 0x202a, 0x0a00, 0x0d00, 0xd800, 0x61, 0];
	for _ in 0..4000 {
		let len = rng.next() as usize % 300;
		let raw8: Vec<u8> = (0..len).map(|_| rng.pick(&bytes)).collect();
		let values8: Vec<u32> = raw8.iter().map(|&b| u32::from(b)).collect();
		let raw16: Vec<u16> = (0..len).map(|_| rng.pick(&wide) as u16).collect();
		let values16: Vec<u32> = raw16.iter().map(|&u| u32::from(u)).collect();
		let swapped16: Vec<u32> = raw16.iter().map(|&u| u32::from(u.swap_bytes())).collect();
		let raw32: Vec<u32> = (0..len).map(|_| rng.pick(&wide)).collect();
		let swapped32: Vec<u32> = raw32.iter().map(|&u| u.swap_bytes()).collect();
		for ends in MODES {
			check::<Utf8>(&raw8, &values8, ends);
			check::<Utf16<false>>(&raw16, &values16, ends);
			check::<Utf16<true>>(&raw16, &swapped16, ends);
			check::<Utf32<false>>(&raw32, &raw32, ends);
			check::<Utf32<true>>(&raw32, &swapped32, ends);
		}
	}
}

/// Every encoding of the same text yields the same lines, and `Lf` agrees
/// with `str::lines`.
#[test]
fn encodings_agree_on_valid_text() {
	let mut rng = Rng(0x2545_f491_4f6c_dd1d);
	let alphabet = [
		'a', 'Z', ' ', '\n', '\r', '\x0b', '\x0c', '\x1c', '\x1e', '\u{85}', '\u{a8}', '\u{145}',
		'\u{2026}', '\u{2028}', '\u{2029}', '\u{20a8}', '中', '🦀',
	];
	for _ in 0..3000 {
		let len = rng.next() as usize % 160;
		let text: String = (0..len).map(|_| rng.pick(&alphabet)).collect();
		for ends in MODES {
			let expected: Vec<(&str, &str)> = lines_str(&text, ends)
				.map(|line| (line.text(), line.end()))
				.collect();
			let utf16: Vec<u16> = text.encode_utf16().collect();
			let from16: Vec<(String, String)> = lines::<Utf16>(&utf16, ends)
				.map(|line| {
					(String::from_utf16(line.text()).unwrap(), String::from_utf16(line.end()).unwrap())
				})
				.collect();
			let utf32: Vec<u32> = text.chars().map(u32::from).collect();
			let from32: Vec<(String, String)> = lines::<Utf32>(&utf32, ends)
				.map(|line| {
					let decode =
						|units: &[u32]| units.iter().map(|&u| char::from_u32(u).unwrap()).collect();
					(decode(line.text()), decode(line.end()))
				})
				.collect();
			let expected_owned: Vec<(String, String)> = expected
				.iter()
				.map(|&(t, e)| (t.to_owned(), e.to_owned()))
				.collect();
			assert_eq!(from16, expected_owned, "{ends:?} {text:?}");
			assert_eq!(from32, expected_owned, "{ends:?} {text:?}");
		}
		let std_lines: Vec<&str> = text.lines().collect();
		let ours: Vec<&str> = lines_str(&text, LineEnds::Lf)
			.map(|line| line.text())
			.collect();
		assert_eq!(ours, std_lines, "{text:?}");
	}
}

#[test]
fn final_terminator_starts_no_line() {
	for ends in MODES {
		assert_eq!(lines_str("", ends).count(), 0);
		assert_eq!(
			lines_str("\n", ends)
				.map(|line| line.text())
				.collect::<Vec<_>>(),
			[""]
		);
		let tail = lines_str("a\nb", ends).last().unwrap();
		assert_eq!((tail.text(), tail.end(), tail.at()), ("b", "", 2));
	}
}
