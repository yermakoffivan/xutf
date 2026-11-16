//! Line splitting and counting against `memchr`, `std`, and scalar loops.
//!
//! Run default ratio bench: `cargo bench --bench lines`
//! Run Divan bench: `DIVAN=1 cargo bench --bench lines`
//!
//! Every splitter sums `text.len() + 1` per line so each one materializes the
//! same spans.

use std::hint::black_box;

use divan::{Bencher, counter::BytesCount};
use rayon::prelude::*;
use xutf::{LineEnds, Utf16, lines, lines_str};

mod common;
use common::{build_input, measure, print_ratio_table};

common::bench_main!();

const TARGET: usize = 1 << 20;

const SOURCE: &str = "\
impl<'a, E: Encoding> Iterator for Lines<'a, E> {
	type Item = Line<'a, [E::Unit]>;

	#[inline]
	fn next(&mut self) -> Option<Self::Item> {
		// Pop the next candidate.
		let at = self.base + self.bits.trailing_zeros() as usize;
		self.bits &= self.bits - 1;
	}
}
";
const LOG: &str = "2026-11-16T12:31:11.402Z INFO  http::server request completed method=GET \
                   path=/api/v1/items status=200 elapsed=12ms\r\n";
const PROSE: &str = "It was the best of times, it was the worst of times, it was the age of \
                     wisdom, it was the age of foolishness, it was the epoch of belief, it was \
                     the epoch of incredulity, it was the season of Light, it was the season of \
                     Darkness, it was the spring of hope, it was the winter of despair, we had \
                     everything before us, we had nothing before us, we were all going direct to \
                     Heaven, we were all going direct the other way.\n";
const CJK: &str = "日本語のテキストを一行ずつ処理する。\n中文文本按行处理，速度很重要。\n";
const SHORT: &str = "42\n7\n1999\n\n-3\n";
/// One call per iteration on a short buffer: the per-call setup cost.
const CALL: &str = "let value = compute(x);\nnext";

fn inputs() -> Vec<(&'static str, String)> {
	vec![
		("source", build_input(SOURCE, TARGET)),
		("log crlf", build_input(LOG, TARGET)),
		("prose", build_input(PROSE, TARGET)),
		("cjk", build_input(CJK, TARGET)),
		("short lines", build_input(SHORT, TARGET)),
		("32 B call", CALL.to_owned()),
	]
}

fn xutf_spans(s: &str, ends: LineEnds) -> usize {
	lines_str(s, ends).map(|line| line.text().len() + 1).sum()
}

/// `memchr_iter` over LF, trimming one CR: the oh-my-pi `LineReader` loop.
fn memchr_lf(s: &[u8]) -> usize {
	let (mut total, mut start) = (0, 0);
	for nl in memchr::memchr_iter(b'\n', s) {
		let end = if nl > start && s[nl - 1] == b'\r' {
			nl - 1
		} else {
			nl
		};
		total += end - start + 1;
		start = nl + 1;
	}
	total
		+ if start < s.len() {
			s.len() - start + 1
		} else {
			0
		}
}

/// `memchr2_iter` over CR/LF, pairing CRLF.
fn memchr_universal(s: &[u8]) -> usize {
	let (mut total, mut start) = (0, 0);
	for at in memchr::memchr2_iter(b'\n', b'\r', s) {
		if at < start {
			continue; // LF of a CRLF already taken.
		}
		let next = if s[at] == b'\r' && s.get(at + 1) == Some(&b'\n') {
			at + 2
		} else {
			at + 1
		};
		total += at - start + 1;
		start = next;
	}
	total
		+ if start < s.len() {
			s.len() - start + 1
		} else {
			0
		}
}

/// Byte loop pairing CRLF: lpy's `bytes.splitlines`.
fn scalar_universal(s: &[u8]) -> usize {
	let (mut total, mut start, mut i) = (0, 0, 0);
	while i < s.len() {
		if matches!(s[i], b'\n' | b'\r') {
			let end = i;
			i += 1;
			if s[end] == b'\r' && s.get(i) == Some(&b'\n') {
				i += 1;
			}
			total += end - start + 1;
			start = i;
		} else {
			i += 1;
		}
	}
	total
		+ if start < s.len() {
			s.len() - start + 1
		} else {
			0
		}
}

/// Char loop over Python's `str.splitlines` set: lpy's `str.splitlines`.
fn scalar_unicode(s: &str) -> usize {
	let (mut total, mut start) = (0, 0);
	let mut chars = s.char_indices().peekable();
	while let Some((at, c)) = chars.next() {
		if matches!(
			c,
			'\n' | '\r' | '\x0b' | '\x0c' | '\x1c'..='\x1e' | '\u{85}' | '\u{2028}' | '\u{2029}'
		) {
			total += at - start + 1;
			start = if c == '\r' && chars.next_if(|&(_, n)| n == '\n').is_some() {
				at + 2
			} else {
				at + c.len_utf8()
			};
		}
	}
	total
		+ if start < s.len() {
			s.len() - start + 1
		} else {
			0
		}
}

/// Unit loop pairing CRLF over UTF-16: oh-my-pi's `line_tokens_u16`.
fn scalar_universal_u16(s: &[u16]) -> usize {
	let (mut total, mut start, mut i) = (0, 0, 0);
	while i < s.len() {
		if s[i] == 0x0a || s[i] == 0x0d {
			let end = i;
			i += 1;
			if s[end] == 0x0d && s.get(i) == Some(&0x0a) {
				i += 1;
			}
			total += end - start + 1;
			start = i;
		} else {
			i += 1;
		}
	}
	total
		+ if start < s.len() {
			s.len() - start + 1
		} else {
			0
		}
}

fn run_ratio_table() {
	let inputs = inputs();
	let table = |title: &str, impls: &[&str], row: &(dyn Fn(&str) -> Vec<f64> + Sync)| {
		let rows: Vec<_> = inputs
			.par_iter()
			.map(|(name, input)| (*name, row(input)))
			.collect();
		print_ratio_table(title, impls, &rows);
	};

	table("Lines, LF (CR trimmed)", &["xutf", "memchr_iter", "str::lines"], &|s| {
		let expected = memchr_lf(s.as_bytes());
		assert_eq!(xutf_spans(s, LineEnds::Lf), expected);
		vec![
			measure(s.len(), || xutf_spans(s, LineEnds::Lf)),
			measure(s.len(), || memchr_lf(black_box(s.as_bytes()))),
			measure(s.len(), || black_box(s).lines().map(|line| line.len() + 1).sum()),
		]
	});

	table("Lines, universal (LF, CRLF, CR)", &["xutf", "memchr2_iter", "byte loop"], &|s| {
		let expected = scalar_universal(s.as_bytes());
		assert_eq!(xutf_spans(s, LineEnds::Universal), expected);
		assert_eq!(memchr_universal(s.as_bytes()), expected);
		vec![
			measure(s.len(), || xutf_spans(s, LineEnds::Universal)),
			measure(s.len(), || memchr_universal(black_box(s.as_bytes()))),
			measure(s.len(), || scalar_universal(black_box(s.as_bytes()))),
		]
	});

	table("Lines, Unicode (Python str.splitlines)", &["xutf", "char loop"], &|s| {
		assert_eq!(xutf_spans(s, LineEnds::Unicode), scalar_unicode(s));
		vec![
			measure(s.len(), || xutf_spans(s, LineEnds::Unicode)),
			measure(s.len(), || scalar_unicode(black_box(s))),
		]
	});

	table("Lines, universal, UTF-16", &["xutf", "unit loop"], &|s| {
		let units: Vec<u16> = s.encode_utf16().collect();
		let xutf = |units: &[u16]| -> usize {
			lines::<Utf16>(units, LineEnds::Universal)
				.map(|line| line.text().len() + 1)
				.sum()
		};
		assert_eq!(xutf(&units), scalar_universal_u16(&units));
		vec![
			measure(s.len(), || xutf(black_box(&units))),
			measure(s.len(), || scalar_universal_u16(black_box(&units))),
		]
	});

	table("Line count, LF", &["xutf", "memchr_iter", "str::lines"], &|s| {
		let trailing = usize::from(!s.is_empty() && !s.ends_with('\n'));
		let expected = memchr::memchr_iter(b'\n', s.as_bytes()).count() + trailing;
		assert_eq!(lines_str(s, LineEnds::Lf).count(), expected);
		vec![
			measure(s.len(), || lines_str(black_box(s), LineEnds::Lf).count()),
			measure(s.len(), || {
				let s = black_box(s);
				memchr::memchr_iter(b'\n', s.as_bytes()).count()
					+ usize::from(!s.is_empty() && !s.ends_with('\n'))
			}),
			measure(s.len(), || black_box(s).lines().count()),
		]
	});
}

#[divan::bench]
fn xutf_lf(bencher: Bencher) {
	let input = build_input(SOURCE, TARGET);
	bencher
		.counter(BytesCount::of_slice(input.as_bytes()))
		.bench_local(|| xutf_spans(&input, LineEnds::Lf));
}

#[divan::bench]
fn memchr_iter_lf(bencher: Bencher) {
	let input = build_input(SOURCE, TARGET);
	bencher
		.counter(BytesCount::of_slice(input.as_bytes()))
		.bench_local(|| memchr_lf(input.as_bytes()));
}
