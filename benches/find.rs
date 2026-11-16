//! Substring search benchmark against `memchr::memmem`, `std` and naive
//! loops.
//!
//! Run default ratio bench: `cargo bench --bench find`
//! Run Divan bench: `DIVAN=1 cargo bench --bench find`

use std::hint::black_box;

use divan::{Bencher, counter::BytesCount};
use memchr::memmem;
use rayon::prelude::*;
use xutf::{Utf8, Utf16, find, find_ignore_ascii_case, find_iter, find_iter_ignore_ascii_case};

mod common;
use common::{build_input, measure, print_ratio_table};

common::bench_main!();

const TARGET: usize = 1 << 20;

const LOG: &str = "2026-11-16T12:31:11.123Z INFO  [http] GET /api/v1/users?id=42&page=3 200 12ms \
                   ua=\"curl/8.4\"\n2026-11-16T12:31:11.140Z WARN  [db] slow query: SELECT * FROM \
                   orders WHERE user_id = 42 (183ms)\n2026-11-16T12:31:11.162Z INFO  [http] POST \
                   /api/v1/orders 201 31ms\n2026-11-16T12:31:11.201Z ERROR [worker] job 7731 \
                   failed: connection reset by peer\n";
const PROSE: &str = "It was the best of times, it was the worst of times, it was the age of \
                     wisdom, it was the age of foolishness, it was the epoch of belief, it was \
                     the epoch of incredulity, it was the season of Light, it was the season of \
                     Darkness, it was the spring of hope, it was the winter of despair, we had \
                     everything before us, we had nothing before us.\n";
const CJK: &str =
	"状态更新：东京的服务器在凌晨三点重启，日志显示一切正常。用户反馈延迟降低了百分之三十。\
	 東京オフィスのチームは新しいリリースを準備しています。詳細は添付ファイルを参照してください。\n";

/// One row: pieces searched one call each (a single 1 MiB piece, or many
/// short lines), with the needle in every form the implementations need.
struct Case {
	name:   &'static str,
	pieces: Vec<String>,
	wide:   Vec<Vec<u16>>,
	needle: &'static str,
	bytes:  usize,
}

impl Case {
	fn new(name: &'static str, pieces: Vec<String>, needle: &'static str) -> Self {
		let wide = pieces.iter().map(|p| p.encode_utf16().collect()).collect();
		let bytes = pieces.iter().map(String::len).sum();
		Self { name, pieces, wide, needle, bytes }
	}

	fn long(name: &'static str, seed: &str, needle: &'static str) -> Self {
		Self::new(name, vec![build_input(seed, TARGET)], needle)
	}

	/// ~1 MiB of `LOG` cut into 16..=40-byte lines: per-call overhead.
	fn short(needle: &'static str) -> Self {
		let text = build_input(LOG, TARGET);
		let mut pieces = Vec::new();
		let (mut at, mut k) = (0, 0);
		while at + 40 <= text.len() {
			let mut len = 16 + k % 25;
			while !text.is_char_boundary(at + len) {
				len += 1;
			}
			pieces.push(text[at..at + len].to_owned());
			at += len;
			k += 7;
		}
		Self::new("short 16-40B", pieces, needle)
	}
}

const CODE: &str = include_str!("../src/strip.rs");

/// Exact-case rows: an identifier and a frequent token in source code, a
/// frequent and an absent word in prose, a log level and a CJK word.
fn exact_cases() -> Vec<Case> {
	vec![
		Case::long("code ident", CODE, "consume_ansi"),
		Case::long("code frequent", CODE, "let "),
		Case::long("prose word", PROSE, "the"),
		Case::long("prose absent", PROSE, "Zanzibar"),
		Case::long("log level", LOG, "ERROR"),
		Case::long("cjk word", CJK, "東京"),
		Case::short("GET"),
	]
}

/// Caseless rows: the same corpora with needles in a different case.
fn caseless_cases() -> Vec<Case> {
	vec![
		Case::long("code ident", CODE, "CONSUME_ANSI"),
		Case::long("code frequent", CODE, "LET "),
		Case::long("prose word", PROSE, "THE"),
		Case::long("prose absent", PROSE, "zanzibar"),
		Case::long("log level", LOG, "error"),
		Case::long("cjk word", CJK, "東京"),
		Case::short("get"),
	]
}

/// Sums `count` over the pieces. Long rows count every occurrence; short rows
/// ask whether each line contains the needle, the way a filter does.
fn total<T>(pieces: &[T], count: impl Fn(&T) -> usize) -> usize {
	pieces.iter().map(|p| count(black_box(p))).sum()
}

const fn is_short(case: &Case) -> bool {
	case.pieces.len() > 1
}

/// Non-overlapping occurrences found by restarting `windows().position()`.
fn naive_count_u16(mut hay: &[u16], needle: &[u16]) -> usize {
	let mut hits = 0;
	while let Some(p) = hay.windows(needle.len()).position(|w| w == needle) {
		hits += 1;
		hay = &hay[p + needle.len()..];
	}
	hits
}

fn utf8_rows(cases: &[Case]) -> Vec<(&'static str, Vec<f64>)> {
	cases
		.par_iter()
		.map(|case| {
			let n = case.needle;
			let short = is_short(case);
			let xutf = |p: &String| {
				if short {
					usize::from(find::<Utf8>(p.as_bytes(), n.as_bytes()).is_some())
				} else {
					find_iter::<Utf8>(p.as_bytes(), n.as_bytes()).count()
				}
			};
			let memchr = |p: &String| {
				if short {
					usize::from(memmem::find(p.as_bytes(), n.as_bytes()).is_some())
				} else {
					memmem::find_iter(p.as_bytes(), n.as_bytes()).count()
				}
			};
			let stdlib = |p: &String| {
				if short {
					usize::from(p.contains(n))
				} else {
					p.matches(n).count()
				}
			};
			let expected = total(&case.pieces, stdlib);
			assert_eq!(total(&case.pieces, xutf), expected, "{}", case.name);
			assert_eq!(total(&case.pieces, memchr), expected, "{}", case.name);
			(case.name, vec![
				measure(case.bytes, || total(&case.pieces, xutf)),
				measure(case.bytes, || total(&case.pieces, memchr)),
				measure(case.bytes, || total(&case.pieces, stdlib)),
			])
		})
		.collect()
}

fn utf16_rows(cases: &[Case]) -> Vec<(&'static str, Vec<f64>)> {
	cases
		.par_iter()
		.map(|case| {
			let n: Vec<u16> = case.needle.encode_utf16().collect();
			let short = is_short(case);
			let xutf = |p: &Vec<u16>| {
				if short {
					usize::from(find::<Utf16>(p, &n).is_some())
				} else {
					find_iter::<Utf16>(p, &n).count()
				}
			};
			let naive = |p: &Vec<u16>| {
				if short {
					usize::from(p.windows(n.len()).any(|w| w == n))
				} else {
					naive_count_u16(p, &n)
				}
			};
			let expected = total(&case.wide, naive);
			assert_eq!(total(&case.wide, xutf), expected, "{}", case.name);
			// Throughput in UTF-8 bytes of the same text, so rows compare
			// across tables.
			(case.name, vec![
				measure(case.bytes, || total(&case.wide, xutf)),
				measure(case.bytes, || total(&case.wide, naive)),
			])
		})
		.collect()
}

fn caseless_rows(cases: &[Case]) -> Vec<(&'static str, Vec<f64>)> {
	cases
		.par_iter()
		.map(|case| {
			let n = case.needle;
			let short = is_short(case);
			let xutf = |p: &String| {
				if short {
					usize::from(find_ignore_ascii_case::<Utf8>(p.as_bytes(), n.as_bytes()).is_some())
				} else {
					find_iter_ignore_ascii_case::<Utf8>(p.as_bytes(), n.as_bytes()).count()
				}
			};
			// What callers write without a caseless search: lowercase both
			// sides, then search the copy.
			let lowered = |p: &String| {
				let hay = p.to_ascii_lowercase();
				let needle = n.to_ascii_lowercase();
				if short {
					usize::from(memmem::find(hay.as_bytes(), needle.as_bytes()).is_some())
				} else {
					memmem::find_iter(hay.as_bytes(), needle.as_bytes()).count()
				}
			};
			let expected = total(&case.pieces, lowered);
			assert_eq!(total(&case.pieces, xutf), expected, "{}", case.name);
			(case.name, vec![
				measure(case.bytes, || total(&case.pieces, xutf)),
				measure(case.bytes, || total(&case.pieces, lowered)),
			])
		})
		.collect()
}

fn run_ratio_table() {
	let exact = exact_cases();
	print_ratio_table(
		"Substring search, UTF-8 (count all; short: contains)",
		&["xutf find_iter", "memchr memmem", "std str"],
		&utf8_rows(&exact),
	);
	print_ratio_table(
		"Substring search, UTF-16 (count all; short: contains)",
		&["xutf find_iter", "naive windows"],
		&utf16_rows(&exact),
	);
	print_ratio_table(
		"ASCII-caseless search, UTF-8 (count all; short: contains)",
		&["xutf ignore_case", "lowercase+memmem"],
		&caseless_rows(&caseless_cases()),
	);
}

fn divan_input() -> String {
	build_input(CODE, TARGET)
}

#[divan::bench]
fn xutf_find_iter(bencher: Bencher) {
	let input = divan_input();
	bencher
		.counter(BytesCount::of_slice(input.as_bytes()))
		.bench_local(|| find_iter::<Utf8>(input.as_bytes(), b"consume_ansi").count());
}

#[divan::bench]
fn memchr_find_iter(bencher: Bencher) {
	let input = divan_input();
	bencher
		.counter(BytesCount::of_slice(input.as_bytes()))
		.bench_local(|| memmem::find_iter(input.as_bytes(), b"consume_ansi").count());
}
