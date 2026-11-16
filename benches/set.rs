//! `UnitSet` scanning benchmark against scalar loops, `memchr` and `std`.
//!
//! Run default ratio bench: `cargo bench --bench set`
//! Run Divan bench: `DIVAN=1 cargo bench --bench set`

use std::hint::black_box;

use divan::{Bencher, counter::BytesCount};
use rayon::prelude::*;
use xutf::{UnitSet, Utf8, Utf16, Utf32};

mod common;
use common::{build_input, measure, print_ratio_table};

common::bench_main!();

const TARGET: usize = 1 << 20;

const WS: UnitSet = UnitSet::ASCII_WHITESPACE;
const MARKUP: UnitSet = UnitSet::new(b"<>&");
const JSON_ESCAPE: UnitSet = UnitSet::from_ranges(&[0..=0x1f, b'"'..=b'"', b'\\'..=b'\\']);

const LOG: &str = "2026-11-16T12:31:11.123Z INFO  [http] GET /api/v1/users?id=42&page=3 200 12ms \
                   ua=\"curl/8.4\" <cache:hit>\n";
const PROSE: &str = "It was the best of times, it was the worst of times, it was the age of \
                     wisdom, it was the age of foolishness; \"we had everything before us,\" & we \
                     had nothing before us.\n";
const CJK: &str = "状态更新：服务器在凌晨三点重启，日志显示一切正常。 \
                   用户反馈延迟降低了百分之三十 <见附件> & 详情。\n";
const JSON: &str = "{\"id\":42,\"name\":\"Ada \\\"the\\\" \
                    Countess\",\"tags\":[\"math\",\"poetry\"],\"bio\":\"line one\\nline two \\\\ \
                    ok\",\"score\":3.14}\n";

/// One benchmark input: a list of pieces scanned one call each (a single
/// 1 MiB piece, or many short tokens), in every encoding.
struct Input {
	name:   &'static str,
	pieces: Vec<String>,
	wide:   Vec<Vec<u16>>,
	wide32: Vec<Vec<u32>>,
	bytes:  usize,
}

impl Input {
	fn new(name: &'static str, pieces: Vec<String>) -> Self {
		let wide = pieces.iter().map(|p| p.encode_utf16().collect()).collect();
		let wide32 = pieces
			.iter()
			.map(|p| p.chars().map(u32::from).collect())
			.collect();
		let bytes = pieces.iter().map(String::len).sum();
		Self { name, pieces, wide, wide32, bytes }
	}

	fn long(name: &'static str, seed: &str) -> Self {
		Self::new(name, vec![build_input(seed, TARGET)])
	}

	/// ~1 MiB of `LOG` cut into 16..=40-byte pieces: per-token call overhead.
	fn short() -> Self {
		let text = build_input(LOG, TARGET);
		let mut pieces = Vec::new();
		let (mut at, mut k) = (0, 0);
		while at + 40 <= text.len() {
			let len = 16 + k % 25;
			pieces.push(text[at..at + len].to_owned());
			at += len;
			k += 7;
		}
		Self::new("short 16-40B", pieces)
	}
}

fn inputs() -> Vec<Input> {
	vec![
		Input::long("code", include_str!("../src/strip.rs")),
		Input::long("log", LOG),
		Input::long("prose", PROSE),
		Input::long("cjk", CJK),
		Input::long("json", JSON),
		Input::short(),
	]
}

/// Counts members of `set` by restarting `find` after each hit, the way a
/// tokenizer walks delimiters.
fn find_all(mut s: &[u8], mut find: impl FnMut(&[u8]) -> Option<usize>) -> usize {
	let mut hits = 0;
	while let Some(i) = find(s) {
		hits += 1;
		s = &s[i + 1..];
	}
	hits
}

fn find_all_wide(mut s: &[u16], set: &UnitSet) -> usize {
	let mut hits = 0;
	while let Some(i) = set.find::<Utf16>(s) {
		hits += 1;
		s = &s[i + 1..];
	}
	hits
}

fn sum<T>(pieces: &[T], f: impl Fn(&T) -> usize) -> usize {
	pieces.iter().map(f).sum()
}

fn find_row(input: &Input) -> Vec<f64> {
	let p = &input.pieces;
	let xutf = sum(p, |s| find_all(s.as_bytes(), |s| MARKUP.find::<Utf8>(s)));
	let scalar = |s: &String| {
		find_all(s.as_bytes(), |s| s.iter().position(|&b| MARKUP.contains(u32::from(b))))
	};
	let memchr3 = |s: &String| find_all(s.as_bytes(), |s| memchr::memchr3(b'<', b'>', b'&', s));
	assert_eq!(xutf, sum(p, scalar));
	assert_eq!(xutf, sum(p, memchr3));
	assert_eq!(xutf, sum(&input.wide, |s| find_all_wide(s, &MARKUP)));
	vec![
		measure(input.bytes, || sum(p, |s| find_all(s.as_bytes(), |s| MARKUP.find::<Utf8>(s)))),
		measure(input.bytes, || sum(p, memchr3)),
		measure(input.bytes, || sum(p, scalar)),
		measure(input.bytes, || sum(&input.wide, |s| find_all_wide(s, &MARKUP))),
	]
}

const fn is_json_escape(b: u8) -> bool {
	b < 0x20 || b == b'"' || b == b'\\'
}

fn json_row(input: &Input) -> Vec<f64> {
	let p = &input.pieces;
	let xutf = |s: &String| find_all(s.as_bytes(), |s| JSON_ESCAPE.find::<Utf8>(s));
	let scalar = |s: &String| find_all(s.as_bytes(), |s| s.iter().position(|&b| is_json_escape(b)));
	assert_eq!(sum(p, xutf), sum(p, scalar));
	vec![
		measure(input.bytes, || sum(p, xutf)),
		measure(input.bytes, || sum(p, scalar)),
		measure(input.bytes, || sum(&input.wide, |s| find_all_wide(s, &JSON_ESCAPE))),
	]
}

fn count_row(input: &Input) -> Vec<f64> {
	let p = &input.pieces;
	let scalar = |s: &String| {
		s.bytes()
			.filter(|&b| matches!(b, b'\t'..=b'\r' | b' '))
			.count()
	};
	assert_eq!(sum(p, |s| WS.count::<Utf8>(s.as_bytes())), sum(p, scalar));
	vec![
		measure(input.bytes, || sum(p, |s| WS.count::<Utf8>(s.as_bytes()))),
		measure(input.bytes, || sum(p, scalar)),
		measure(input.bytes, || sum(&input.wide, |s| WS.count::<Utf16>(s))),
		measure(input.bytes, || sum(&input.wide32, |s| WS.count::<Utf32>(s))),
	]
}

fn fields_row(input: &Input) -> Vec<f64> {
	let p = &input.pieces;
	let xutf = |s: &String| WS.fields_str(s).count();
	let reference = |s: &String| s.split_ascii_whitespace().count();
	assert_eq!(sum(p, xutf), sum(p, reference));
	vec![
		measure(input.bytes, || sum(p, xutf)),
		measure(input.bytes, || sum(p, reference)),
		measure(input.bytes, || sum(p, |s| WS.fields::<Utf8>(s.as_bytes()).count())),
		measure(input.bytes, || sum(&input.wide, |s| WS.fields::<Utf16>(s).count())),
	]
}

/// Trims every line of the input (or every short piece), summing the kept
/// lengths.
fn trim_row(input: &Input) -> Vec<f64> {
	let lines: Vec<&str> = input.pieces.iter().flat_map(|p| p.split('\n')).collect();
	let wide: Vec<Vec<u16>> = lines.iter().map(|l| l.encode_utf16().collect()).collect();
	let xutf = |l: &&str| WS.trim::<Utf8>(l.as_bytes()).len();
	let reference = |l: &&str| l.as_bytes().trim_ascii().len();
	assert_eq!(sum(&lines, xutf), sum(&lines, reference));
	vec![
		measure(input.bytes, || sum(&lines, xutf)),
		measure(input.bytes, || sum(&lines, reference)),
		measure(input.bytes, || sum(&lines, |l| WS.trim_str(l).len())),
		measure(input.bytes, || sum(&wide, |l| WS.trim::<Utf16>(l).len())),
	]
}

fn run_ratio_table() {
	let inputs = inputs();
	let table = |title: &str, impls: &[&str], row: fn(&Input) -> Vec<f64>| {
		let rows: Vec<_> = inputs
			.par_iter()
			.map(|input| (input.name, row(input)))
			.collect();
		print_ratio_table(title, impls, &rows);
	};
	table(
		"UnitSet::find, restart per hit, set <>&",
		&["xutf", "memchr3", "scalar", "xutf utf16"],
		find_row,
	);
	table("JSON escape scan (\" \\ < 0x20)", &["xutf", "scalar", "xutf utf16"], json_row);
	table(
		"UnitSet::count, ASCII whitespace",
		&["xutf", "scalar", "xutf utf16", "xutf utf32"],
		count_row,
	);
	table(
		"UnitSet::fields_str, ASCII whitespace",
		&["xutf str", "split_ascii_ws", "xutf bytes", "xutf utf16"],
		fields_row,
	);
	table(
		"UnitSet::trim per line, ASCII whitespace",
		&["xutf", "trim_ascii", "xutf str", "xutf utf16"],
		trim_row,
	);
}

fn divan_input() -> String {
	build_input(PROSE, TARGET)
}

#[divan::bench]
fn xutf_fields(bencher: Bencher) {
	let input = divan_input();
	bencher
		.counter(BytesCount::of_slice(input.as_bytes()))
		.bench_local(|| WS.fields_str(black_box(&input)).count());
}

#[divan::bench]
fn std_split_ascii_whitespace(bencher: Bencher) {
	let input = divan_input();
	bencher
		.counter(BytesCount::of_slice(input.as_bytes()))
		.bench_local(|| black_box(&input).split_ascii_whitespace().count());
}
