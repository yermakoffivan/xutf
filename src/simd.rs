//! SIMD ASCII fast-path kernels (`utf_ascii_enum` / `utf_ascii_cmp`).
//!
//! Both kernels use the overlapped-tail trick from the original: when fewer
//! than `N` units remain before `limit`, the window is moved back to
//! `limit - N` and the overlap is re-processed (loads are idempotent, stores
//! rewrite identical values).

use core::{
	mem::size_of,
	simd::{
		Mask, MaskElement, Simd, SimdElement,
		cmp::{SimdPartialEq, SimdPartialOrd},
	},
};

use crate::unit::Unit;

/// Wide transcode lane count: four NEON registers on ARM and two
/// register-resident x86 widening groups on Sapphire Rapids.
#[cfg(target_arch = "aarch64")]
pub const TRANS_WIDE: usize = 64;
#[cfg(target_arch = "x86_64")]
pub(crate) const TRANS_WIDE: usize = 64;
#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub(crate) const TRANS_WIDE: usize = 16;
pub const CMP_WIDE: usize = 16;
pub const NARROW: usize = 8;

/// Units per block for the bitmask scanners ([`crate::lines`],
/// [`crate::UnitSet`], [`crate::find`]): one `u64` holds a block's matches.
pub const BLOCK: usize = 64;

/// Mask of the low `n` bits, `n <= 64`.
#[inline(always)]
pub const fn low_bits(n: usize) -> u64 {
	if n >= BLOCK { !0 } else { (1 << n) - 1 }
}

/// Bit `i` set when `pred` holds for lane `i` of `block`.
///
/// `pred` sees one 16-lane quarter at a time. `AArch64` has no movemask: a
/// 64-lane mask packed to bits makes LLVM shuffle a 64-lane `i1` vector, which
/// it scalarizes lane by lane, while 16-lane masks stay one compare per
/// register and pack with the `vpaddq` reduction of
/// [`crate::kernel::bitmask64`]. Elsewhere each quarter is a `to_bitmask`
/// (`pmovmskb` on x86).
#[inline(always)]
pub fn bits64<U: SimdElement, M: MaskElement>(
	block: Simd<U, BLOCK>,
	pred: impl Fn(Simd<U, 16>) -> Mask<M, 16>,
) -> u64 {
	let lanes = block.as_array();
	let quarter = |k: usize| pred(Simd::from_slice(&lanes[k * 16..]));
	#[cfg(target_arch = "aarch64")]
	{
		use core::arch::aarch64::uint8x16_t;

		let bytes = |k: usize| {
			// SAFETY: `Simd<i8, 16>` and `uint8x16_t` are both 16 plain bytes.
			unsafe {
				core::mem::transmute::<Simd<i8, 16>, uint8x16_t>(quarter(k).cast::<i8>().to_simd())
			}
		};
		crate::kernel::bitmask64(bytes(0), bytes(1), bytes(2), bytes(3))
	}
	#[cfg(not(target_arch = "aarch64"))]
	{
		(0..4).fold(0, |bits, k| bits | quarter(k).to_bitmask() << (k * 16))
	}
}

/// [`bits64`] for 16 lanes: bit `i` set when `pred` holds for lane `i`.
#[inline(always)]
fn bits16<U: SimdElement, M: MaskElement>(
	v: Simd<U, 16>,
	pred: &impl Fn(Simd<U, 16>) -> Mask<M, 16>,
) -> u64 {
	#[cfg(target_arch = "aarch64")]
	{
		use core::arch::aarch64::{uint8x16_t, vdupq_n_u8};

		// SAFETY: `Simd<i8, 16>` and `uint8x16_t` are both 16 plain bytes;
		// AArch64 mandates NEON.
		unsafe {
			let bytes =
				core::mem::transmute::<Simd<i8, 16>, uint8x16_t>(pred(v).cast::<i8>().to_simd());
			let zero = vdupq_n_u8(0);
			crate::kernel::bitmask64(bytes, zero, zero, zero)
		}
	}
	#[cfg(not(target_arch = "aarch64"))]
	{
		pred(v).to_bitmask()
	}
}

/// Up to [`BLOCK`] units that a scanner classifies into one `u64`, bit `i`
/// for unit `i`; built by [`window`].
#[derive(Clone, Copy)]
pub enum Window<'a, U: SimdElement> {
	/// A whole block.
	Block(Simd<U, BLOCK>),
	/// Fewer than [`BLOCK`] units, read 16 lanes at a time with the last read
	/// overlapping, so neither a padded copy nor an out-of-bounds load is
	/// needed; bits at or past the end are clear.
	Short(&'a [U]),
}

impl<U: SimdElement + Default> Window<'_, U> {
	/// Bit `i` set when `pred` holds for unit `i`, as [`bits64`] computes it.
	#[inline(always)]
	pub fn bits<M: MaskElement>(&self, pred: impl Fn(Simd<U, 16>) -> Mask<M, 16>) -> u64 {
		match *self {
			Self::Block(block) => bits64(block, pred),
			Self::Short(units) => short_bits(units, pred),
		}
	}

	/// Whether `pred` holds for any unit; a block ORs its quarters and reduces
	/// once instead of packing bits.
	#[inline(always)]
	pub fn any<M: MaskElement>(&self, pred: impl Fn(Simd<U, 16>) -> Mask<M, 16>) -> bool {
		match *self {
			Self::Block(block) => {
				let lanes = block.as_array();
				(1..4)
					.fold(pred(Simd::from_slice(lanes)), |hit, k| {
						hit | pred(Simd::from_slice(&lanes[k * 16..]))
					})
					.any()
			},
			Self::Short(_) => self.bits(pred) != 0,
		}
	}
}

/// [`Window::bits`] of a short window, out of line: it runs at most once per
/// input (the tail), and inlined it would bloat every block loop.
#[inline(never)]
fn short_bits<U: SimdElement + Default, M: MaskElement>(
	units: &[U],
	pred: impl Fn(Simd<U, 16>) -> Mask<M, 16>,
) -> u64 {
	let n = units.len();
	if n < 16 {
		let mut padded = [U::default(); 16];
		padded[..n].copy_from_slice(units);
		return bits16(Simd::from_array(padded), &pred) & low_bits(n);
	}
	let mut bits = 0;
	let mut at = 0;
	while at + 16 <= n {
		bits |= bits16(Simd::from_slice(&units[at..]), &pred) << at;
		at += 16;
	}
	if at < n {
		// Overlapping lanes repeat bits already set.
		bits |= bits16(Simd::from_slice(&units[n - 16..]), &pred) << (n - 16);
	}
	bits
}

/// The window holding `data[at..]` (at most [`BLOCK`] units of it), and the
/// offset of the window's first unit.
///
/// A tail shorter than a block is classified as the last whole block when
/// one exists, so the caller shifts the bits right by `at - origin`;
/// otherwise the window is [`Window::Short`] and `origin == at`. Requires
/// `at < data.len()`.
#[inline(always)]
pub fn window<U: SimdElement + Default>(data: &[U], at: usize) -> (Window<'_, U>, usize) {
	let len = data.len();
	debug_assert!(at < len, "window past the end");
	if len - at >= BLOCK {
		(Window::Block(Simd::from_slice(&data[at..])), at)
	} else if len >= BLOCK {
		(Window::Block(Simd::from_slice(&data[len - BLOCK..])), len - BLOCK)
	} else {
		(Window::Short(&data[at..]), at)
	}
}

/// Streams `N`-lane windows of `data[..limit]` through `f(window, offset)`
/// while they are all-ASCII, returning the length of the leading ASCII run.
///
/// `f` is also invoked on the window containing the first non-ASCII unit
/// (its clean prefix is live data; anything past the returned count is
/// scratch the caller must overwrite or discard).
#[inline(always)]
pub fn ascii_run<U: Unit, const N: usize>(
	data: &[U],
	limit: usize,
	mut f: impl FnMut(Simd<U, N>, usize),
) -> usize {
	debug_assert!(N <= limit && limit <= data.len());
	let mut i = 0;
	loop {
		let v = Simd::<U, N>::from_slice(&data[i..]);
		let bad = U::non_ascii_index(v);
		f(v, i);
		if let Some(p) = bad {
			return i + p;
		}
		i += N;
		if i + N > limit {
			if i == limit {
				return i;
			}
			i = limit - N;
		}
	}
}

/// Raw unit-slice equality tuned for long inputs: 256-byte blocks fold four
/// XOR/OR trees into a single reduction each, so the equal-buffer sweep
/// carries one branch per 256 bytes instead of a libc `memcmp` call's
/// per-chunk exits.
#[inline]
pub fn raw_eq<U: Unit>(left: &[U], right: &[U]) -> bool {
	if left.len() != right.len() {
		return false;
	}
	// SAFETY: unit types are plain integers; the byte view covers the same
	// memory with length scaled by the unit width.
	let left = unsafe { core::slice::from_raw_parts(left.as_ptr().cast::<u8>(), size_of_val(left)) };
	// SAFETY: as above.
	let right =
		unsafe { core::slice::from_raw_parts(right.as_ptr().cast::<u8>(), size_of_val(right)) };
	#[cfg(target_arch = "x86_64")]
	{
		// LLVM lowers equal byte slices to the platform `bcmp`/`memcmp`
		// implementation. That routine is runtime-multiversioned; a fixed
		// portable-SIMD loop is limited to the crate's baseline target and
		// loses roughly a third of throughput on AVX2/AVX-512 machines.
		left == right
	}
	#[cfg(not(target_arch = "x86_64"))]
	{
		let byte_len = left.len();
		let mut offset = 0;
		while byte_len - offset >= 256 {
			let difference = ((Simd::<u8, 64>::from_slice(&left[offset..])
				^ Simd::from_slice(&right[offset..]))
				| (Simd::<u8, 64>::from_slice(&left[offset + 64..])
					^ Simd::from_slice(&right[offset + 64..])))
				| ((Simd::<u8, 64>::from_slice(&left[offset + 128..])
					^ Simd::from_slice(&right[offset + 128..]))
					| (Simd::<u8, 64>::from_slice(&left[offset + 192..])
						^ Simd::from_slice(&right[offset + 192..])));
			if difference != Simd::splat(0) {
				return false;
			}
			offset += 256;
		}
		while byte_len - offset >= 128 {
			let difference = (Simd::<u8, 64>::from_slice(&left[offset..])
				^ Simd::from_slice(&right[offset..]))
				| (Simd::<u8, 64>::from_slice(&left[offset + 64..])
					^ Simd::from_slice(&right[offset + 64..]));
			if difference != Simd::splat(0) {
				return false;
			}
			offset += 128;
		}
		while byte_len - offset >= 16 {
			let difference =
				Simd::<u8, 16>::from_slice(&left[offset..]) ^ Simd::from_slice(&right[offset..]);
			if difference != Simd::splat(0) {
				return false;
			}
			offset += 16;
		}
		if offset == byte_len {
			return true;
		}
		if byte_len >= 16 {
			// Overlapped tail window; loads are idempotent.
			let difference = Simd::<u8, 16>::from_slice(&left[byte_len - 16..])
				^ Simd::from_slice(&right[byte_len - 16..]);
			return difference == Simd::splat(0);
		}
		left[offset..] == right[offset..]
	}
}
/// Tests raw code units for equality after folding ASCII uppercase units.
///
/// This is a complete equality shortcut, not an ASCII-prefix scan: non-ASCII
/// units are left unchanged. It is therefore sound for native same-encoding
/// strings, where an equal folded unit sequence necessarily decodes equally.
#[inline(always)]
pub fn raw_eq_ignore_ascii_case<U: Unit, const N: usize>(a: &[U], b: &[U]) -> bool {
	if a.len() != b.len() {
		return false;
	}
	if a.len() < N {
		return a.iter().zip(b).all(|(&x, &y)| {
			let x = x.to_u32();
			let y = y.to_u32();
			let fx = if x.wrapping_sub('A' as u32) <= 25 {
				x ^ 0x20
			} else {
				x
			};
			let fy = if y.wrapping_sub('A' as u32) <= 25 {
				y ^ 0x20
			} else {
				y
			};
			fx == fy
		});
	}

	let mut i = 0;
	loop {
		let va = U::fold_case(Simd::<U, N>::from_slice(&a[i..]), 'A' as u32);
		let vb = U::fold_case(Simd::<U, N>::from_slice(&b[i..]), 'A' as u32);
		if va != vb {
			return false;
		}
		i += N;
		if i + N > a.len() {
			if i == a.len() {
				return true;
			}
			i = a.len() - N;
		}
	}
}

/// Outcome of one [`ascii_cmp`] sweep.
pub enum CmpStep {
	/// Hit a non-ASCII unit; resume scalar at the advanced position.
	NonAscii,
	/// Both sides equal through `limit`.
	Equal,
	/// Mismatch: folded difference (or `1` when only equality was asked).
	Diff(i64),
}
/// Equality-only ASCII sweep specialized for native UTF-8 versus UTF-16.
///
/// Keeping the comparison in the `u16` domain avoids widening both operands
/// to `u32`, as the ordering-capable generic kernel must do.
#[inline(always)]
pub fn ascii_eq_u8_u16<const N: usize>(
	a: &[u8],
	b: &[u16],
	limit: usize,
	caseless: bool,
	advanced: &mut usize,
) -> CmpStep {
	debug_assert!(N <= limit && limit <= a.len() && limit <= b.len());
	let mut i = 0;
	let step = loop {
		let mut va = Simd::<u8, N>::from_slice(&a[i..]);
		let mut vb = Simd::<u16, N>::from_slice(&b[i..]);
		if u8::non_ascii_index(va).is_some() || u16::non_ascii_index(vb).is_some() {
			break CmpStep::NonAscii;
		}
		if caseless {
			va = u8::fold_case(va, 'A' as u32);
			vb = u16::fold_case(vb, 'A' as u32);
		}
		if u8::cast::<u16, N>(va).simd_ne(vb).any() {
			break CmpStep::Diff(1);
		}
		i += N;
		if i + N > limit {
			if i == limit {
				break CmpStep::Equal;
			}
			i = limit - N;
		}
	};
	*advanced = i;
	step
}

/// Compares `a[..limit]` to `b[..limit]` while both are ASCII, in `N`-lane
/// windows widened to `u32`. Writes the number of confirmed-equal leading
/// units to `advanced`.
#[inline(always)]
pub fn ascii_cmp<A: Unit, B: Unit, const N: usize>(
	a: &[A],
	b: &[B],
	limit: usize,
	caseless: bool,
	for_eq: bool,
	advanced: &mut usize,
) -> CmpStep {
	debug_assert!(N <= limit && limit <= a.len() && limit <= b.len());
	let mut i = 0;
	let step = loop {
		let va = A::widen(Simd::<A, N>::from_slice(&a[i..]));
		let vb = B::widen(Simd::<B, N>::from_slice(&b[i..]));

		// Unlike the C++ (which only checked the narrow side and could
		// false-match truncated wide units), require both sides ASCII.
		if (va | vb).simd_gt(Simd::splat(0x7f)).any() {
			break CmpStep::NonAscii;
		}

		let (fa, fb) = if caseless {
			(u32::fold_case(va, 'A' as u32), u32::fold_case(vb, 'A' as u32))
		} else {
			(va, vb)
		};

		let ne = fa.simd_ne(fb);
		if ne.any() {
			if for_eq {
				break CmpStep::Diff(1);
			}
			let p = ne.first_set().unwrap();
			break CmpStep::Diff(fa[p] as i64 - fb[p] as i64);
		}

		i += N;
		if i + N > limit {
			if i == limit {
				break CmpStep::Equal;
			}
			i = limit - N;
		}
	};
	*advanced = i;
	step
}

/// Length of the longest prefix of `data` whose units are all printable
/// single-cell ASCII (`0x20..=0x7E`).
#[inline]
pub fn plain_prefix<U: Unit>(data: &[U]) -> usize {
	// Scalar reject: cluster-scanning loops re-enter here once per cluster on
	// non-ASCII text; a head-unit check skips the vector setup entirely.
	if data
		.first()
		.is_none_or(|u| !(0x20..=0x7e).contains(&u.to_u32()))
	{
		return 0;
	}
	if size_of::<U>() == 1 {
		// A single-register probe catches the short ASCII spans in mixed text
		// without putting the long-run loop on a narrower stride. A clean
		// probe is deliberately re-read by the throughput path below.
		if data.len() >= CMP_WIDE {
			let v = Simd::<U, CMP_WIDE>::from_slice(data);
			if let Some(p) = U::non_plain_index(v) {
				return p;
			}
		}
		plain_prefix_n::<U, TRANS_WIDE>(data)
	} else {
		plain_prefix_n::<U, 16>(data)
	}
}

#[inline(always)]
fn plain_prefix_n<U: Unit, const N: usize>(data: &[U]) -> usize {
	let mut i = 0;
	while i + N <= data.len() {
		let v = Simd::<U, N>::from_slice(&data[i..i + N]);
		if let Some(p) = U::non_plain_index(v) {
			return i + p;
		}
		i += N;
	}
	// What is left a narrow window at a time, the last window overlapping
	// units already found plain.
	while i + CMP_WIDE <= data.len() {
		let v = Simd::<U, CMP_WIDE>::from_slice(&data[i..i + CMP_WIDE]);
		if let Some(p) = U::non_plain_index(v) {
			return i + p;
		}
		i += CMP_WIDE;
	}
	if i < data.len()
		&& let Some(start) = data.len().checked_sub(CMP_WIDE)
	{
		let v = Simd::<U, CMP_WIDE>::from_slice(&data[start..]);
		return U::non_plain_index(v).map_or(data.len(), |p| start + p);
	}
	while i < data.len() && (0x20..=0x7e).contains(&data[i].to_u32()) {
		i += 1;
	}
	i
}
