# Offline replay of the live contact estimator

`E_BTS_contact_replay` re-runs the GUI's contact-location estimator over a `.raw`
that has already been recorded, and writes a `contact.csv` in the same format the
GUI writes live.

## Why it exists

The ladder run of 2026-09-06 (`recordings/ladder_20260906_150159`,
`docs/contact_accuracy_ladder_20260906_150159.pdf`) measured a positional **gain**
of 0.59 in x and 0.69 in y: move the indenter 1 mm and the reported contact moved
about 0.6 mm. The cause was found — `CircleDetector::detect_from_map()` searched a
±9 px square around each marker's **rest** position, while a marker beside a 5 mm
indent moves 15–25 px, so the markers carrying the contact signal sat outside their
own search window. The fix (`TemporalCircleTracker::circle_map_search_centers()`)
follows the marker instead.

That left one question worth 2.8 h of robot time: **did it work?**

Re-running the campaign answers it, but on a *different* random sample of poke
locations, so an improved gain there is confounded with having drawn easier pokes.
Replaying the recording answers it on the **same** pokes — same locations, same
depths, same baseline, the same 211 342 event windows. The only thing that changes
is the code, so the difference is attributable to the fix and to nothing else.

It also answers something the robot cannot. `kContactMinimumDivergence` is a
starting value, not a calibrated one, and 17 % of that run's pokes produced no
estimate at all. You cannot re-poke at a different threshold; you *can* re-threshold
a recording. See **Sweeping the threshold** below.

## Running it

```sh
./build/E_BTS_contact_replay \
    --raw      recordings/<run>/camera.raw \
    --time-ref recordings/<run>/contact.csv \
    --out      recordings/<run>/contact_replay.csv \
    --accum-us 40000

python3 ml/contact_error.py recordings/<run> --contact contact_replay.csv
```

Run it from the repo root so `calibration/pixel_to_mm.json` resolves, or pass
`--calib`. It reads the `.raw` at roughly 8–19× real time (a 2.3 h run takes about
10–20 min) and uses one core.

`--limit-s 60` processes only the first minute — worth doing first, since it
reaches the baseline and the first few pokes in a few seconds.

### The flags that matter

| flag | why |
| --- | --- |
| `--accum-us` | **Must match the run.** The ladder run used 40000; the live *default* is 10000, so this is not a value to leave alone. Window length changes the estimate. |
| `--time-ref` | Recovers the wall clock. Without it the output cannot be joined to `franka.csv` at all — see below. |
| `--min-divergence` | The accept threshold. Peak strength is recorded for every window regardless, so one pass supports any threshold afterwards. |

## What is and is not identical to the live run

**The estimator is the same code.** `CircleTrackingSource::process_window()` is
what the GUI calls. The replay differs only in taking every window in order
(`pop_oldest`) rather than the most recent (`pop_latest`), and in not rendering a
frame nobody will look at. This class acquired its gain defect from a divergence
between what the tracker was told and what it searched, so a *second* copy of the
estimator loop — one for the robot, one for the analysis — is exactly the thing not
to add.

**The baseline is rebuilt, not copied.** It comes from the head of the recording,
exactly as it was built live; the campaign starts with the pad unloaded, which is
the condition a baseline needs.

**Window boundaries differ, and this is the one real caveat.** Each 40 ms window
grid is anchored to the first event the buffer sees, and live that was a different
event than the one the file opens with, so the two grids sit at different phases
(live `window_end_us % 40000` = 28691, the replay's = 48). Per-window rows
therefore do **not** correspond one-to-one with the live run's. Per-poke aggregates
over a ~1 s dwell do, and that is what `ml/contact_error.py` reports.

## The clock

`ml/contact_error.py` joins camera to robot on `unix_time_s`, and a replay has no
wall clock of its own — the recording is being read now, not lived through.

`--time-ref` recovers it from the run's own `contact.csv`, which carries both
`window_end_us` (camera clock) and `unix_time_s` (workstation clock) for every
window. The two clocks agree to 1 part in 10⁶ over 8454 s, with a residual of
3.96 ms sd against 40 ms windows.

Both series are anchored at their **first row** — the same physical instant on both
clocks, since the contact log starts when recording starts. The summary prints the
replay's span beside the live log's as a standing check that they really are the
same recording.

> ⚠ Three clock traps, each of which produced entirely plausible output. The
> estimator was never at fault in any of them.
>
> **The epoch.** The `.raw` is rebased to zero when written; the GUI's
> `contact.csv` is not. On the ladder run the live log spans `window_end_us`
> 90.75 → 8545.19 s and the file spans 0 → 8454.4 s: the same 8454.4 s, offset by a
> constant 90.7 s. Mapping raw timestamps through a fit made against live ones put
> every window 90.7 s from where it belonged. It hid because the poke cycle is
> ~7.5 s and 90.7 s is 12.09 of them — only ~0.7 s from a whole number of cycles —
> so the first ~150 s still lined up and the alignment slipped only as the cycle
> length varied with the depth ladder. The symptom was contacts reported during
> out-of-contact holds (50%) and missed during dwells (56%), which reads exactly
> like a broken estimator. Anchoring both series at their first row removes it.
>
> **Locale.** `QCoreApplication` calls `setlocale(LC_ALL, "")`, and this
> workstation runs `LC_NUMERIC=kk_KZ.UTF-8`, where the decimal separator is a
> comma. `atof` therefore stopped at the `.` and read `1788688925.731514` as
> `1788688925`, discarding the fractional second. The residual was 289 ms — which
> is 1/√12 s, the sd of a uniform error over exactly one second, the fingerprint of
> truncation rather than of noise. `main()` now forces `LC_NUMERIC=C` after Qt has
> had its way. Writing was never affected: C++ streams obey `std::locale`, which Qt
> does not touch.
>
> **Conditioning.** `window_end_us` spans ~8.5×10⁹. Solving the normal equations on
> the raw values makes the denominator a difference of two ~6×10²⁹ quantities, and
> centring on the *first* sample does not help. The fit is centred on the **mean**.

## Validating the harness before trusting it

A replay is only evidence if the **pre-fix** code, replayed, reproduces what the
run actually logged. `--legacy-search` restores the old behaviour (search around
marker rest sites) for exactly this purpose. Over the first 400 s of the ladder
run:

| | tare windows with a contact | dwell windows with a contact |
| --- | --- | --- |
| live `contact.csv` (pre-fix build) | 5.3 % | 84.8 % |
| replay `--legacy-search` | 5.6 % | 84.7 % |
| replay, fixed | 5.3 % | 85.0 % |

The middle row is the control: it says the harness reproduces the run. Without it,
any number the replay produces is unattributable — which is what the first several
attempts here were, before the epoch bug was found.

The tare column is what exposes a bad replay, and it is why `ml/contact_error.py`
prints the tare check before it prints any result.

## Sweeping the threshold

The live path records a peak's strength only for windows its own threshold already
accepted, so a live `contact.csv` cannot be re-thresholded upward — the rejected
peaks are not in the file. The replay writes two extra columns so that it can:

| column | meaning |
| --- | --- |
| `peak_found` | a peak survived **pruning** (the calibrated coherence, resolvability and separation gates), whatever its strength |
| `second_divergence_px_per_cell` | the runner-up peak, so `ambiguous` can be re-derived |

`divergence_px_per_cell` is now written whenever `peak_found`, not only when the
window was accepted. One replay pass therefore supports any threshold ≥ 0:

```sh
for t in 0.75 1.0 1.25 1.5 2.0 3.0; do
    python3 ml/contact_error.py recordings/<run> \
        --contact contact_replay.csv --min-divergence "$t"
done
```

`ml/contact_error.py --min-divergence` refuses to run against a live
`contact.csv`, rather than silently reporting a truncation as a sweep.

This also repairs the calibration procedure described above
`kContactMinimumDivergence` in `src/contact_localiser.h` — "watch the div figure
and see whether it wanders near this". Previously a below-threshold window reported
a divergence of exactly 0.0, so the figure was pinned at zero by construction and
the instruction could not be carried out. It now shows the peak that is actually
there.

## Cost

Zero robot time and no new recording. The `.raw` is read, not written.

## Results on the ladder run (2026-09-06)

Full replay of all 8454.4 s: 211 362 windows against the live log's 211 342, spans
equal to 0.00 s, zero windows dropped.

Paired over the **890 pokes present in all three**, so the rows differ only in the
code that produced them:

| | gain x | gain y | median \|err\| | p90 |
| --- | --- | --- | --- | --- |
| live `contact.csv` — pre-fix, as run | 0.598 | 0.700 | 3.001 mm | 7.401 mm |
| replay `--legacy-search` — **control** | 0.637 | 0.719 | 2.727 mm | 7.210 mm |
| replay, fixed | **0.712** | **0.790** | **1.826 mm** | 6.528 mm |

Whole-run figures for the fixed replay: 1094 of 1100 pokes usable (live: 914, i.e.
the 17 % that produced no estimate is now 0.5 %), |bias| 0.985 mm, scatter 3.29 mm
(x) / 2.28 mm (y), median error 1.888 mm, tare 5.49 % and not drifting
(5.77 % → 5.22 %).

**What this says.** The fix is real and worth keeping: on identical pokes it cuts
median error by a third and lifts the gain by ~0.075. But the gain is **0.71 / 0.79,
not 1.0** — most of the under-travel is still there. Marker search was one cause of
the compression, not the whole of it.

The control row is 0.637/0.719 where the live run measured 0.598/0.700, a ~6 %
disagreement that bounds how finely this comparison can be read. It comes from the
window-phase offset and from the replay building its own baseline. Differences
smaller than that should not be interpreted.

### The divergence threshold

One replay pass, re-thresholded (`--min-divergence`); `kContactMinimumDivergence`
is currently 1.5:

| threshold px/cell | pokes usable | median | p90 | max |
| --- | --- | --- | --- | --- |
| 1.0 | 1097 | 1.898 | 6.798 | 25.9 |
| **1.5** (current) | 1094 | 1.888 | 6.770 | 25.9 |
| 2.5 | 1050 | 1.799 | 6.154 | 10.8 |
| 4.0 | 936 | 1.696 | 5.709 | 10.9 |
| 6.0 | 775 | 1.579 | 5.283 | 9.0 |

Raising 1.5 → 2.5 costs 4 % of pokes and removes the entire tail: worst case falls
from 25.9 mm to 10.8 mm. The outliers are weak peaks, and 1.5 is low enough to
admit them. On this evidence 1.5 is not the right operating point and ~2.5 is —
but note this is one run on one pad, and the gate it replaces was never the reason
pokes were being lost (that was the search-centre defect, now fixed).
