# Intel GPU monitoring: backend architecture

This document describes fork-specific changes to how `btop` monitors Intel GPUs on
Linux. It intentionally lives outside `README.md` so the main upstream documentation
stays untouched and easy to keep in sync with `aristocratos/btop`.

(The name is `INTEL_XPU.md` rather than `INTEL_GPU.md` because future backends may
target other Intel accelerator classes beyond GPUs, e.g. NPUs.)

## The bug this fixes

Upstream `btop`'s Intel GPU backend (`Gpu::Intel`, in `src/linux/btop_collect.cpp`,
backed by the vendored `igt-gpu-tools`-derived sources in
`src/linux/intel_gpu_top/`) computes the headline "GPU busy%" as:

```cpp
double max_util = 0;
for (each i915/xe PMU per-engine "busy" counter)
    max_util = max(max_util, engine_busy_pct);
```

On Xe-architecture GPUs (Arc, Lunar Lake, Battlemage, Meteor Lake and newer), a single
fully-saturating workload is commonly dispatched across **multiple parallel engine
instances** (e.g. two CCS compute queues). Each instance individually peaks around
50% even though the GPU die is fully busy, so `max()` across engines never reports
anywhere close to 100%. This is the root cause of the "graph stuck around 50% under
100% load" symptom.

Two adjacent bugs live in the same code path:
- The PMU device name is hardcoded to `"i915"`, so GPUs bound to the newer `xe` kernel
  driver (Arc, Lunar Lake, Battlemage, some Meteor Lake configurations) aren't detected
  at all by this backend.
- `device_count` is hardcoded to `1`, so only the first Intel GPU on a multi-GPU system
  is ever reported.

## The fix: a 3-tier runtime fallback chain

`Shared::init()` now tries three backends in order, the richest/most broadly-available
first, falling back only if the previous one fails to initialize:

| Priority | Backend | Trigger | Notes |
|---|---|---|---|
| 1 | `Gpu::Intel::LevelZero` | `dlopen("libze_loader.so.1")` succeeds and `zesInit`/`zesDriverGet` finds a driver | Richest telemetry: accurate VRAM via `zesMemoryGetState`, true per-engine activity via `zesEngineGetActivity` (not susceptible to the split-engine undercount). Optional — gracefully skipped if the library isn't installed. |
| 2 | `Gpu::Intel::Sysfs` | GT RC6/idle-residency sysfs files found under `/sys/class/drm/card*` | The practical default fix. Pure sysfs reads, no special permissions needed. |
| 3 | `Gpu::Intel` (legacy PMU) | Only reached if both tiers above fail to initialize | Unmodified upstream code, kept byte-for-byte for upstream traceability. Only reachable on a kernel old enough to lack the GT residency sysfs files entirely (i915 predating ~2012-era kernels) — effectively a theoretical last resort on any supported system today. |

Exactly one tier is ever active. `Gpu::collect()` and the `btop.cpp` shutdown sequence
call all three unconditionally; each is a no-op (`device_count == 0`) when it isn't the
active tier — the same pattern already used to let `Rsmi` and `Asysfs` coexist for AMD.

### Why `Gpu::Intel::Sysfs` fixes the bug

Instead of per-engine PMU busy counters, this backend reads each GT's **RC6/idle
residency** counter — "% of time the GPU block was *not* power-gated":

- `xe` driver: `.../device/tileN/gtM/gtidle/idle_residency_ms`
- `i915` driver, multi-GT (Meteor Lake+): `.../card/gt/gtN/rc6_residency_ms`
- `i915` driver, legacy single-GT: `.../card/power/rc6_residency_ms`

`busy% = 100 - (100 * Δresidency_ms / Δwall_ms)`, taking the max across a card's GTs.
This reflects whether the GPU was doing *any* work at all, regardless of how many
engine instances that work was split across, so it doesn't suffer the legacy PMU
backend's undercount. It also dynamically detects whether the bound driver is `i915`
or `xe` (via the `device/driver` symlink) instead of assuming `i915`, and enumerates
every Intel GPU under `/sys/class/drm/card*` instead of only the first one found.

Power is read from `hwmon/power1_average` (falling back to `power1_input`), with the
power cap preferred from `power1_max` → `power1_rated_max` → `power1_cap` over letting
the observed peak set the scale. Clock speed is the average of each GT's active
frequency (falling back to the requested/current frequency when a GT is parked in
RC6, which reads an active frequency of 0).

### Why `Gpu::Intel::LevelZero` is now `dlopen`-based

Previously, Level Zero support (in the separate `tkalceca/btop-gpu-lz` repository this
was prototyped in) was gated behind a compile-time CMake flag and hard-linked against
`ze_loader`. That meant a binary built with the flag on would refuse to start on a
machine without the library installed. This fork instead loads `libze_loader.so.1`
(falling back to `libze_loader.so`) via `dlopen`/`dlsym` at runtime, exactly like the
existing `Nvml` (`libnvidia-ml.so`) and `Rsmi` (`librocm_smi64.so`) backends already
do — always compiled in, gracefully skipped if the library isn't present.

It also sets `ZES_ENABLE_SYSMAN=1` (without clobbering a user's own setting) before
calling `zesInit`, for compatibility with older loader versions that relied on it.
**Correction/update**: the current upstream Level Zero specification documents this
environment variable as the *deprecated* initialization method — `zesInit()` itself is
the current, correct, and sufficient call (no `zeInit()` call is required before or
after it). Setting the variable is a harmless no-op on current loaders, kept only for
backward compatibility; it is not what makes `zesInit()` succeed or fail. A `zesInit()`
failure (e.g. `ZE_RESULT_ERROR_UNINITIALIZED`) on a real system most commonly means no
Sysman-capable GPU backend/ICD was discoverable by the loader at that moment (missing
`intel-level-zero-gpu`/`libze-intel-gpu1` or equivalent, insufficient permissions on
the render node, etc.) — in which case this tier correctly falls back to `Sysfs`.

## Fixed: crash in `Gpu::Intel::Sysfs` when a card's power source is absent/unreadable

The initial version of `Gpu::Intel::Sysfs` set `supported_functions.pwr_usage` from
whether an `hwmon*` **directory** existed for the card, while the code that actually
populates `gpu_percent["gpu-pwr-totals"]` required a successfully-read, positive power
value. On a card whose `hwmon` directory exists but whose `power1_average`/
`power1_input` files are missing, unreadable, or read as non-positive on every sample
(observed on real hardware: a VM/virtualized Intel GPU with a stub hwmon node and no
real power sensor), the flag was `true` while the deque stayed **permanently empty**.
`Cpu::draw()` trusts that invariant unconditionally and calls `.back()` on it —
undefined behavior (plain segfault in a release build) or an assertion abort (in a
`_GLIBCXX_ASSERTIONS`-enabled debug build, which is what actually surfaced it).

Fixed by:
- Deriving `supported_functions.pwr_usage` from whether a power source **file** (not
  just the hwmon directory) actually exists: `power1_average`, `power1_input`, or
  `energy1_input`.
- Adding an `energy1_input`-based wattage fallback (cumulative microjoules, delta over
  wall time — same technique `xpu-top` uses) for cards that expose only an energy
  counter, so more real hardware gets genuine wattage instead of "unsupported".
- Seeding `"gpu-pwr-totals"` with `0` during `is_init` in every code path that doesn't
  produce a usable reading, guaranteeing the deque can never be empty whenever
  `supported_functions.pwr_usage` is `true` — mirroring the safeguard already in place
  for `"gpu-totals"`.

## Build requirements

`Gpu::Intel::LevelZero` needs the Level Zero **headers** (`level_zero/zes_api.h`) at
**build time only** (for struct/enum definitions — the library itself is resolved at
runtime via `dlopen`). This follows the exact precedent `Rsmi` already sets for
`<rocm_smi/rocm_smi.h>` in its static-linking mode.

Install the relevant package before building with `GPU_SUPPORT=true` (or `-DBTOP_GPU=ON`):

| Distro | Package |
|---|---|
| Alpine | `level-zero` (`community` repo) |
| Debian / Ubuntu | `libze-dev` (or `liboneapi-level-zero-dev` on some releases) |
| Fedora | `oneapi-level-zero-devel` |

No extra link-time library is required — there is no `ze_loader` entry in
`target_link_libraries`/the Makefile's link step; the library is resolved purely at
runtime via `dlopen`.

## Deferred / out of scope for this change

- **VRAM reporting for the `Sysfs` tier.** Unlike AMD (`mem_info_vram_total`/`_used`
  directly in sysfs), Intel has no simple universal sysfs node for VRAM usage. Level
  Zero's `zesMemoryGetState` already covers this when available; a sysfs-only
  equivalent would need a render-node ioctl query, which is a larger, separable
  follow-up.
- **RAPL "uncore" power fallback** for the `Sysfs` tier on integrated GPUs without a
  dedicated hwmon power node. Not needed to fix the reported busy% bug; a nice-to-have
  for a future pass.
- `Nvml`/`Rsmi`/`Asysfs` (NVIDIA/AMD) are unchanged — this pass is Intel-only.

## Intel NPU monitoring (`Npu` namespace)

Added alongside the GPU work, following the same pure-sysfs approach: a new `Npu`
namespace in `src/linux/btop_collect.cpp` discovers Intel NPUs (`ivpu`/`intel_vpu`
driver — Meteor Lake, Lunar Lake, Arrow Lake, Panther Lake, etc.) under
`/sys/class/accel/accelN`, mirroring `xpu-top`'s `discover_npus()`/`NpuDev` model:

- **Busy%**: `npu_busy_time_us` (microseconds, kernel 6.10+) delta-over-wall-time,
  falling back to the coarser `power/runtime_active_time` (milliseconds, "device
  resumed" rather than "jobs executing") on older kernels.
- **Power**: same `power1_average` → `power1_input` → `energy1_input`-delta fallback
  chain built for the GPU `Sysfs` tier, including the same empty-deque seeding
  safeguard from day one (learned from the GPU crash above).
- **Temp**: `hwmon/temp1_input`.

This is intentionally **Phase 1 only**: brief one-line NPU rows in the CPU panel
(`Cpu::draw`), stacked directly below the existing GPU brief rows, reusing
`Gpu::gpu_info`/`gpu_info_supported` as-is (VRAM/PCIe/encode-decode fields stay
`false` — not applicable; the NPU shares system RAM rather than having dedicated
memory). New config: `shown_npus` (default `"intel"`) and `show_npu_info`
(Auto/On/Off, default `Auto`), mirroring the GPU equivalents.

**Explicitly deferred**: a dedicated `npuN` box/column (`shown_boxes` integration,
mirroring `Gpu::draw()`'s per-device detail box) — real follow-up if Phase 1 proves
useful, but most of that box's content (VRAM, PCIe, encode/decode) wouldn't apply to
an NPU anyway. Also deferred: AMD Ryzen AI (`amdxdna`), which exposes utilization via
a DRM ioctl rather than a plain sysfs busy-time file — separate work from what's here.

### NPU segments in the CPU box's split graph

The CPU box's big upper/lower graph area already splits horizontally into one
segment per GPU whenever `cpu_graph_upper`/`cpu_graph_lower` resolves to
`"gpu-totals"` (e.g. via `"Auto"`, which picks `gpu-totals` for the lower graph
whenever a GPU — now also an NPU — is present). Since `Npu::npus` stores busy% under
that exact same `gpu-totals` key (it reuses `Gpu::gpu_info`), `Cpu::draw()`'s
`init_graphs`/`draw_graphs` now simply walk GPUs **then** NPUs into the same segment
row: the available width is divided evenly across the combined GPU+NPU count, GPU
segments are drawn first (labeled `GPU0`, `GPU1`, ...), NPU segments follow
immediately after (labeled `NPU0`, `NPU1`, ...), with the same divider-line styling
carrying seamlessly across the boundary. No new config value — an NPU-only machine
(no GPU at all) now also triggers this split via `"Auto"`, showing just its NPU
segment(s) alone.

