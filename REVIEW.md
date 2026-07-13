# Code Review: `ocf-data-sampler` — `dev_feb2026_speedups` branch

**Reviewer:** Claude  
**Branch:** [`dev_feb2026_speedups`](https://github.com/openclimatefix/ocf-data-sampler/tree/dev_feb2026_speedups)  
**Base:** `main` (merge base `efd90e3`)  
**Date:** 2026-07-03

---

## Part 1: Branch-specific changes

The branch is a substantial rewrite, not purely a speedup pass: pandas→numpy datetime migration, a new `LightDataArray` backend over TensorStore, a hand-ported solar ephemeris replacing pvlib, flattened sample dicts, and deletion of the collate/validation utilities. The test suite passes (97 passed, 1 skipped). All findings marked **[verified]** were reproduced empirically.

### High severity

#### 1. Dropout crashes in `use_xarray=False` mode, stochastically **[verified]**

`apply_history_dropout` calls `da.where(...)`, which `LightDataArray` doesn't implement. Confirmed: `AttributeError: 'LightDataArray' object has no attribute 'where'`. Because dropout only fires with probability `dropout_frac`, a training run with `use_xarray=False` and generation/satellite dropout configured will appear healthy and then die mid-epoch.

This is not caught by tests because `pvnet_test_config.yaml` sets generation and satellite dropout to zero. Only NWP dropout is enabled in tests, and the NWP path doesn't use `.where`.

**Fix:** Implement `where` on `LightDataArray`, restructure dropout to operate on `.data` + a time mask, or raise at `__init__` if `use_xarray=False` and any non-NWP dropout is configured.

---

#### 2. `validate_sample_request` always raises for the incomplete-generation path **[verified]**

`find_valid_t0_and_location_ids` ends with `.reset_index(names="t0")`, so the resulting DataFrame has a `RangeIndex` and `t0` as a plain column. But `validate_sample_request` checks `t0 in self.valid_t0_and_location_ids.index` — comparing a datetime against integers. Verified: this is always `False`, so `get_sample()` is entirely broken whenever generation data contains NaNs.

A secondary issue: even if the index were correct, `.loc[t0, "location_id"] == location_id` returns a Series when multiple locations share a t0, and evaluating truth on a Series raises.

**Fix:** Use a proper `(t0, location_id)` membership check. Add a test for this path.

---

#### 3. `fill_nans(copy=False)` mutates the shared source dataset through views **[verified]**

Integer/slice `.isel()` returns numpy views into the eagerly-loaded generation array. `np.nan_to_num(..., copy=False)` writes through those views to the parent array. Verified in isolation.

Today the corruption is value-identical to what any future fill would produce, so samples don't differ — but it silently modifies the NaN state in the shared `datasets_dict`. There is a nastier consequence: if `capacity_mwp` contains NaNs, they get filled with `dropout_value` (typically `0.0`), and then `if capacity_value != 0` skips normalisation, leaving **raw MW values in a sample stream that is otherwise normalised to 0–1**, silently.

The `copy=False` saves one allocation per sample — not worth it.

**Fix:** Remove `copy=False`, or explicitly copy the generation slice before it is filled.

---

#### 4. `select_time_slice_nwp` silently uses a future init time on index wraparound **[verified]**

```python
selected_init_time_index = np.searchsorted(all_init_times, t0_available, side="right") - 1
```

When `t0_available` precedes the first init time, `searchsorted` returns `0` and the subtraction gives `-1`, selecting `all_init_times[-1]` — the **last** init time. This is temporal leakage: the model silently receives data from a future model run.

The valid-period machinery in `find_contiguous_t0_periods_nwp` should prevent this in normal operation, but any mismatch between the dropout accounting there and the actual draw here — or a direct call via `get_sample` — triggers the bug with no error.

**Fix:**
```python
if selected_init_time_index < 0:
    raise ValueError(
        f"No valid init time found: t0_available={t0_available} precedes all init times"
    )
```

---

#### 5. Two silent-wrongness paths in `convert_to_numpy_sample` **[verified]**

- `np.argmax(da.gen_param.values == "generation_mw")` returns `0` when the label is absent (no match), silently using the first parameter rather than raising.
- When `capacity_value == 0` (a degenerate site or a fill-NaN artifact from issue #3), normalisation is skipped and raw MW values enter the sample. The branch should raise or zero rather than pass through.

**Fix:** Check explicitly that the label exists before indexing. Decide and document the intended behaviour for zero-capacity sites.

---

### Medium

#### `LightDataArray.__getattr__` causes infinite recursion on uninitialised instances **[verified]**

If `coords` isn't set (e.g. during `__new__` before `__init__`, unpickling from a partial state, or a failed `__init__`), `__getattr__("coords")` calls itself recursively. Confirmed `RecursionError`.

**Fix:**
```python
def __getattr__(self, name: str) -> "LightDataArray":
    try:
        coords = object.__getattribute__(self, "coords")
    except AttributeError:
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    if name in coords:
        return self[name]
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
```

---

#### `PickleCacheMixin.__setstate__` fails silently when the pickle path is missing

If `_pickle_path` was set but the file no longer exists (different machine, cleaned temp dir, path mismatch across worker nodes), the object is restored with only `_pickle_path` set — every subsequent attribute access is a confusing crash far from the cause.

**Fix:** Raise `FileNotFoundError` explicitly. Add a docstring noting the presaved path must be accessible to all DataLoader workers.

---

#### Global numpy RNG in `__getitem__` under forked DataLoader workers

Both the satellite/generation dropout path and the NWP dropout path draw from `np.random`'s global state. With `num_workers > 0` under fork, all workers inherit identical RNG state, producing correlated dropout patterns — the classic duplicated-augmentation bug.

**Fix:** Use `np.random.Generator` seeded from `torch.utils.data.get_worker_info()` in a `worker_init_fn`, or seed via `np.random.seed(worker_id)` at minimum. Since this library owns the `Dataset`, it should own the fix.

---

#### `datetime_ceil`/`floor` use float division on nanosecond offsets

`(datetimes - epoch) / freq` converts int64 nanosecond values to float64. A 2026 timestamp in ns is ~1.8×10¹⁸; the float64 ULP at that scale is ~256 ns, leaving a margin of ~1×10⁻¹⁰ against a ceil boundary. Empirically no off-by-one errors were found over a full year of 5-min and 30-min aligned timestamps, but the failure mode is a `ValueError` from `get_indices_in_sorted_unique` at a random point in training.

Integer arithmetic is exact at no performance cost:

```python
# ceil: -(-a // b) pattern on int64 ns
periods = -(-((datetimes - epoch).astype("int64")) // int(freq / np.timedelta64(1, "ns")))
```

---

#### Dead validation code and asymmetric input checking in `select_time_slice_nwp`

```python
if len(dropout_timedeltas) > 0:
    if not all(t < np.timedelta64(0) for t in dropout_timedeltas):
        raise ValueError(...)
    if len(dropout_timedeltas) < 1:   # dead: always False inside `> 0` block
        raise ValueError(...)
```

Additionally, `apply_history_dropout` doesn't validate that timedeltas are negative while the NWP path does — asymmetric contracts for the same concept.

---

#### `get_t0_embedding` silently changes output dimension on bad input

- An unrecognised period suffix (neither `h` nor `y`) leaves `frac` unbound, raising `NameError`.
- An unrecognised `embedding_type` (neither `"cyclic"` nor `"linear"`) is silently skipped, changing the feature vector length with no error or warning.

The config validator in `T0Embedding` prevents most paths to this function, but it is a public function and the lack of internal guards is surprising.

---

#### pvlib attribution is incomplete

The ported ephemeris is accurate (verified against `pvlib.solarposition.ephemeris`: agreement to <1×10⁻¹²°; vs the SPA used in main: max 0.03° azimuth / 0.01° elevation over a year — negligible for ML features). However pvlib is BSD-3-Clause, which requires retaining the copyright notice, not only a URL reference. The docstring comment "the SPA algorithm needs time..." is also misleading — this is the low-precision ephemeris, not SPA.

---

### API and coordination notes

This is a breaking release. Before merging, a coordinated version bump and downstream migration should cover:

| Removed | Replacement |
|---|---|
| `collate.stack_np_samples_into_batch` | None (callers must use `default_collate` directly) |
| `torch_batch_utils.batch_to_tensor` etc. | None |
| `validation_utils` | None |
| Nested `nwp` sample key | Flat `nwp_{provider}` keys |
| `satellite_actual` | `satellite` |
| `time_utc` | `generation_time_utc` |

**Presaved batches on disk are key-incompatible.** They cannot be loaded by any model that expects the old schema without a migration script.

**`t0` is POSIX seconds; `generation_time_utc` / `satellite_time_utc` are datetime64[ns]-as-float (nanoseconds).** Mixed epoch units in one sample dict is a latent bug in any downstream script that treats them uniformly.

**`add_alterate_coordinate_projections`** — the typo is now in a public function name. Rename before it fossilises.

---

### Test coverage gaps

The following paths have no test coverage:

- `use_xarray=False` with generation or satellite dropout enabled (which is where issue #1 lives)
- Incomplete-generation `get_sample` / `validate_sample_request` path
- `PickleCacheMixin` round-trip with a missing or unreachable pickle path
- `datetime_ceil(aligned_timestamp, freq) == aligned_timestamp` across a large ns range

Recommended addition: an integration test asserting the source `datasets_dict` is bit-identical before and after generating a sample. This would have caught issue #3 and guards the whole class of in-place mutation bugs.

---

## Part 2: Broader package review

This section covers the parts of the package the branch did not change: config, loaders, geospatial, packaging, and CI.

### High severity

#### 6. ICON-EU is unusable through the config path — a name deadlock **[verified]**

`NWP_PROVIDERS` in `config/model.py` contains `"icon_eu"` (underscore). `open_nwp` dispatches on `"icon-eu"` (hyphen).

```
open_nwp(provider="icon_eu")   → ValueError: Unknown provider: icon_eu
NWP(provider="icon-eu", ...)   → OSError: NWP provider icon-eu is not in [...]
```

Neither spelling works end-to-end. The loader test avoids this by calling `open_nwp` directly, bypassing the config.

The underlying cause is a triplicated provider registry: `NWP_PROVIDERS` in the config, the dispatch chain in `open_nwp`, and the dtype table in `_validate_nwp_data` — all maintained independently and already drifted. A single registry dict mapping provider name → (opener, expected coords/dtypes), consumed by all three, makes this class of bug structurally impossible.

As a secondary issue, the validator raises `OSError` for what is a validation error. It should raise `ValueError`.

---

#### 7. `convert_coordinates` has an unraised exception with an inverted guard **[verified]**

In `geospatial.py`:

```python
if "geostationary" in (from_coords, target_coords) and area_string is not None:
    ValueError("If using geostationary coords the `area_string` must be provided")
```

Two bugs in one line: the `ValueError` is instantiated and silently discarded (no `raise`), and the condition fires when `area_string` **is** provided rather than when it's absent. As written this is a pure no-op: a missing `area_string` passes the guard, falls through to `load_area_from_string(None)`, and dies with an unrelated traceback from inside pyresample.

This is exactly the class of issue a ruff `B` rule (useless expression / unraised exception) should catch — worth checking why the current config misses it.

---

#### 8. `open_zarrs` silently overwrites mismatched coordinates

```python
except ValueError:
    logger.warning("Coordinate mismatch found ... coordinates will be overwritten! ...")
    ds = xr.concat(..., join="override")
```

Any `ValueError` from the strict `join="exact"` concat triggers a retry with `join="override"`, silently clobbering coordinate values. Two zarrs on genuinely different grids — a regridded satellite archive next to an older one, a UKV domain change between operational runs — are silently merged with their coordinates papered over, and a model trains on spatially misaligned data with only a log line between you and the problem.

The bare `except ValueError` also catches unrelated concat failures.

**Fix:** Quantify the mismatch (max coordinate delta) and only override below a tolerance, restrict the override path to explicitly configured allowlists, or make it opt-in via a caller argument rather than automatic.

---

### Medium

#### Backend capability matrix is undocumented and inconsistent

GFS and ICON-EU load via the dask backend; UKV, ECMWF, GDM, and cloudcasting use TensorStore. Consequences:

- `LightDataArray.from_xarray` raises `ValueError` on dask-backed arrays **[verified]** — so `use_xarray=False` is incompatible with any GFS or ICON-EU config. It fails at `__init__` (loud), but nothing in the docs or config validation tells you this.
- The two dask-backed providers silently forfeit the async-prefetch speedup the branch introduces.
- `xtr_read` passes dask arrays through unchanged, so they work in xarray mode but without any speedup.

The backend a provider uses is an undocumented implementation detail with significant performance and API consequences. At minimum a table in the module docstring; ideally migration of GFS/ICON to TensorStore.

---

#### `open_mfdataset` in the dask path uses `compat="no_conflicts"` with `coords="different"`

This forces loading and comparing all coordinate values across files during open — slow at scale — and permits merging files whose coordinates "don't conflict" rather than requiring equality. Duplicate init-times across files are not deduplicated; they will only surface as `assert_values_unique_increasing` failures at sample time, far from the cause.

---

#### Module-level pyproj Transformers are shared and not thread-safe

`_osgb_to_lon_lat` and `_lon_lat_to_osgb` are instantiated at import time and shared globally. pyproj Transformers are not thread-safe: concurrent `transform()` calls produce wrong coordinates, not just crashes. Under DataLoader multiprocessing this is safe (separate processes), but this module is imported and used in threaded contexts elsewhere in the OCF stack.

**Fix:** A `threading.local` wrapper, or at minimum a thread-safety caveat in the module docstring.

---

#### Config validation gaps

| Gap | Effect |
|---|---|
| `dropout_timedeltas_minutes` allows `0` (validator: `if m > 0`), but `select_time_slice_nwp` requires strictly negative | A config with a 0-minute timedelta validates, fails per-sample at runtime |
| `dropout_fraction: float \| list[float]` in `DropoutMixin`, but NWP sampling only handles scalar (`np.random.uniform() < dropout_frac`) | A list value passes config validation, raises at sample time |
| `zarr_path: str \| tuple[str] \| list[str]` — `tuple[str]` means a 1-tuple; a 2-element path tuple is rejected | Should be `tuple[str, ...]` |
| `Generation.public` description reads "Whether the NWP data is public or private" | Copy-paste error |

---

#### `get_dataset_dict` drops location 0 with a label-slice assumption

```python
da_generation = da_generation.sel(location_id=slice(1, None))
```

This assumes location IDs are sorted ascending, so label-slicing from 1 excludes exactly ID 0. `open_generation` does assert sorted IDs, so it holds — but the intent is invisible. More critically, this is a hard-coded GSP-domain rule (national total = ID 0) in a supposedly domain-neutral function, with the GSP CSV files now deleted from the package. This is the last stranded piece of GSP-specific logic and deserves a config flag or a prominent comment.

---

#### `_validate_nwp_data` has incomplete coverage and duplicates the provider registry

The provider-specific dtype table covers ecmwf, icon-eu, gfs, mo_global, ukv, and cloudcasting, but not gencast/gdm — a gdm zarr with wrong spatial dtypes passes with only the common checks. This table is the third copy of the provider list (alongside `NWP_PROVIDERS` and the dispatch chain), and has already diverged from both.

---

### Packaging and CI

**`pyproject.toml` issues:**

- `zarr` is listed twice (`"zarr"` and `"zarr>=3"`). The resolver takes the intersection, but it's confusing.
- `matplotlib` is a declared runtime dependency with no imports in the package (verified by grep). It's a non-trivial transitive dependency for a library that has no visualisation functionality.
- `h5netcdf` similarly has no direct imports. If it's required as an xarray backend for external workflows, it belongs in an optional extra.
- `torch` is unpinned. By default pip resolves to the CUDA-bundled wheel — several GB — for a sampling library that never needs a GPU. The CPU-wheel install path (`--index-url https://download.pytorch.org/whl/cpu`) is worth documenting.
- `xarray-tensorstore==0.1.5` is an exact pin on a package whose private internals (`_TensorStoreAdapter`, `_raise_if_mask_and_scale_used_for_data_vars`, `da.variable._data`) are accessed directly. The pin is load-bearing and should say so explicitly.
- `package-data "*" = ["*.csv"]` is likely vestigial now the GSP CSVs have been deleted.

**Dev deps use PEP 735 `[dependency-groups]`, not extras.** `pip install ".[dev]"` silently does nothing (confirmed: "does not provide the extra 'dev'"). The development installation instructions should use `pip install --group dev` or `uv sync --group dev` explicitly.

**CI:**

- `enable_typechecking: false` while `pyproject.toml` carries a full `[tool.mypy] strict = true` configuration. The config is aspirational decoration — a strict-mypy block that CI ignores actively misleads contributors. Either ratchet it on per-module, or remove it.
- Test matrix covers 3.11/3.12, but `requires-python` allows 3.13 — untested-but-claimed support.
- The GFS loader test is permanently skipped (`"Fixture 'nwp_gfs_zarr_path' is not yet defined"`). The one dask-backend NWP provider with a `public` flag has zero test coverage.

---

### Smaller observations

**`Location.add_coord_system` uses exact float equality** to detect conflicting re-adds. Two independent pyproj transform calls for the same point will differ at ~1e-9 and raise a spurious mismatch error. Use `np.allclose` with a tolerance appropriate to the coordinate system (e.g. ~1 m for OSGB).

**`find_coord_system` failure message doesn't name the variables found.** One f-string change away from a much faster diagnosis.

**`open_ukv`'s conditional rename** (`if k in ds.coords`) quietly supports two schema generations of UKV zarr. The tolerated schemas are only discoverable by reading the rename dict. A short "supported UKV schema versions" note in the module docstring would save future archaeology.

**`get_xr_data_array_from_xr_dataset`** raises on multi-variable datasets but doesn't name the variables found — one f-string away from a significantly faster diagnosis.

**`encode_datetimes` divides day-of-year by 365 unconditionally** while `get_t0_embedding` handles leap years. Inconsistent; probably inherited from an earlier version.

**Typos across the codebase:** "dta" (`lightarray.py`), "entending" (`time_utils.py`), "frequecies" (`time_utils.py`), "Normalsation" (`config/model.py` ×2), "alterate" (public function name — rename before release).

---

## Summary by priority

### Fix before merge

| # | Issue | File(s) |
|---|---|---|
| 1 | Dropout crashes with `use_xarray=False` + generation/satellite dropout | `lightarray.py`, `select/dropout.py` |
| 2 | `validate_sample_request` always raises for incomplete-generation path | `torch_datasets/pvnet_dataset.py` |
| 3 | `fill_nans(copy=False)` mutates parent dataset through views | `torch_datasets/utils/fill_nans.py` |
| 4 | `select_time_slice_nwp` silently uses future init time on wraparound | `select/select_time_slice.py` |
| 5 | `convert_to_numpy_sample` silent-wrong on missing gen_param label | `numpy_sample/convert.py` |
| 6 | ICON-EU deadlocked between config name and loader name | `config/model.py`, `load/nwp/nwp.py` |
| 7 | `convert_coordinates` has unraised exception with inverted condition | `select/geospatial.py` |

### Fix before next major release

| # | Issue |
|---|---|
| 8 | `open_zarrs` silently overwrites coordinate mismatches |
| 9 | `LightDataArray.__getattr__` recursion on uninitialised instances |
| 10 | `PickleCacheMixin.__setstate__` silent failure on missing pickle path |
| 11 | Global numpy RNG not seeded per DataLoader worker |
| 12 | Float division in `datetime_ceil`/`floor` — integer arithmetic is exact |
| 13 | `get_t0_embedding` silent wrong output on bad inputs |
| 14 | Triplicated provider registry — consolidate into one dict |
| 15 | pvlib copyright notice missing from ephemeris port |
| 16 | Backend capability matrix undocumented (GFS/ICON incompatible with `use_xarray=False`) |
| 17 | Config validation gaps (dropout zero timedelta, list fraction, tuple path, copy-paste description) |
| 18 | `Location.add_coord_system` exact float equality comparison |
| 19 | `add_alterate_coordinate_projections` — rename typo |

### Housekeeping

- Remove `matplotlib`, `h5netcdf` from runtime deps; document CPU-torch wheel
- Remove vestigial `*.csv` package-data glob
- Deduplicate `zarr` dep entry
- Fix dev dep installation docs (PEP 735 group, not extra)
- Either enable mypy in CI or remove the strict config
- Add GFS test fixture; add 3.13 to test matrix or remove from `requires-python`
- Rename `add_alterate_coordinate_projections`
- Add sample-dict key schema to torch_datasets README