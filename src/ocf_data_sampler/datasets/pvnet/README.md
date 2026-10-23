# PVNet datasets

This module provides PyTorch datasets that construct input and target samples for PVNet
training and inference from configured generation, NWP and satellite sources.

## Sample construction

Initialisation reads the configuration and locations, opens the configured sources, and builds
an index of valid samples. Generation data is loaded into memory at this stage; NWP and satellite
data are loaded when samples are retrieved. Source coordinates and normalisation arrays are
prepared once for reuse.

```mermaid
graph TD
    Config[Configuration and requested locations] --> Sources[Open configured sources]
    Sources --> Generation[Load generation into memory]
    Sources --> Lazy[Keep NWP and satellite data lazy]
    Generation --> Availability[Find valid times for the requested windows]
    Lazy --> Availability
    Periods[Requested time periods] --> Availability
    Availability --> Index[Build sample index]
    Sources --> Prepared[Prepare source coordinates and normalisation arrays]
```

- `PVNetDataset`: each index selects one forecast reference time (`t0`) and location and
  returns a dictionary of NumPy arrays and metadata.
- `PVNetConcurrentDataset`: each index selects one `t0` that is valid at every requested
  location and returns a batch of tensors containing all locations.

The retrieval paths differ so that concurrent samples share loading and preprocessing:

### PVNetDataset

```mermaid
graph TD
    Pair[Index selects t0 and location] --> Space[Slice by space]
    Space --> Time[Slice by time]
    Time --> Load[Load selected data and preprocess]
    Load --> Sample[Build sample with configured features]
```

### PVNetConcurrentDataset

```mermaid
graph TD
    T0[Index selects t0] --> SharedTime[Slice by time]
    SharedTime --> SharedLoad[Load selected data and preprocess once]
    SharedLoad --> Locations[Slice by space for each location]
    Locations --> Samples[Build samples with configured features]
    Samples --> Batch[Collate into tensor batch]
```

### Shared processing and availability

Preprocessing includes NWP accumulation differencing, normalisation, input dropout and NaN
filling. Sample construction adds the configured solar position and time features.

Every requested location must have valid samples. Use `sampling_grid.exclude_location_ids` to
exclude locations intentionally. For concurrent samples, times that are unavailable at any
requested location are excluded from the index.

## Sample fields

See `NumpySample` and `TensorBatch` in [types.py](types.py) for fields, shapes and units.
NWP arrays are accessed as `sample["nwp"][provider]`. Optional diagnostics are grouped
under `sample["metadata"]`, including `sample["metadata"]["nwp"][provider]` for forecast
metadata. Generation capacities and timestamps remain alongside the generation arrays.

## Time conventions

`t0` is the forecast reference time. Each source's `interval_start_minutes` and
`interval_end_minutes` define its time window relative to `t0`. The bounds of each selected time 
slice for each source are floored to that source's configured time resolution, and both endpoints 
are included. This applies to generation, satellite, NWP, solar and datetime feature windows.

For example, with `t0=12:15`:

- A generation input window configured from `-60` to `0` at a 30-minute resolution will contain 
  values at `11:00`, `11:30` and `12:00`. 
- A generation target window configured from `30` to `120` at a 30-minute resolution will contain 
  values at `12:30`, `13:00`, `13:30`, and `14:00`. 
- A satellite window configured from `-15` to `0` at a 5-minute resolution will contain values at 
  `12:00`, `12:05`, `12:10` and `12:15`. 
- An NWP window configured from `-120` to `120` at a 60-minute resolution will contain values at 
  `10:00`, `11:00`, `12:00`, `13:00`, and `14:00`.
- A solar or datetime window configured from `0` to `120` at a 60-minute resolution will contain 
  values at `12:00`, `13:00` and `14:00`

Generation input and target windows are configured separately. A target window starting at
`0` includes the floored reference time; to start with the next time step, set its start
offset to the generation time resolution.

NWP windows refer to forecast valid times. Sampling selects the most recent eligible NWP
initialisation that covers the requested window, accounting for any configured availability
delay. Availability delays and the exclusive staleness limit are measured relative to `t0`
before flooring.

The dataset's `time_periods` argument restricts which **`t0` values** are sampled. These periods
include their start and exclude their end, independently of the source-window conventions above.

## DataLoader workers and randomness

When using multiple `DataLoader` workers, set `multiprocessing_context="forkserver"`.
The `fork` context is unsupported for TensorStore-backed datasets.

Dropout choices use NumPy's global random state (`np.random`). Repeated access to the same
sample index can therefore choose different dropout; the choice depends on the random state
at the time of sampling.

PyTorch seeds NumPy in each `DataLoader` worker. Seed PyTorch before creating the loader's
iterator, or supply a seeded `torch.Generator` to the loader, to control its worker seeds.
When accessing the dataset directly or using `num_workers=0`, seed `np.random` yourself.
Reproducing dropout choices also depends on keeping the worker setup and sampling order the same.
