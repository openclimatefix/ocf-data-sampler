# ocf-data-sampler

[![tags badge](https://img.shields.io/github/v/tag/openclimatefix/ocf-data-sampler?include_prereleases&sort=semver&color=FFAC5F)](https://github.com/openclimatefix/ocf-data-sampler/tags)
[![ease of contribution: easy](https://img.shields.io/badge/ease%20of%20contribution:%20easy-32bd50)](https://github.com/openclimatefix#how-easy-is-it-to-get-involved) 

**ocf-data-sampler** prepares samples to feed training and inference of OCF's renewable energy forecasting models such as [PVNet](https://github.com/openclimatefix/PVNet). It handles generation data, satellite imagery, and numerical weather predictions. For a given forecast init-time (t0) and geographical location the sampler selects the time periods and spatial crops needed for each sample, and applies normalisation and feature transformations. These datasets are often very large so loading data for these samples is done lazily and on-the-fly.


> [!Note]
> This repository is still in early development and large changes to the user facing functions may still occur.

## Licence 

This project is primarily licensed under the MIT License (see LICENSE).

It includes and adapts the `ephemeris()` function from the [pvlib-python](https://github.com/pvlib/pvlib-python) project, licensed under the BSD 3-Clause License. The original notice is retained in `src/ocf_data_sampler/features/solar.py`.

## Documentation

See the [PVNet dataset guide](src/ocf_data_sampler/datasets/pvnet/README.md) for sample construction, time conventions, and DataLoader configuration.

## FAQ

If you have any questions about this or any other of our repos, don't hesitate to hop to our [Discussions Page](https://github.com/orgs/openclimatefix/discussions)!

### How does ocf-data-sampler deal with data sources that use different projections (e.g. some are in latitude-longitude, and some in OSGB)?

When creating samples, we select a spatial crop of each data source around a given geographical location. This location might be an individual solar farm or wind turbine, or a representative point for a region or country. Each data source is configured with a spatial crop size in pixels rather than kilometres. The physical area sliced out depends on the data source’s resolution and coordinate system. For example, on a regular longitude–latitude grid the physical size in kilometres depends on whether the grid is say 0.1° or 1°. The physical size also varies with latitude because the distance between lines of longitude decreases towards the poles.

Each data source is kept on its original grid. We transform the sampling location into the source’s coordinate system and select a crop approximately centred on that location, without reprojecting or resampling the data onto a shared grid. This gives the model a fixed number of pixels from each source, although those crops may cover different physical areas.

## Development

With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed, clone the
repository and install the package and development dependencies into `.venv`:

```bash
git clone https://github.com/openclimatefix/ocf-data-sampler.git
cd ocf-data-sampler
uv sync --group dev
```

### Running the test suite

The development dependencies include `pytest`, `pvlib`, and `dask`, which the tests
require. After completing the setup above, run the suite from the repository directory:

```bash
uv run python -m pytest
```

## Contributing and community

[![issues badge](https://img.shields.io/github/issues/openclimatefix/ocf-data-sampler?color=FFAC5F)](https://github.com/openclimatefix/ocf-data-sampler/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc)

- PR's are welcome! See the [Organisation Profile](https://github.com/openclimatefix) for details on contributing
- Find out about our other projects in the [OCF Meta Repo](https://github.com/openclimatefix/ocf-meta-repo)
- Check out the [OCF blog](https://openclimatefix.org/blog) for updates
- Follow OCF on [LinkedIn](https://uk.linkedin.com/company/open-climate-fix)

---

*Part of the [Open Climate Fix](https://github.com/orgs/openclimatefix/people) community.*

[![OCF Logo](https://cdn.prod.website-files.com/62d92550f6774db58d441cca/6324a2038936ecda71599a8b_OCF_Logo_black_trans.png)](https://openclimatefix.org)

