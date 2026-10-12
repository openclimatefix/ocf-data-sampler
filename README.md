# ocf-data-sampler

[![tags badge](https://img.shields.io/github/v/tag/openclimatefix/ocf-data-sampler?include_prereleases&sort=semver&color=FFAC5F)](https://github.com/openclimatefix/ocf-data-sampler/tags)
[![ease of contribution: easy](https://img.shields.io/badge/ease%20of%20contribution:%20easy-32bd50)](https://github.com/openclimatefix#how-easy-is-it-to-get-involved) 

**ocf-data-sampler** contains all the tools needed to create samples and feed them to our models, such as [PVNet](https://github.com/openclimatefix/PVNet/). The data we work with—typically energy data, satellite imagery, and numerical weather predictions (NWPs)—is usually too heavy to do this on the fly, so that's where this repo comes in: handling steps like opening the data, selecting the right samples, normalising and reshaping, and saving to and reading from disk.


> [!Note]
> This repository is still in early development development and large changes to the user facing functions may still occur.

## Licence 

This project is primarily licensed under the MIT License (see LICENSE).

It includes and adapts the `ephemeris()` function from the [pvlib-python](https://github.com/pvlib/pvlib-python) project, licensed under the BSD 3-Clause License. The original notice is retained in `src/ocf_data_sampler/features/solar.py`.

## Documentation

**ocf-data-sampler** doesn't have external documentation _yet_; you can read a bit about how our torch datasets work in the README [here](src/ocf_data_sampler/datasets/pvnet/README.md).

## FAQ

If you have any questions about this or any other of our repos, don't hesitate to hop to our [Discussions Page](https://github.com/orgs/openclimatefix/discussions)!

### How does ocf-data-sampler deal with data sources that use different projections (e.g. some are in latitude-longitude, and some in OSGB)?

When creating samples, we make a spatial crop of a preset size centred around a point of interest (POI, usually a solar or wind farm). The size of the crop is set not in miles or kilometres, but in 'pixels', which would be different for different data sources, depending on their spatial resolution, projections they use, and where the POI is. For example, a latitude-longitude source with a 1° resolution will have pixel sizes corresponding to very different 'surface' distances (that you might measure in, e.g., kilometres) from a source with 0.1° resolution. The pixel size will even be different for the same source depending on how close the POI is to the equator!

Instead of trying to accommodate for all these differences and make all the sources use the same spatial grid, we translate the POI's position into the corresponding coordinate system and select the crop using the source's original grid. This 'snapshot' is then passed to the model with no additional information on what specific coordinates it represents; instead, since the size is always the same and the POI is always in the centre, the model gets consistent information on the measurements at a location near the POI and how it affects the target, without any explicit knowledge of where that location is in coordinate system terms.

## Development

You can install **ocf-data-sampler** for development as follows:

``` 
pip install git+https://github.com/openclimatefix/ocf-data-sampler.git
```

### Running the test suite

The tests in this project use `pytest`. Once you have it installed, you can run it from the project's directory:

```
cd ocf-data-sampler
pytest
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

