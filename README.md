# ocf-data-sampler
 
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-7-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![workflows badge](https://img.shields.io/github/actions/workflow/status/openclimatefix/ocf-data-sampler/release.yaml?branch=maine&color=FFD053&label=workflow)](https://github.com/openclimatefix/ocf-data-sampler/actions/workflows/workflows.yaml)
[![tags badge](https://img.shields.io/github/v/tag/openclimatefix/ocf-data-sampler?include_prereleases&sort=semver&color=FFAC5F)](https://github.com/openclimatefix/ocf-data-sampler/tags)
[![ease of contribution: easy](https://img.shields.io/badge/ease%20of%20contribution:%20easy-32bd50)](https://github.com/openclimatefix#how-easy-is-it-to-get-involved) 

**ocf-data-sampler** contains all the infrastructure needed to 
create batches and feed them to our models, such as 
[PVNet](https://github.com/openclimatefix/PVNet/). The data we work 
with is usually too heavy to do this on the fly, so that's where this repo
comes in: handling steps like opening the data, selecting the right
samples, normalising and reshaping, and saving to and reading 
from disk.

We are currently migrating to this repo from [ocf_datapipes](https://github.com/openclimatefix/ocf_datapipes/), which 
has performed the same functions but was centered around PyTorch DataPipes, 
which were quite cumbersome to work with and are no longer maintained by
PyTorch. **ocf-data-sampler** uses PyTorch Datasets, and we've
taken the opportunity to make the code much cleaner and more manageable.

> [!Note]
> This repository is still in the development stage and does not yet have the full 
> functionality of its predecessor, [ocf_datapipes](https://github.com/openclimatefix/ocf_datapipes/).
> It might not be ready for use out-of-the-box! So we would really appreciate any help to let us make the transition faster.

## Documentation

> [!Note] Side note: I'd like to put some high-level overview here of what dsampler does, but open to suggestions


## FAQ

If you have any questions about this or any other of our repos,
don't hesitate to hop to our [Discussions Page](https://github.com/orgs/openclimatefix/discussions)!

### How does ocf-data-sampler deal with data sources using different projections (e.g., some are in latitude-longitude, and some in OSGB)?

[Clever and concise answer here]

I don't like the FAQ having less than 2 q-ns 
and also not sure this should go here, 
but open to suggestions as well



## Development

You can install **ocf-data-sampler** for development as follows:

``` 
pip install git+https://github.com/openclimatefix/ocf-data-sampler.git
```

### Running the test suite

The tests in this project use `pytest`. Once you have it installed, 
you can run it from the project's directory:

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


## Contributors

---

*Part of the [Open Climate Fix](https://github.com/orgs/openclimatefix/people) community.*

[![OCF Logo](https://cdn.prod.website-files.com/62d92550f6774db58d441cca/6324a2038936ecda71599a8b_OCF_Logo_black_trans.png)](https://openclimatefix.org)

