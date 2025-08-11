# ocf-data-sampler
 
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-14-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![tags badge](https://img.shields.io/github/v/tag/openclimatefix/ocf-data-sampler?include_prereleases&sort=semver&color=FFAC5F)](https://github.com/openclimatefix/ocf-data-sampler/tags)
[![ease of contribution: easy](https://img.shields.io/badge/ease%20of%20contribution:%20easy-32bd50)](https://github.com/openclimatefix#how-easy-is-it-to-get-involved) 

**ocf-data-sampler** contains all the tools needed to create samples and feed them to our models, such as [PVNet](https://github.com/openclimatefix/PVNet/). The data we work with—typically energy data, satellite imagery, and numerical weather predictions (NWPs)—is usually too heavy to do this on the fly, so that's where this repo comes in: handling steps like opening the data, selecting the right samples, normalising and reshaping, and saving to and reading from disk.

We are currently migrating to this repo from [ocf_datapipes](https://github.com/openclimatefix/ocf_datapipes/), which performs the same functions but is built around `PyTorch DataPipes`, which are quite cumbersome to work with and are no longer maintained by PyTorch. **ocf-data-sampler** uses `PyTorch Datasets`, and we've taken the opportunity to make the code much cleaner and more manageable.

> [!Note]
> This repository is still in early development development and large changes to the user facing functions may still occur.

## Licence 

This project is primarily licensed under the MIT License (see LICENSE).

It includes and adapts internal functions from the Google xarray-tensorstore project, licensed under the Apache License, Version 2.0.

## Documentation

**ocf-data-sampler** doesn't have external documentation _yet_; you can read a bit about how our torch datasets work in the README [here](ocf_data_sampler/torch_datasets/README.md).

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


## Contributors

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/dfulu"><img src="https://avatars.githubusercontent.com/u/41546094?v=4?s=100" width="100px;" alt="James Fulton"/><br /><sub><b>James Fulton</b></sub></a><br /><a href="https://github.com/openclimatefix/ocf-data-sampler/commits?author=dfulu" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/AUdaltsova"><img src="https://avatars.githubusercontent.com/u/43303448?v=4?s=100" width="100px;" alt="Alexandra Udaltsova"/><br /><sub><b>Alexandra Udaltsova</b></sub></a><br /><a href="https://github.com/openclimatefix/ocf-data-sampler/commits?author=AUdaltsova" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Sukh-P"><img src="https://avatars.githubusercontent.com/u/42407101?v=4?s=100" width="100px;" alt="Sukhil Patel"/><br /><sub><b>Sukhil Patel</b></sub></a><br /><a href="https://github.com/openclimatefix/ocf-data-sampler/commits?author=Sukh-P" title="Code">💻</a> <a href="https://github.com/openclimatefix/ocf-data-sampler/issues?q=author%3ASukh-P" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/peterdudfield"><img src="https://avatars.githubusercontent.com/u/34686298?v=4?s=100" width="100px;" alt="Peter Dudfield"/><br /><sub><b>Peter Dudfield</b></sub></a><br /><a href="https://github.com/openclimatefix/ocf-data-sampler/commits?author=peterdudfield" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/VikramsDataScience"><img src="https://avatars.githubusercontent.com/u/45002417?v=4?s=100" width="100px;" alt="Vikram Pande"/><br /><sub><b>Vikram Pande</b></sub></a><br /><a href="https://github.com/openclimatefix/ocf-data-sampler/commits?author=VikramsDataScience" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/SophiaLi20"><img src="https://avatars.githubusercontent.com/u/163532536?v=4?s=100" width="100px;" alt="Unnati Bhardwaj"/><br /><sub><b>Unnati Bhardwaj</b></sub></a><br /><a href="https://github.com/openclimatefix/ocf-data-sampler/commits?author=SophiaLi20" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/alirashidAR"><img src="https://avatars.githubusercontent.com/u/110668489?v=4?s=100" width="100px;" alt="Ali Rashid"/><br /><sub><b>Ali Rashid</b></sub></a><br /><a href="https://github.com/openclimatefix/ocf-data-sampler/commits?author=alirashidAR" title="Code">💻</a></td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/felix-e-h-p"><img src="https://avatars.githubusercontent.com/u/137530077?v=4?s=100" width="100px;" alt="Felix"/><br /><sub><b>Felix</b></sub></a><br /><a href="https://github.com/openclimatefix/ocf-data-sampler/commits?author=felix-e-h-p" title="Code">💻</a> <a href="https://github.com/openclimatefix/ocf-data-sampler/commits?author=felix-e-h-p" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://timothyajaniportfolio-b6v3zq29k-timthegreat.vercel.app/"><img src="https://avatars.githubusercontent.com/u/60073728?v=4?s=100" width="100px;" alt="Ajani Timothy"/><br /><sub><b>Ajani Timothy</b></sub></a><br /><a href="https://github.com/openclimatefix/ocf-data-sampler/commits?author=Tim1119" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://rupeshmangalam.vercel.app/"><img src="https://avatars.githubusercontent.com/u/91172425?v=4?s=100" width="100px;" alt="Rupesh Mangalam"/><br /><sub><b>Rupesh Mangalam</b></sub></a><br /><a href="https://github.com/openclimatefix/ocf-data-sampler/commits?author=RupeshMangalam21" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://siddharth7113.github.io"><img src="https://avatars.githubusercontent.com/u/114160268?v=4?s=100" width="100px;" alt="Siddharth"/><br /><sub><b>Siddharth</b></sub></a><br /><a href="https://github.com/openclimatefix/ocf-data-sampler/commits?author=siddharth7113" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/Sachin-G13"><img src="https://avatars.githubusercontent.com/u/190184500?v=4?s=100" width="100px;" alt="Sachin-G13"/><br /><sub><b>Sachin-G13</b></sub></a><br /><a href="https://github.com/openclimatefix/ocf-data-sampler/commits?author=Sachin-G13" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://drona-gyawali.github.io/"><img src="https://avatars.githubusercontent.com/u/170401554?v=4?s=100" width="100px;" alt="Dorna Raj Gyawali"/><br /><sub><b>Dorna Raj Gyawali</b></sub></a><br /><a href="https://github.com/openclimatefix/ocf-data-sampler/commits?author=drona-gyawali" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/adnanhashmi25"><img src="https://avatars.githubusercontent.com/u/55550094?v=4?s=100" width="100px;" alt="Adnan Hashmi"/><br /><sub><b>Adnan Hashmi</b></sub></a><br /><a href="https://github.com/openclimatefix/ocf-data-sampler/commits?author=adnanhashmi25" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

---

*Part of the [Open Climate Fix](https://github.com/orgs/openclimatefix/people) community.*

[![OCF Logo](https://cdn.prod.website-files.com/62d92550f6774db58d441cca/6324a2038936ecda71599a8b_OCF_Logo_black_trans.png)](https://openclimatefix.org)

