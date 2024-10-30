# drum-dev

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/XinmingTu/drum-dev/test.yaml?branch=main
[link-tests]: https://github.com/XinmingTu/drum-dev/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/drum-dev

Drum is a sequence-based model based on single cell multi-omics data

## Getting started

Please refer to the [documentation][link-docs]. In particular, the

-   [API documentation][link-api].

## Installation

You need to have Python 3.9 or newer installed on your system. If you don't have
Python installed, we recommend installing [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge).

There are several alternative options to install drum-dev:

<!--
1) Install the latest release of `drum-dev` from `PyPI <https://pypi.org/project/drum-dev/>`_:

```bash
pip install drum-dev
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/XinmingTu/drum-dev.git@main
```

2. Set up environment and then install the package:

```bash
conda env create -f environment.yml
conda activate drum-dev
pip install -e .
```

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/XinmingTu/drum-dev/issues
[changelog]: https://drum-dev.readthedocs.io/latest/changelog.html
[link-docs]: https://drum-dev.readthedocs.io
[link-api]: https://drum-dev.readthedocs.io/latest/api.html
