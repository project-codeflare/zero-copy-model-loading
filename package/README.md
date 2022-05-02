# `zerocopy` Python package

This directory contains the source code for the `zerocopy` Python package.

## Release instructions:

1. `conda activate ../env`
1. `python setup.py sdist bdist_wheel`
1. (Optional test upload) `python -m twine upload --repository testpypi dist/*`
1. `python -m twine upload dist/*`

## Testing

There are currently no automated tests. Run the code in
`../notebooks/benchmark` manually to test prior to each release.

