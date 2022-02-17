## Contributor's Guidelines
Reporting bugs and any problem you encounter with HyMD is the most important contribution for us.  
You can do that by submitting them (along with feature requests or proposals) on the
[issue tracker](https://github.com/Cascella-Group-UiO/HyMD/issues).  

If you want to contribute your own code, this short guide takes you through the
necessary steps.  

1. Getting started  
Fork the repository from [here](https://github.com/Cascella-Group-UiO/hymd).
Clone the fork on your local machine and install the dependencies following the
instructions given in the [documentation](https://cascella-group-uio.github.io/HyMD/doc_pages/installation.html#dependencies).   

Finally install the package in development mode using
```bash
cd HyMD
pip install -e .
```

2. Contribute  
Write your code, test it (we welcome setting up tests with [pytest](https://docs.pytest.org/en/7.0.x/)) and follow the instructions
[here](https://cascella-group-uio.github.io/HyMD/doc_pages/overview.html#running-parallel-simulations) to make sure it runs correctly.
When you feel ready you can open a pull request to merge your contributions in the main HyMD branch. 

If you are contributing new FORTRAN force calculation routines, you need to include the files in
the `Makefile` and inside the `force_kernel` extension in `setup.py` so they can be compiled with
[f2py](https://numpy.org/doc/stable/f2py/) when running `pip install`.
We also require you provide both a single and double precision version of the FORTRAN routines.

To import them you simply call
```python3
from force_kernel import <routine_name>
```
and use the imported module to manipulate NumPy arrays.

## Code style
For python files we follow the [PEP8](https://www.python.org/dev/peps/pep-0008/) guidelines, so please adapt to them when implementing new code. 

The most important stylistic requirements are:

    use 4 spaces for indentation instead of tabs
    wrap lines to max 80 characters whenever possible
    name variables and functions all lowercase with underscore as a separator (e.g. some_variable)
    name classes with starting letters capitalized and no separator (e.g. SomeClass)

While for FORTRAN the basic requirements are:

    use 2 spaces for indentation
    wrap lines to max 80 characters whenever possible
    do not use GOTO statements
    use lowercase for commands and functions

