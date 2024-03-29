# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../hymd/'))
from hymd import __version__  # noqa: E402


# -- Project information -----------------------------------------------------

project = 'hymd'
author = (
    'Morten Ledum, Xinmeng Li, Samiran Sen, Manuel Carrer, Sigbjørn Løland Bore'  # noqa: E501
)

copyright = f'2021, {author}'

# The full version, including alpha/beta/rc tags
version = __version__
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'sphinxcontrib.bibtex',
    'sphinx_tabs.tabs',
    'sphinx_panels',
    'sphinx.ext.intersphinx',
    'matplotlib.sphinxext.plot_directive',
    'numpydoc',
    "sphinx_multiversion",
]

# Bibtex configuration.
bibtex_bibfiles = ['ref.bib']
bibtex_encoding = 'latin'
bibtex_reference_style = 'author_year'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'collapse_navigation': False,
}
html_logo = 'img/hymd_logl_white_text_abbr.png'
html_favicon = 'img/hymd_icon_white.png'
html_show_sphinx = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css']

html_context = {
    "sidebar_external_links_caption": "Links",
    "sidebar_external_links": [
        (
            '<i class="fa fa-cube fa-fw"></i> PyPI',
            f"https://pypi.org/project/{project.lower()}",
        ),
        (
            '<i class="fa fa-code fa-fw"></i> Source code',
            f"https://github.com/Cascella-Group-UiO/{project.lower()}",
        ),
        (
            '<i class="fa fa-bug fa-fw"></i> Issue tracker',
            f"https://github.com/Cascella-Group-UiO/{project.lower()}/issues/",
        ),
        (
            '<i class="fa fa-file-text fa-fw"></i> Citation',
            "about:blank",
        ),
    ],
}

html_sidebars = {
    '**': [
        'versions.html',
    ],
}

numpydoc_show_class_members = False

# whitelists tags and branch to build the docs for
smv_tag_whitelist = r'^v\d+\.\d+.\d+$' # vX.Y.Z

smv_branch_whitelist = r'^(?!gh-pages|joss-paper-final).*$' # all branches except gh-pages

