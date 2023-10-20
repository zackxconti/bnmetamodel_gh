import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "bnmetamodel_gh"
copyright = "2023, Zack Xuereb Conti"
author = "Zack Xuereb Conti"
release = "0.0.1"

# -- General configuration ---------------------------------------------------

sys.path.append(os.path.abspath("../"))

extensions = [
    "nbsphinx",
    # "autoapi.extension",  # TODO: activate after Python 3 migration
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autoapi_dirs = ["../bnmetamodel_gh"]
autoapi_root = "api"

nbsphinx_execute = "always"
nbsphinx_allow_errors = True

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
