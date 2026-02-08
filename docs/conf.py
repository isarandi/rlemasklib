import contextlib
import importlib
import inspect
import os
import re
import sys
from enum import Enum

import setuptools_scm
import toml

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "_ext")))


pyproject_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
)

with open(pyproject_path) as f:
    data = toml.load(f)

project_info = data["project"]
project_slug = project_info["name"].replace(" ", "-").lower()
tool_urls = project_info.get("urls", {})

repo_url = tool_urls.get("Repository", "")
author_url = tool_urls.get("Author", "")
github_username = re.match(r"https://github\.com/([^/]+)/?", repo_url)[1]

project = project_info["name"]
release = setuptools_scm.get_version("..")
version = ".".join(release.split(".")[:2])
main_module_name = project_slug.replace("-", "_")
repo_name = project_slug
module = importlib.import_module(main_module_name)
globals()[main_module_name] = module


# -- Project information -----------------------------------------------------
linkcode_url = repo_url

author = project_info["authors"][0]["name"]
copyright = "2023-%Y"

# -- General configuration ---------------------------------------------------
add_module_names = False
python_use_unqualified_type_names = True
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.autodoc.typehints",
    "sphinxcontrib.bibtex",
    "autoapi.extension",
    "sphinx.ext.inheritance_diagram",
    "mask_grid",
    "sphinx_codeautolink",
]
bibtex_bibfiles = ["abbrev_long.bib", "references.bib"]
bibtex_footbibliography_header = ".. rubric:: References"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/main/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

github_username = github_username
github_repository = repo_name
autodoc_show_sourcelink = False
html_show_sourcelink = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
python_display_short_literal_types = True

html_title = project
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_toc_level": 3,
    "icon_links": [
        {
            "name": "GitHub",
            "url": repo_url,
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }
    ],
    # "footer_items": ["copyright"],
}
html_static_path = ["_static"]
html_css_files = ["styles/my_theme.css", "styles/mask_grid.css"]

html_context = {
    "author_url": author_url,
    "author": author,
}

toc_object_entries_show_parents = "hide"

autoapi_root = "api"
autoapi_member_order = "bysource"
autodoc_typehints = "description"
autoapi_own_page_level = "attribute"
autoapi_type = "python"
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "undoc-members": False,
    "exclude-members": "__init__, __weakref__, __repr__, __str__",
}
autoapi_options = [
    "members",
    "show-inheritance",
    "special-members",
    "show-module-summary",
]
autoapi_add_toctree_entry = True
autoapi_dirs = ["../src"]
autoapi_template_dir = "_templates/autoapi"

# C source links for methods with C implementations
C_SOURCE_MAP = {
    "rlemasklib.RLEMask.area": ("moments.c", "rleArea"),
    "rlemasklib.RLEMask.centroid": ("moments.c", "rleCentroid"),
    "rlemasklib.RLEMask.moments": ("moments.c", "rleMoments"),
    "rlemasklib.RLEMask.hu_moments": ("moments.c", "rleHuMoments"),
    "rlemasklib.RLEMask.nonzero": ("moments.c", "rleNonZeroIndices"),
    "rlemasklib.RLEMask.connected_components": (
        "connected_components.c",
        "rleConnectedComponents",
    ),
    "rlemasklib.RLEMask.connected_component_stats": (
        "connected_components_refactor.c",
        "rleConnectedComponentStats",
    ),
    "rlemasklib.RLEMask.connected_components_with_stats": (
        "connected_components_refactor.c",
        "rleConnectedComponentStats",
    ),
    "rlemasklib.RLEMask.count_connected_components": (
        "connected_components_refactor.c",
        "rleCountConnectedComponents",
    ),
    "rlemasklib.RLEMask.largest_interior_rectangle": (
        "largest_interior_rectangle.c",
        "rleLargestInteriorRectangle",
    ),
    "rlemasklib.RLEMask.largest_interior_rectangle_around": (
        "largest_interior_rectangle.c",
        "rleLargestInteriorRectangleAroundCenter",
    ),
    "rlemasklib.RLEMask.bbox": ("misc.c", "rleToBbox"),
    "rlemasklib.RLEMask.iou": ("iou_nms.c", "rleIou"),
    "rlemasklib.RLEMask.iou_matrix": ("iou_nms.c", "rleIou"),
}

C_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "rlemasklib", "c")
_c_func_lines_cache = {}


def get_c_func_lines(c_file, func_name):
    """Find the line range of a C function in a file."""
    cache_key = (c_file, func_name)
    if cache_key in _c_func_lines_cache:
        return _c_func_lines_cache[cache_key]

    filepath = os.path.join(C_SRC_DIR, c_file)
    if not os.path.exists(filepath):
        return None

    with open(filepath) as f:
        lines = f.readlines()

    # Find function start: line containing "funcname("
    start = None
    for i, line in enumerate(lines, 1):
        if f"{func_name}(" in line and not line.strip().startswith("//"):
            start = i
            break

    if start is None:
        return None

    # Find function end: next line that is just "}"
    end = start
    for i, line in enumerate(lines[start:], start + 1):
        if line.rstrip() == "}":
            end = i
            break

    _c_func_lines_cache[cache_key] = (start, end)
    return start, end


def get_c_source_link(fullname):
    if fullname not in C_SOURCE_MAP:
        return None
    c_file, c_func = C_SOURCE_MAP[fullname]
    lines = get_c_func_lines(c_file, c_func)
    if lines:
        url = f"{repo_url}/blob/v{release}/src/rlemasklib/c/{c_file}#L{lines[0]}-L{lines[1]}"
    else:
        url = f"{repo_url}/blob/v{release}/src/rlemasklib/c/{c_file}"
    return c_file, c_func, url


def add_c_source_links(app, pagename, templatename, context, doctree):
    """Post-process HTML to add [C source] links next to [source] links."""
    if "body" not in context:
        return

    body = context["body"]
    for py_name, (c_file, c_func) in C_SOURCE_MAP.items():
        # Build the anchor ID from the Python name (e.g., rlemasklib.RLEMask.area)
        anchor_id = py_name

        # Check if this method is on the page
        if f'id="{anchor_id}"' not in body:
            continue

        # Get line range and build URL
        lines = get_c_func_lines(c_file, c_func)
        if lines:
            url = f"{repo_url}/blob/v{release}/src/rlemasklib/c/{c_file}#L{lines[0]}-L{lines[1]}"
        else:
            url = f"{repo_url}/blob/v{release}/src/rlemasklib/c/{c_file}"

        # Build the [C source] link HTML (same structure as [source])
        c_link = (
            f'<a class="reference external" href="{url}">'
            f'<span class="viewcode-link"><span class="pre">[C source]</span></span></a>'
        )

        # Insert after the existing [source] link for this method
        # Pattern: find the dt with this id, then insert after its [source] link
        import re

        pattern = rf'(id="{re.escape(anchor_id)}"[^>]*>.*?<span class="pre">\[source\]</span></span></a>)'
        replacement = rf"\1{c_link}"
        body = re.sub(pattern, replacement, body, count=1, flags=re.DOTALL)

    context["body"] = body


autodoc_member_order = "bysource"
autoclass_content = "class"

autosummary_generate = True
autosummary_imported_members = False


def autodoc_skip_member(app, what, name, obj, skip, options):
    """
    Skip members (functions, classes, modules) without docstrings.
    """
    # Check if the object has a __doc__ attribute
    if not getattr(obj, "docstring", None):
        print("no docstring", name)
        return True  # Skip if there's no docstring
    elif what in ("class", "function", "attribute"):
        # Check if the module of the class has a docstring
        print("checking module", name)
        module_name = ".".join(name.split(".")[:-1])

        try:
            module = importlib.import_module(module_name)
            return not getattr(module, "__doc__", None)
        except ModuleNotFoundError as e:
            print("module not found", module_name, str(e))
            return None


def linkcode_resolve(domain, info):
    if domain != "py":
        return None

    file, start, end = get_line_numbers(eval(info["fullname"]))
    relpath = os.path.relpath(file, os.path.dirname(module.__file__))
    return (
        f"{repo_url}/blob/v{release}/src/{main_module_name}/{relpath}#L{start}-L{end}"
    )


def get_line_numbers(obj):
    if isinstance(obj, property):
        obj = obj.fget

    if isinstance(obj, Enum):
        return get_enum_member_line_numbers(obj)

    with module_restored(obj):
        lines = inspect.getsourcelines(obj)
        file = inspect.getsourcefile(obj)

    start, end = lines[1], lines[1] + len(lines[0]) - 1
    return file, start, end


def get_enum_member_line_numbers(obj):
    class_ = obj.__class__
    with module_restored(class_):
        source_lines, start_line = inspect.getsourcelines(class_)

        for i, line in enumerate(source_lines):
            if f"{obj.name} =" in line:
                return inspect.getsourcefile(class_), start_line + i, start_line + i
        else:
            raise ValueError(f"Enum member {obj.name} not found in {class_}")


@contextlib.contextmanager
def module_restored(obj):
    if not hasattr(obj, "_module_original_"):
        yield
    else:
        fake_module = obj.__module__
        obj.__module__ = obj._module_original_
        yield
        obj.__module__ = fake_module


def setup(app):
    app.connect("autoapi-skip-member", autodoc_skip_member)
    app.connect("autodoc-skip-member", autodoc_skip_member)
    app.connect("html-page-context", add_c_source_links)
