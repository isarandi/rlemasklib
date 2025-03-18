from docutils import nodes
from sphinx.util.docutils import SphinxDirective

def setup(app):
    app.add_directive("groupedautoapisummary", GroupedAutoAPISummary)
    app.connect("env-before-read-docs", collect_grouped_functions)
    app.connect("doctree-resolved", insert_grouped_tables)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}


class GroupedAutoAPISummary(SphinxDirective):
    """
    Custom directive to group AutoAPI functions by tags in docstrings.
    """
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {}

    def run(self):
        # Placeholder node to mark where the grouped tables should be inserted
        return [GroupedTablesPlaceholder()]


class GroupedTablesPlaceholder(nodes.General, nodes.Element):
    pass


def collect_grouped_functions(app, env, docnames):
    """
    Collect functions and their `:group:` tags during the build process.
    """
    if not hasattr(env, "grouped_functions"):
        env.grouped_functions = {}

    print(dir(app.env))
    for docname, obj in app.env.autoapi_objects.items():
        group = extract_group_from_docstring(obj.__doc__)
        if group:
            if group not in env.grouped_functions:
                env.grouped_functions[group] = []
            env.grouped_functions[group].append(obj["full_name"])


def extract_group_from_docstring(docstring):
    """
    Extract the `:group:` tag from the docstring.
    """
    if not docstring:
        return None
    for line in docstring.splitlines():
        if line.strip().startswith(":group:"):
            return line.split(":group:")[1].strip()
    return None


def insert_grouped_tables(app, doctree, docname):
    """
    Insert grouped function tables at the placeholder nodes.
    """
    if not hasattr(app.env, "grouped_functions"):
        return

    env = app.env
    for node in doctree.traverse(GroupedTablesPlaceholder):
        for group, functions in env.grouped_functions.items():
            # Create a section for each group
            section = nodes.section(ids=[group], names=[group])
            section += nodes.title(text=f"{group} Functions")

            # Create a table for the group
            table = nodes.table()
            tgroup = nodes.tgroup(cols=1)
            table += tgroup
            tgroup += nodes.colspec(colwidth=1)

            tbody = nodes.tbody()
            tgroup += tbody

            for func in functions:
                row = nodes.row()
                entry = nodes.entry()
                entry += nodes.paragraph(text=func)
                row += entry
                tbody += row

            section += table
            node.replace_self(section)

