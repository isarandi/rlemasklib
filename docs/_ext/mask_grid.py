"""
Sphinx extension for rendering binary masks as HTML grids.

Usage in RST (horizontal layout - masks side by side):

    .. mask-demo::

       [A] | [B] == [C]

       [A]:    [B]:    [C]:
       .##.    ..##    .###
       ####    ..##    ####
       .##.    ..##    .###

Characters:
    '#', 'X', '1', '*' -> foreground (filled)
    '.', '0'           -> background (empty)
"""

import re

from docutils import nodes
from docutils.parsers.rst import Directive, directives

FOREGROUND_CHARS = {"#", "X", "1", "*"}
BACKGROUND_CHARS = {".", "0"}
MASK_CHARS = FOREGROUND_CHARS | BACKGROUND_CHARS


def parse_mask_pattern(text):
    """Parse ASCII mask pattern into 2D list of booleans."""
    lines = [line for line in text.strip().split("\n") if line.strip()]

    grid = []
    for line in lines:
        row = []
        for char in line:
            if char in FOREGROUND_CHARS:
                row.append(True)
            elif char in BACKGROUND_CHARS:
                row.append(False)
            # Skip other characters
        if row:
            grid.append(row)

    return grid


def parse_horizontal_masks(lines):
    """Parse multiple masks defined side-by-side.

    Format:
        [A]:    [B]:    [C]:
        .##.    ..##    .###
        ####    ..##    ####
        .##.    ..##    .###

    Returns dict mapping name -> grid.
    """
    if not lines:
        return {}

    # First line should contain [name]: headers
    header_line = lines[0]
    header_pattern = re.compile(r"\[(\w+)\]:")

    # Find all headers and their positions
    headers = []
    for match in header_pattern.finditer(header_line):
        headers.append((match.group(1), match.start(), match.end()))

    if not headers:
        return {}

    # For each header, find the column range it covers
    # The range extends from header start to next header start (or end of line)
    column_ranges = []
    for i, (name, start, end) in enumerate(headers):
        if i + 1 < len(headers):
            col_end = headers[i + 1][1]
        else:
            col_end = max(len(line) for line in lines)
        column_ranges.append((name, start, col_end))

    # Extract mask data for each header
    masks = {}
    for name, col_start, col_end in column_ranges:
        grid = []
        for line in lines[1:]:  # Skip header line
            # Extract the portion of this line for this mask
            if col_start < len(line):
                segment = line[col_start:col_end]
                row = []
                for char in segment:
                    if char in FOREGROUND_CHARS:
                        row.append(True)
                    elif char in BACKGROUND_CHARS:
                        row.append(False)
                if row:
                    grid.append(row)
        if grid:
            masks[name] = grid

    return masks


def mask_to_inline_html(grid, cell_size=20, rect=None):
    """Convert boolean grid to inline HTML div structure.

    Args:
        grid: 2D list of booleans
        cell_size: size of each cell in pixels
        rect: optional (x, y, w, h) tuple for rectangle border overlay
    """
    if not grid:
        return '<span class="mask-grid-inline-empty"></span>'

    height = len(grid)
    width = len(grid[0]) if grid else 0

    html_parts = [
        f'<span class="mask-grid-inline" style="'
        f"display: inline-grid; "
        f"grid-template-columns: repeat({width}, {cell_size}px); "
        f'grid-template-rows: repeat({height}, {cell_size}px);">'
    ]

    for row_idx, row in enumerate(grid):
        for col_idx, cell in enumerate(row):
            cell_class = "mask-cell-fg" if cell else "mask-cell-bg"

            # Add border classes if this cell is on the edge of the rect
            if rect:
                rx, ry, rw, rh = rect
                border_classes = []
                # Check if cell is within rect bounds
                in_rect_cols = rx <= col_idx < rx + rw
                in_rect_rows = ry <= row_idx < ry + rh

                if in_rect_cols and in_rect_rows:
                    if row_idx == ry:  # Top edge
                        border_classes.append("rect-border-top")
                    if row_idx == ry + rh - 1:  # Bottom edge
                        border_classes.append("rect-border-bottom")
                    if col_idx == rx:  # Left edge
                        border_classes.append("rect-border-left")
                    if col_idx == rx + rw - 1:  # Right edge
                        border_classes.append("rect-border-right")

                if border_classes:
                    cell_class += " " + " ".join(border_classes)

            html_parts.append(f'<span class="mask-cell {cell_class}"></span>')

    html_parts.append("</span>")
    return "".join(html_parts)


class MaskDemoDirective(Directive):
    """Directive for flexible mask expressions with named masks.

    Example:
        .. mask-demo::

           [A].erode() == [B]

           [A]:
           .##.
           ####
           .##.

           [B]:
           ....
           .##.
           ....
    """

    has_content = True
    optional_arguments = 0
    option_spec = {
        "caption": directives.unchanged,
        "cell-size": directives.positive_int,
    }

    def run(self):
        content = "\n".join(self.content)
        cell_size = self.options.get("cell-size", 20)

        # Parse mask definitions and template
        masks, template_lines = self._parse_content(content)

        # Build HTML by replacing [name] with mask HTML
        html_lines = []
        for line in template_lines:
            html_line = self._substitute_masks(line, masks, cell_size)
            html_lines.append(html_line)

        template_html = "<br>".join(html_lines) if html_lines else ""
        html = f'<div class="mask-demo">{template_html}</div>'

        caption = self.options.get("caption")
        if caption:
            html = f'<figure class="mask-figure">{html}<figcaption>{caption}</figcaption></figure>'

        raw_node = nodes.raw("", html, format="html")
        return [raw_node]

    def _parse_content(self, content):
        """Parse content into mask definitions and template lines.

        Supports two formats:

        Horizontal (masks side-by-side):
            [A] | [B] == [C]

            [A]:    [B]:    [C]:
            .##.    ..##    .###
            ####    ..##    ####

        Vertical (masks stacked):
            [A] | [B] == [C]

            [A]:
            .##.
            ####

            [B]:
            ..##
            ..##
        """
        lines = content.split("\n")
        template_lines = []
        definition_lines = []
        in_definitions = False

        # Pattern for horizontal format: line with multiple [name]: headers
        horiz_header_pattern = re.compile(r"\[(\w+)\]:.*\[(\w+)\]:")
        # Pattern for vertical format: line with single [name]:
        vert_def_pattern = re.compile(r"^\s*\[(\w+)\]:\s*$")

        for line in lines:
            if not in_definitions:
                # Check if this starts horizontal definitions
                if horiz_header_pattern.search(line):
                    in_definitions = True
                    definition_lines.append(line)
                # Check if this starts vertical definitions
                elif vert_def_pattern.match(line):
                    in_definitions = True
                    definition_lines.append(line)
                elif line.strip():
                    template_lines.append(line.strip())
            else:
                definition_lines.append(line)

        # Determine format and parse accordingly
        if definition_lines:
            first_def_line = definition_lines[0]
            if horiz_header_pattern.search(first_def_line):
                # Horizontal format
                masks = parse_horizontal_masks(definition_lines)
            else:
                # Vertical format
                masks = self._parse_vertical_masks(definition_lines)
        else:
            masks = {}

        return masks, template_lines

    def _parse_vertical_masks(self, lines):
        """Parse vertically stacked mask definitions."""
        masks = {}
        current_mask_name = None
        current_mask_lines = []
        vert_def_pattern = re.compile(r"^\s*\[(\w+)\]:\s*$")

        for line in lines:
            match = vert_def_pattern.match(line)
            if match:
                # Save previous mask if any
                if current_mask_name and current_mask_lines:
                    masks[current_mask_name] = parse_mask_pattern(
                        "\n".join(current_mask_lines)
                    )
                current_mask_name = match.group(1)
                current_mask_lines = []
            elif current_mask_name is not None:
                current_mask_lines.append(line)

        # Save last mask
        if current_mask_name and current_mask_lines:
            masks[current_mask_name] = parse_mask_pattern("\n".join(current_mask_lines))

        return masks

    def _substitute_masks(self, line, masks, cell_size):
        """Replace [name] or [name(x,y,w,h)] references with inline mask HTML."""
        # Pattern to match [name] or [name(x,y,w,h)] references
        ref_pattern = re.compile(r"\[(\w+)(?:\((\d+),(\d+),(\d+),(\d+)\))?\]")

        def replace_mask(match):
            name = match.group(1)
            if name in masks:
                # Check if rect coordinates are specified
                if match.group(2) is not None:
                    rect = (
                        int(match.group(2)),
                        int(match.group(3)),
                        int(match.group(4)),
                        int(match.group(5)),
                    )
                    return mask_to_inline_html(masks[name], cell_size, rect=rect)
                else:
                    return mask_to_inline_html(masks[name], cell_size)
            else:
                # Unknown mask, leave as-is
                return match.group(0)

        # Split into parts and process
        result_parts = []
        last_end = 0

        for match in ref_pattern.finditer(line):
            # Add text before this match
            if match.start() > last_end:
                text = line[last_end : match.start()]
                result_parts.append(f'<span class="mask-demo-text">{text}</span>')
            # Add the mask
            result_parts.append(replace_mask(match))
            last_end = match.end()

        # Add remaining text
        if last_end < len(line):
            text = line[last_end:]
            result_parts.append(f'<span class="mask-demo-text">{text}</span>')

        return "".join(result_parts)


def setup(app):
    """Register the directive with Sphinx."""
    app.add_directive("mask-demo", MaskDemoDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
