API Reference
=============

Start at :class:`rlemasklib.RLEMask` to explore the object-oriented API.

.. toctree::
   :titlesonly:

   {% for page in pages|selectattr("is_top_level_object") %}
   {{ page.include_path }}
   {% endfor %}
