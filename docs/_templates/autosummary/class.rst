{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

{#- A `TypedDict` subclasses `dict`, so autodoc reports the whole mapping API as inherited
    members. Those are not part of the documented surface. #}
{%- set dict_api = ['clear', 'copy', 'fromkeys', 'get', 'items', 'keys', 'pop',
                    'popitem', 'setdefault', 'update', 'values'] %}
{%- set own_methods = methods | reject('in', dict_api) | reject('eq', '__init__') | list %}
{%- set dunders = all_methods | select('eq', '__call__') | list %}
{%- set shown_methods = own_methods + dunders %}

{#- Attributes render inline, with their type and their own docstring. A summary table would
    print the names untyped, and give every field its own page and sidebar entry -- which for
    a parameter bag is a page saying one sentence. Methods keep the table: they are
    substantial enough to be worth a page each. #}
.. autoclass:: {{ objname }}
{%- if attributes %}
    :members: {{ attributes | join(', ') }}
    :undoc-members:
    :exclude-members: {{ dict_api | join(', ') }}
{%- endif %}
    {% block methods %}
    {%- if shown_methods %}
    .. rubric:: {{ _('Methods') }}

    .. autosummary::
        :toctree: .
    {% for item in shown_methods %}
        ~{{ name }}.{{ item }}
    {%- endfor %}
    {%- endif %}
    {%- endblock %}
