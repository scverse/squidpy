{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

{#- A `TypedDict` subclasses `dict` and a `NamedTuple` subclasses `tuple`, so autodoc reports
    each container's whole API as inherited members. None of it is the documented surface. #}
{%- set dict_api = ['clear', 'copy', 'fromkeys', 'get', 'items', 'keys', 'pop',
                    'popitem', 'setdefault', 'update', 'values'] %}
{%- set tuple_api = ['count', 'index'] %}
{%- set inherited_api = dict_api + tuple_api %}
{%- set own_methods = methods | reject('in', inherited_api) | reject('eq', '__init__') | list %}
{%- set dunders = all_methods | select('eq', '__call__') | list %}
{%- set shown_methods = own_methods + dunders %}

{#- Attributes render inline, with their type and their own docstring; methods keep the table
    and a page each. `:members:` is already an allowlist of this class's own attributes, so no
    `:exclude-members:` -- that subtracts from the allowlist, and would silently drop a field
    whose name collides with the container API, such as a params key called `copy`.
    `bysource` because for a NamedTuple the order is the unpacking contract, and the
    project-wide `autodoc_member_order` is alphabetical. #}
.. autoclass:: {{ objname }}
{%- if attributes %}
    :members: {{ attributes | join(', ') }}
    :undoc-members:
    :member-order: bysource
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
