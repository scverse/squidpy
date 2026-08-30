{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

{#- TypedDict params classes subclass `dict`, so autodoc reports the whole
    mapping API as inherited members. Those are not part of the documented
    surface: list only what the class itself declares. #}
{%- set dict_api = ['clear', 'copy', 'fromkeys', 'get', 'items', 'keys', 'pop',
                    'popitem', 'setdefault', 'update', 'values'] %}
{%- set own_methods = methods | reject('in', dict_api) | reject('eq', '__init__') | list %}
{%- set dunders = all_methods | select('eq', '__call__') | list %}
{%- set shown_methods = own_methods + dunders %}

.. autoclass:: {{ objname }}
{%- if attributes and not shown_methods %}
    :members:
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
    {% block attributes %}
    {#- When `:members:` is in play above, autodoc has already rendered each
        attribute inline *with its type*. A summary table would repeat all of
        them untyped, so emit it only for classes that keep the table form. #}
    {%- if attributes and shown_methods %}
    .. rubric:: {{ _('Attributes') }}

    .. autosummary::
        :toctree: .
    {% for item in attributes %}
        ~{{ name }}.{{ item }}
    {%- endfor %}
    {%- endif %}
    {% endblock %}
