import os
import re

import yaml

config = {}


def _env_var_constructor(_, node):
    return re.sub(r'\${([^}]+)}', lambda m: os.getenv(m.group(1), m.group(0)), node.value)


# add_implicit_resolver appears to anchor the regex to the beginning of the line, hence the .*?
yaml.add_implicit_resolver('!path', re.compile(r'.*?\${[^}]+}'), None, yaml.SafeLoader)
yaml.add_constructor('!path', _env_var_constructor, yaml.SafeLoader)


def load_config(filename):
    with open(filename) as file:
        config.update(yaml.safe_load(file))
