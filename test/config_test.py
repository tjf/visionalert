import os

import pytest

from visionalert import load_config, config


@pytest.fixture
def config_file(tmpdir, monkeypatch):
    monkeypatch.setenv('FOO', 'BAR')

    fh = tmpdir.join('config.yml')
    fh.write("""line1: line1value
line2: line2${FOO}value
line3: line3${FOO}valueline3${FOO}value
    """)
    return os.path.join(fh.dirname, fh.basename)


def test_load_config_with_variables(config_file):
    load_config(config_file)
    assert config['line1'] == 'line1value'
    assert config['line2'] == 'line2BARvalue'
    assert config['line3'] == 'line3BARvalueline3BARvalue'
