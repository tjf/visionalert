import os

import pytest

from visionalert.config import load_config, Config


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
    assert Config['line1'] == 'line1value'
    assert Config['line2'] == 'line2BARvalue'
    assert Config['line3'] == 'line3BARvalueline3BARvalue'
