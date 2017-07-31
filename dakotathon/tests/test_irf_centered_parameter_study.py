#!/usr/bin/env python
import os
import yaml
from nose.tools import assert_is, assert_true, assert_equal, with_setup
from dakotathon.bmi import CenteredParameterStudy
from dakotathon.utils import is_dakota_installed
from . import dakota_files


config_val = {'method': 'centered_parameter_study'}


def setup():
    """Called at start of any test @with_setup()"""
    pass


def teardown():
    """Called at end of any test @with_setup()"""
    for f in list(dakota_files.values()):
        if os.path.exists(f):
            os.remove(f)


def test_component_name():
    model = CenteredParameterStudy()
    name = model.get_component_name()
    assert_equal(name, 'CenteredParameterStudy')
    assert_is(model.get_component_name(), name)


def test_start_time():
    model = CenteredParameterStudy()
    model.initialize()
    assert_equal(model.get_start_time(), 0.0)


def test_end_time():
    model = CenteredParameterStudy()
    model.initialize()
    assert_equal(model.get_end_time(), 1.0)


def test_current_time():
    model = CenteredParameterStudy()
    assert_equal(model.get_current_time(), 0.0)


def test_time_step():
    model = CenteredParameterStudy()
    assert_equal(model.get_time_step(), 1.0)


@with_setup(setup, teardown)
def test_initialize_defaults():
    model = CenteredParameterStudy()
    model.initialize()
    assert_true(os.path.exists(dakota_files['input']))


@with_setup(setup, teardown)
def test_initialize_from_file_like():
    from io import BytesIO

    config = BytesIO(yaml.dump(config_val, encoding=('utf-8')))
    model = CenteredParameterStudy()
    model.initialize(config)
    assert_true(os.path.exists(dakota_files['input']))


@with_setup(setup, teardown)
def test_initialize_from_file():
    import tempfile

    with tempfile.NamedTemporaryFile('w', delete=False) as fp:
        fp.write(yaml.dump(config_val))
        fname = fp.name

    model = CenteredParameterStudy()
    model.initialize(fname)
    os.remove(fname)
    assert_true(os.path.exists(dakota_files['input']))


def test_update():
    if is_dakota_installed():
        model = CenteredParameterStudy()
        model.initialize()
        model.update()
        assert_true(os.path.exists(dakota_files['input']))
        assert_true(os.path.exists(dakota_files['output']))
        assert_true(os.path.exists(dakota_files['data']))


def test_finalize():
    if is_dakota_installed():
        model = CenteredParameterStudy()
        model.initialize()
        model.update()
        model.finalize()
