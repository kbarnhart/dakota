#!/usr/bin/env python
import os
import yaml
from nose.tools import assert_is, assert_true, assert_equal, with_setup
from dakotathon.bmi import MultidimParameterStudy
from dakotathon.utils import is_dakota_installed
from . import dakota_files


config_val = {'method': 'multidim_parameter_study',
              'lower_bounds':[-2,-2], 'upper_bounds':[2,2]}


def setup():
    """Called at start of any test @with_setup()"""
    pass


def teardown():
    """Called at end of any test @with_setup()"""
    for f in dakota_files.values():
        if os.path.exists(f):
            os.remove(f)


def test_component_name():
    model = MultidimParameterStudy()
    name = model.get_component_name()
    assert_equal(name, 'MultidimParameterStudy')
    assert_is(model.get_component_name(), name)


def test_start_time():
    model = MultidimParameterStudy()
    model.initialize()
    assert_equal(model.get_start_time(), 0.0)


def test_end_time():
    model = MultidimParameterStudy()
    model.initialize()
    assert_equal(model.get_end_time(), 1.0)


def test_current_time():
    model = MultidimParameterStudy()
    assert_equal(model.get_current_time(), 0.0)


def test_time_step():
    model = MultidimParameterStudy()
    assert_equal(model.get_time_step(), 1.0)


@with_setup(setup, teardown)
def test_initialize_defaults():
    model = MultidimParameterStudy()
    model.initialize()
    assert_true(os.path.exists(dakota_files['input']))


@with_setup(setup, teardown)
def test_initialize_from_file_like():
    from StringIO import StringIO

    config = StringIO(yaml.dump(config_val))
    model = MultidimParameterStudy()
    model.initialize(config)
    assert_true(os.path.exists(dakota_files['input']))


@with_setup(setup, teardown)
def test_initialize_from_file():
    import tempfile

    with tempfile.NamedTemporaryFile('w', delete=False) as fp:
        fp.write(yaml.dump(config_val))
        fname = fp.name

    model = MultidimParameterStudy()
    model.initialize(fname)
    os.remove(fname)
    assert_true(os.path.exists(dakota_files['input']))


def test_update():
    if is_dakota_installed():
        model = MultidimParameterStudy()
        model.initialize()
        model.update()
        assert_true(os.path.exists(dakota_files['input']))
        assert_true(os.path.exists(dakota_files['output']))
        assert_true(os.path.exists(dakota_files['data']))


def test_finalize():
    if is_dakota_installed():
        model = MultidimParameterStudy()
        model.initialize()
        model.update()
        model.finalize()
