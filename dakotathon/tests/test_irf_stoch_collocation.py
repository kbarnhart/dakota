#!/usr/bin/env python
import os
import yaml
import glob
from nose.tools import assert_is, assert_true, assert_equal, with_setup
from dakotathon.bmi import StochasticCollocation
from dakotathon.utils import is_dakota_installed
from . import dakota_files


config_val = {'method': 'stoch_collocation', 'variables': 'uniform_uncertain'}


def setup():
    """Called at start of any test @with_setup()"""
    pass


def teardown():
    """Called at end of any test @with_setup()"""
    for f in dakota_files.values():
        if os.path.exists(f):
            os.remove(f)
    for f in glob.glob('LHS_*'):
        if os.path.exists(f):
            os.remove(f)
    if os.path.exists('S4'):
        os.remove('S4')


def test_component_name():
    model = StochasticCollocation()
    name = model.get_component_name()
    assert_equal(name, 'StochasticCollocation')
    assert_is(model.get_component_name(), name)


def test_start_time():
    model = StochasticCollocation()
    model.initialize()
    assert_equal(model.get_start_time(), 0.0)


def test_end_time():
    model = StochasticCollocation()
    model.initialize()
    assert_equal(model.get_end_time(), 1.0)


def test_current_time():
    model = StochasticCollocation()
    assert_equal(model.get_current_time(), 0.0)


def test_time_step():
    model = StochasticCollocation()
    assert_equal(model.get_time_step(), 1.0)


@with_setup(setup, teardown)
def test_initialize_defaults():
    model = StochasticCollocation()
    model.initialize()
    assert_true(os.path.exists(dakota_files['input']))


@with_setup(setup, teardown)
def test_initialize_from_file_like():
    from StringIO import StringIO

    config = StringIO(yaml.dump(config_val))
    model = StochasticCollocation()
    model.initialize(config)
    assert_true(os.path.exists(dakota_files['input']))


@with_setup(setup, teardown)
def test_initialize_from_file():
    import tempfile

    with tempfile.NamedTemporaryFile('w', delete=False) as fp:
        fp.write(yaml.dump(config_val))
        fname = fp.name

    model = StochasticCollocation()
    model.initialize(fname)
    os.remove(fname)
    assert_true(os.path.exists(dakota_files['input']))


def test_update():
    if is_dakota_installed():
        model = StochasticCollocation()
        model.initialize()
        model.update()
        assert_true(os.path.exists(dakota_files['input']))
        assert_true(os.path.exists(dakota_files['output']))
        assert_true(os.path.exists(dakota_files['data']))


def test_finalize():
    if is_dakota_installed():
        model = StochasticCollocation()
        model.initialize()
        model.update()
        model.finalize()
