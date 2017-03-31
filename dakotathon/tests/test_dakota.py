#!/usr/bin/env python
#
# Tests for the dakotathon.dakota module.
#
# Call with:
#   $ nosetests -sv
#
# Mark Piper (mark.piper@colorado.edu)

import os
import filecmp
from subprocess import CalledProcessError
from nose.tools import (raises, assert_is_instance, assert_true,
                        assert_is_none, assert_equal)
from dakotathon.dakota import Dakota
from dakotathon.utils import is_dakota_installed
from . import start_dir, data_dir


# Global variables -----------------------------------------------------

input_file, \
    output_file, \
    data_file, \
    restart_file = ['dakota.' + ext for ext in ('in', 'out', 'dat', 'rst')]
alt_input_file = 'alt.in'
known_file = os.path.join(data_dir, 'default_vps_dakota.in')
known_config_file = os.path.join(data_dir, 'default_vps_dakota.yaml')
default_config_file = os.path.join(os.getcwd(), 'dakota.yaml')
tmp_files = [input_file, alt_input_file, output_file, data_file,
             restart_file, 'dakota.yaml']

# Fixtures -------------------------------------------------------------


def setup_module():
    """Fixture called before any tests are performed."""
    print('\n*** ' + __name__)
    global d
    d = Dakota()


def teardown_module():
    """Fixture called after all tests have completed."""
    for f in tmp_files:
        if os.path.exists(f):
            os.remove(f)

# Tests ----------------------------------------------------------------


def test_init_no_parameters():
    """Test constructor with no parameters."""
    k = Dakota()
    assert_is_instance(k, Dakota)


def test_init_method_parameter():
    """Test constructor with method parameter."""
    k = Dakota(method='vector_parameter_study')
    assert_is_instance(k, Dakota)


@raises(ImportError)
def test_init_method_parameter_unknown_module():
    """Test constructor with method parameter fails with unknown module."""
    k = Dakota(method='__foo$')


def test_init_from_file_like1():
    """Test creating an instance from a config file."""
    k = Dakota.from_file_like(known_config_file)
    assert_is_instance(k, Dakota)


def test_init_from_file_like2():
    """Test creating an instance from an open config file object."""
    with open(known_config_file, 'r') as fp:
        k = Dakota.from_file_like(fp)
    assert_is_instance(k, Dakota)


def test_get_run_directory():
    """Test getting the run_directory property."""
    assert_equal(d.run_directory, os.getcwd())


def test_set_run_directory():
    """Test setting the run_directory property."""
    k = Dakota()
    run_dir = '/foo/bar'
    k.run_directory = run_dir
    assert_equal(k.run_directory, run_dir)


def test_get_configuration_file():
    """Test getting the configuration_file property."""
    assert_equal(d.configuration_file, default_config_file)


def test_set_configuration_file():
    """Test setting the configuration_file property."""
    k = Dakota()
    k.configuration_file = known_config_file
    assert_equal(k.configuration_file, known_config_file)


def test_get_template_file():
    """Test getting the template_file property."""
    assert_is_none(d.template_file)


def test_set_template_file():
    """Test setting the template_file property."""
    k = Dakota()
    template_file = 'foo.tmpl'
    k.template_file = template_file
    assert_equal(os.path.basename(k.template_file), template_file)


def test_get_auxiliary_files():
    """Test getting the auxiliary_files property."""
    assert_equal(d.auxiliary_files, tuple())


def test_set_auxiliary_files():
    """Test setting the auxiliary_files property."""
    k = Dakota()
    for auxiliary_file in ['foo.in', ['foo.in'], ('foo.in',)]:
        k.auxiliary_files = auxiliary_file
        if type(auxiliary_file) is not str:
            auxiliary_file = auxiliary_file[0]
        pathified_auxiliary_file = os.path.abspath(auxiliary_file)
        assert_equal(k.auxiliary_files, (pathified_auxiliary_file,))


@raises(TypeError)
def test_set_auxiliary_files_fails_if_scalar():
    """Test that auxiliary_files fails with a non-string scalar."""
    k = Dakota()
    auxiliary_file = 42
    k.auxiliary_files = auxiliary_file


def test_write_configuration_file():
    """Test serialize method produces config file."""
    k = Dakota(method='vector_parameter_study')
    k.serialize()
    assert_true(os.path.exists(k.configuration_file))


def test_write_input_file_with_method_default_name():
    """Test write_input_file works when instanced with method."""
    k = Dakota(method='vector_parameter_study')
    k.write_input_file()
    assert_true(os.path.exists(k.input_file))


def test_write_input_file_with_method_new_name():
    """Test write_input_file works when instanced with method and new name."""
    k = Dakota(method='vector_parameter_study')
    k.write_input_file(input_file=alt_input_file)
    assert_true(os.path.exists(k.input_file))


def test_input_file_contents():
    """Test write_input_file results versus a known input file."""
    k = Dakota(method='vector_parameter_study')
    k.write_input_file()
    assert_true(filecmp.cmp(known_file, input_file))


def test_setup():
    """Test the setup method."""
    k = Dakota(method='vector_parameter_study')
    k.setup()
    assert_true(os.path.exists(k.configuration_file))
    assert_true(filecmp.cmp(known_file, input_file))


def test_default_run_with_input_file():
    """Test default object run method with input file."""
    if is_dakota_installed():
        k = Dakota()
        k.write_input_file()
        k.run()
        assert_true(os.path.exists(k.output_file))
        assert_true(os.path.exists(k.environment.data_file))


def test_default_run_without_input_file():
    """Test default object run method fails with no input file."""
    if is_dakota_installed():
        if os.path.exists(input_file):
            os.remove(input_file)
        try:
            k = Dakota()
            k.run()
        except CalledProcessError:
            pass

def test_changing_parameter_names():
    """Test ability to provide parameter names."""
    run_directory = 'looped_runs'
    configuration_file = 'an_excellent_yaml.yaml'
    run_log = 'run_output_here.log'
    error_log = 'no_errors_here.log'
    k = Dakota(method='vector_parameter_study',
               run_directory=run_directory,
               configuration_file=configuration_file,
               run_log=run_log,
               error_log=error_log)
    k.write_input_file()
    k.serialize()
    k.run()

    os.remove(configuration_file)
    os.remove(run_log)
    os.remove(error_log)
    teardown_module()
    os.chdir('..')
    os.rmdir(run_directory)

def test_running_in_different_directory():
    """Test ability to provide parameter names."""
    work_folder = 'dakota_runs'
    run_directory = os.path.abspath(os.path.join(os.getcwd(), 'running_here'))
    work_directory = os.path.abspath(os.path.join(run_directory, '..','working_here'))
    configuration_file = 'another_excellent_yaml.yaml'
    run_log = 'run_output_should_be_here.log'
    error_log = 'no_errors_inside.log'
    parameters_file = 'parameters_here.in'
    results_file = 'neato_results_here.out'
    input_file = 'dakota_LHC.in'
    output_file = 'dakota_LHC.out'
    
    k = Dakota(method='sampling',
               variables='uniform_uncertain',
               input_file=input_file,
               output_file=output_file,
               interface='fork',
               run_directory=run_directory,
               work_directory=work_directory,
               work_folder=work_folder,
               configuration_file=configuration_file,
               run_log=run_log,
               error_log=error_log,
               parameters_file=parameters_file,
               results_file=results_file)
    k.write_input_file()
    k.serialize()
    k.run()
    
    # teardown. First the run directory.
    lhc_filelist = [ f for f in os.listdir(".") if f.startswith("LHS") ]
    for f in lhc_filelist:
        os.remove(f)
    os.remove(configuration_file)
    os.remove(run_log)
    os.remove(error_log)
    os.remove(input_file)
    os.remove(output_file)
    teardown_module()
    os.chdir('..')
    os.rmdir(run_directory)

    # Second the working directory.

    num_runs = k.method.samples
    os.chdir(work_directory)
    for i in range(num_runs):
        folder_name = '.'.join([work_folder, str(i+1)])
        os.chdir(folder_name)
        os.remove(parameters_file)
        os.remove(results_file)
        os.chdir('..')
        os.rmdir(folder_name)
    os.chdir('..')
    os.rmdir(work_directory)


