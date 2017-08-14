#! /usr/bin/env python
# Katy Barnhart; begun July 2017
"""Implementation of a Dakota Gauss Newton Optimization study.

Information about paramter values from the Dakota Documentation:
https://dakota.sandia.gov/content/latest-reference-manual

**Search Method**
The search_method control is defined for all Newton-based optimizers and is used
to select between trust_region, gradient_based_line_search, and
value_based_line_search methods. The gradient_based_line_search option uses the
line search method proposed by[61]. This option satisfies sufficient decrease
and curvature conditions; whereas, value_base_line_search only satisfies the
sufficient decrease condition. At each line search iteration, the
gradient_based_line_search method computes the function and gradient at the
trial point. Consequently, given expensive function evaluations, the
value_based_line_search method is preferred to the gradient_based_line_search
method. Each of these Newton methods additionally supports the tr_pds selection
for unconstrained problems. This option performs a robust trust region search
using pattern search techniques. Use of a line search is the default for
bound-constrained and generally-constrained problems, and use of a trust_region
search method is the default for unconstrained problems.

**Step-length to boundary**
The steplength_to_boundary specification is a parameter (between 0 and 1) that
controls how close to the boundary of the feasible region the algorithm is
allowed to move. A value of 1 means that the algorithm is allowed to take steps
that may reach the boundary of the feasible region. If the user wishes to
maintain strict feasibility of the design parameters this value should be less
than 1.

Default values are .8, .99995, and .95 for the el_bakry, argaez_tapia, and
van_shanno merit functions, respectively.

**Centering Parameter**
The centering_parameter specification is a parameter (between 0 and 1) that
controls how closely the algorithm should follow the "central path". See[89] for
the definition of central path. The larger the value, the more closely the
algorithm follows the central path, which results in small steps. A value of 0
indicates that the algorithm will take a pure Newton step.

Default values are .2, .2, and .1 for the el_bakry, argaez_tapia, and van_shanno
merit functions, respectively.

**Max Step**
The max_step control specifies the maximum step that can be taken when computing
a change in the current design point (e.g., limiting the Newton step computed
from current gradient and Hessian information). It is equivalent to a move limit
or a maximum trust region size. The gradient_tolerance control defines the
threshold value on the L2 norm of the objective function gradient that indicates
convergence to an unconstrained minimum (no active constraints). The
gradient_tolerance control is defined for all gradient-based optimizers.

**Gradient Tolerance**
The gradient_tolerance control defines the threshold value on the L2 norm of the
objective function gradient that indicates convergence to an unconstrained
minimum (no active constraints). The gradient_tolerance control is defined for
all gradient-based optimizers.

**Max Itterations**
The maximum number of iterations is used as a stopping criterion for optimizers
and some adaptive UQ methods. If it has not reached any other stopping criteria
first, the method will stop after it has performed max_iterations iterations.
See also max_function_evaluations.

**Convergence Tolerance**
The convergence_tolerance specification provides a real value for controlling
the termination of iteration.

For optimization, it is most commonly a relative convergence tolerance for the
objective function; i.e., if the change in the objective function between
successive iterations divided by the previous objective function is less than
the amount specified by convergence_tolerance, then this convergence criterion
is satisfied on the current iteration.

Therefore, permissible values are between 0 and 1, non-inclusive.

**Speculative**
When performing gradient-based optimization in parallel, speculative gradients
can be selected to address the load imbalance that can occur between gradient
evaluation and line search phases. In a typical gradient-based optimization, the
line search phase consists primarily of evaluating the objective function and
any constraints at a trial point, and then testing the trial point for a
sufficient decrease in the objective function value and/or constraint violation.
If a sufficient decrease is not observed, then one or more additional trial
points may be attempted sequentially. However, if the trial point is accepted
then the line search phase is complete and the gradient evaluation phase begins.
By speculating that the gradient information associated with a given line search
trial point will be used later, additional coarse grained parallelism can be
introduced by computing the gradient information (either by finite difference or
analytically) in parallel, at the same time as the line search phase
trial-point function values. This balances the total amount of computation to
be performed at each design point and allows for efficient utilization of
multiple processors. While the total amount of work performed will generally
increase (since some speculative gradients will not be used when a trial point
is rejected in the line search phase), the run time will usually decrease
(since gradient evaluations needed at the start of each new optimization cycle
were already performed in parallel during the line search phase). Refer to [13]
for additional details. The speculative specification is implemented for the
gradient-based optimizers in the DOT, CONMIN, and OPT++ libraries, and it can
be used with dakota numerical or analytic gradient selections in the responses
specification (refer to responses gradient section for information on these
specifications). It should not be selected with vendor numerical gradients
since vendor internal finite difference algorithms have not been modified for
this purpose. In full-Newton approaches, the Hessian is also computed
speculatively. NPSOL and NLSSOL do not support speculative gradients, as their
gradient-based line search in user-supplied gradient mode (dakota numerical or
analytic gradients) is a superior approach for load-balanced parallel
execution.

The speculative specification enables speculative computation of gradient and/or
Hessian information, where applicable, for parallel optimization studies. By
speculating that the derivative information at the current point will be used
later, the complete data set (all available gradient/Hessian information) can be
computed on every function evaluation. While some of these computations will be
wasted, the positive effects are a consistent parallel load balance and usually
shorter wall clock time. The speculative specification is applicable only when
parallelism in the gradient calculations can be exploited by Dakota (it will be
ignored for vendor numerical gradients).

**Maximum Function Evalutions**
The maximum number of function evaluations is used as a stopping criterion for
optimizers. If it has not reached any other stopping criteria first, the
optimizer will stop after it has performed max_function_evalutions evaluations.
See also max_iterations.

**Scaling**
Some of the optimization and calibration methods support scaling of continuous
design variables, objective functions, calibration terms, and constraints. This
is activated by by providing the scaling keyword. Discrete variable scaling is
not supported.

When scaling is enabled, variables, functions, gradients, Hessians, etc., are
transformed such that the method iterates in scaled variable space, whereas
evaluations of the computational model as specified in the interface are
performed on the original problem scale. Therefore using scaling does not
require rewriting the interface to the simulation code.

Scaling also requires the specification of additional keywords which are found
in the method, variables, and responses blocks. When the scaling keyword is
omitted, all _scale_types and *_scales specifications are ignored in the method,
variables, and responses sections.

This page describes the usage of all scaling related keywords. The additional
keywords come in pairs, one pair for each set of quantities to be scaled. These
quantities can be constraint equations, variables, or responses.

    a *scales keyword, which gives characteristic values
    a *scale_type keyword, which determines how to use the characteristic values

The pair of keywords both take argument(s), and the length of the arguments can
either be zero, one, or equal to the number of quantities to be scaled. If one
argument is given, it will apply to all quantities in the set. See the examples
in the Dakota online documentation for more information.

**Model Pointer**
The model_pointer is used to specify which model block will be used to perform
the function evaluations needed by the Dakota method.


"""
from .base import MethodBase
import numpy as np

classname = 'OptppGNewton'

class OptppGNewton(MethodBase):

    """Define parameters for a Dakota Gauss Newton Optimization study."""

    def __init__(self,
                 search_method,
                 merit_function=None,
                 step_length_to_boundary=None,
                 centering_parameter=None,
                 max_step=None,
                 gradient_tolerance=None,
                 max_iterations=None,
                 speculative=None,
                 max_function_evaluations=None,
                 scaling=None,
                 model_pointer=None,
                 **kwargs):
        """Create a new Dakota Gauss Newton Optimization study.

        From the Dakota Documentation:

        The Gauss-Newton algorithm is available as optpp_g_newton and supports
        unconstrained, bound-constrained, and generally-constrained problems.
        When interfaced with the unconstrained, bound-constrained, and nonlinear
        interior point full-Newton optimizers from the OPT++ library, it
        provides a Gauss-Newton least squares capability which – on
        zero residual test problems can exhibit quadratic convergence rates
        near the solution. (Real problems almost never have zero residuals,
        i.e., perfect fits.)

        See package_optpp for info related to all optpp methods.

        When an optional parameter is set to None, Dakota's default value or
        method is used.

        Parameters
        ----------
        search_method : string, required.
            Select a search method for Newton-based optimizers. Options are
            'gradient_based_line_search', 'trust_region',
            'value_based_line_search', and 'tr_pds'. For unconstrained
            optimization, Dakota documentation recomends 'trust_region' and for
            constrained optimization, the documenation recommends
            'gradient_based_line_search'.
        merit_function : string or None, optional. Default is 'argaez_tapia'
            Balance goals of reducing objective function and satisfying
            constraints. Options are, 'el_bakry', 'argaez_tapia', and
            'van_shanno'.
        step_length_to_boundary : float or None, optional.
            Controls how close to the boundary of the feasible region the
            algorithm is allowed to move.
            Default is Merit function dependent:
            0.8 (el_bakry), 0.99995 (argaez_tapia), 0.95 (van_shanno)
        centering_parameter : float or None, optional
            Controls how closely the algorithm should follow the "central path".
            Default values are .2, .2, and .1 for the el_bakry, argaez_tapia,
            and van_shanno merit functions, respectively.
        max_step : float or None, optional. Default is 1000.
            Max change in design point
        gradient_tolerance : float or None, optional. Default is 1.e-4
          Stopping critiera based on L2 norm of gradient
        max_iterations : int or None, optional. Default is 100
            Number of iterations allowed for optimizers and adaptive UQ methods
        convergence_tolerance : float or None, optional. Default is 1.e-4
            Stopping criterion based on convergence of the objective function or
            statistics
        speculative : boolean or None, optional. Default is False.
            Compute speculative gradients. Default is no speculation.
        max_function_evaluations : int or None, optional. Default is 1000.
            Number of function evaluations allowed for optimizers
        scaling : boolean or None, optional. Default is False.
            Turn on scaling for variables, responses, and constraints
        model_pointer : string or None, optional. Default is last model block
                        parsed.
            Identifier for model block to be used by a method.

        Examples
        --------
        Create a default vector parameter study experiment:

        >>> v = OptppGNewton(search_method='gradient_based_line_search')

        """
        MethodBase.__init__(self, **kwargs)
        self.method = 'optpp_g_newton'

        # check and set search method.
        if search_method in ['gradient_based_line_search',
                             'trust_region',
                             'value_based_line_search methods',
                             'tr_pds']:
            self._search_method = search_method
        else:
            raise ValueError("Search Method must be one of the following:"
                             "'gradient_based_line_search' \n"
                             "trust_region' \n"
                             "'value_based_line_search methods' \n"
                             "'tr_pds' \n")

        # check and set merit function
        if merit_function is None:
            self._merit_function = merit_function
        else:
            if merit_function in ['el_bakry', 'argaez_tapia', 'van_shanno']:
                self._merit_function = merit_function
            else:
                raise ValueError("Merit function must be 'el_bakry', 'argaez_tapia',"
                                 " or 'van_shanno'")

        # check and set step length.
        if step_length_to_boundary is None:
            step_length_dict = {'el_bakry': 0.8,
                                'argaez_tapia' : 0.99995,
                                'van_shanno': 0.95,
                                None:None}

            self._step_length_to_boundary = step_length_dict[merit_function]
        else:
            try:
                self._step_length_to_boundary = float(step_length_to_boundary)
            except ValueError:
                raise ValueError('step_length_to_boundary must be convertable '
                                 'to float')

        # check and set centering parameter.
        if centering_parameter is None:
            centering_parameter_dict = {'el_bakry': 0.2,
                                        'argaez_tapia': 0.2,
                                        'van_shanno': 0.1,
                                        None:None}

            self._centering_parameter = centering_parameter_dict[merit_function]
        else:
            try:
                self._centering_parameter = float(centering_parameter)
            except ValueError:
                raise ValueError('centering_parameter must be convertable to '
                                 'float')

        # check and set max_step.
        if max_step is None:
            self._max_step = max_step
        else:
            try:
                self._max_step = float(max_step)
            except ValueError:
                raise ValueError('max_step must be convertable to float')

        # check and set gradient_tolerance.
        if gradient_tolerance is None:
            self._gradient_tolerance = gradient_tolerance
        else:
            try:
                self._gradient_tolerance = float(gradient_tolerance)
            except ValueError:
                raise ValueError('gradient_tolerance must be convertable to float')

        # check and set max_iterations.
        if max_iterations is None:
            self._max_iterations = max_iterations
        else:
            try:
                self._max_iterations = int(max_iterations)
            except ValueError:
                raise ValueError('max_iterations must be convertable to int')
            if np.isclose(self._max_iterations, max_iterations) == False:
                raise ValueError('You provided a value for max_iterations that is '
                                 'not an integer.')

        # check and set convergence_tolerance.
        #already done by base

        # check and set speculative.
        # this is not checked b/c KRB not certain what the options are and
        # Dakota documentation is thick on what this does but thin on how to
        # set it.
        self._speculative = speculative
        if speculative is not None:
            raise NotImplementedError('speculative not yet implemented')

        if max_function_evaluations is None:
            self._max_function_evaluations = max_function_evaluations
        else:
            try:
                self._max_function_evaluations = int(max_function_evaluations)
            except ValueError:
                raise ValueError('max_iterations must be convertable to int')
            if np.isclose(self._max_function_evaluations, max_function_evaluations) == False:
                raise ValueError('You provided a value for max_function_evaluations'
                                 ' that is not an integer.')

        # check and set scaling.
        # this is not checked b/c its somewhat complicated... Not done yet.
        self._scaling = scaling
        if scaling is not None:
            raise NotImplementedError('scaling not yet implemented')

        # check and set model_pointer.
        if model_pointer is None:
            self._model_pointer = model_pointer
        else:
            try:
                self._model_pointer = str(model_pointer)
            except:
                raise ValueError('You provided a value for model_pointer that is '
                                 'not convertable to string.')

    @property
    def search_method(self):
        """Search method used by study."""
        return self._search_method

    @property
    def merit_function(self):
        """Merit function used by study."""
        return self._merit_function

    @property
    def step_length_to_boundary(self):
        """Step length to boundary used by study."""
        return self._step_length_to_boundary

    @property
    def centering_parameter(self):
        """Centering parameter used by study."""
        return self._centering_parameter

    @property
    def max_step(self):
        """Maximum step size used by study."""
        return self._max_step

    @property
    def gradient_tolerance(self):
        """Gradient tolerance used by study."""
        return self._gradient_tolerance

    @property
    def max_iterations(self):
        """Maximum number of iterations used by study."""
        return self._max_iterations

    @property
    def convergence_tolerance(self):
        """Convergence tolerance used by study."""
        return self._convergence_tolerance

    @property
    def speculative(self):
        """Speculative parameter used by study."""
        return self._speculative

    @property
    def max_function_evaluations(self):
        """Maximum function evaluations used by study."""
        return self._max_function_evaluations

    @property
    def scaling(self):
        """Scaling used by study."""
        return self._scaling

    @property
    def model_pointer(self):
        """Model pointer used by study."""
        return self._model_pointer


    def __str__(self):
        """Define a Dakota Gauss Newton Optimization study method block for a
        Dakota input file.

        See Also
        --------
        dakotathon.method.base.MethodBase.__str__

        """
        s = MethodBase.__str__(self)
        if self.search_method is not None:
            s += '    search_method = '
            s += '{}\n'.format(self.search_method)
        if self.merit_function is not None:
            s += '    merit_function = '
            s += '{}\n'.format(self.merit_function)
        if self.step_length_to_boundary is not None:
            s += '    step_length_to_boundary = '
            s += '{}\n'.format(self.step_length_to_boundary)
        if self.centering_parameter is not None:
            s += '    centering_parameter = '
            s += '{}\n'.format(self.centering_parameter)
        if self.max_step is not None:
            s += '    max_step = '
            s += '{}\n'.format(self.max_step)
        if self.gradient_tolerance is not None:
            s += '    gradient_tolerance = '
            s += '{}\n'.format(self.gradient_tolerance)
        if self.max_iterations is not None:
            s += '    max_iterations = '
            s += '{}\n'.format(self.max_iterations)
        # if self.speculative is not None:
        #     s += '    speculative = '
        #     s += '{}\n'.format(self.speculative)
        if self.max_function_evaluations is not None:
            s += '    max_function_evaluations = '
            s += '{}\n'.format(self.max_function_evaluations)
        # if self.scaling is not None:
        #     s += '    scaling = '
        #     s += '{}\n'.format(self.scaling)
        if self.model_pointer is not None:
            s += '    model_pointer = '
            s += '{}\n'.format(self.model_pointer)
        s += '\n'
        return(s)
