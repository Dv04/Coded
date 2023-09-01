function [c, ceq] = conprobformin(x)
    % Nonlinear inequality constraints
    c = [2500 / (pi * x(1) * x(2)) - 500; 2500 / (pi * x(1) * x(2)) -
         (pi ^ 2 * (x(1) ^ 2 + x(2) ^ 2)) / 0.5882; -x(1) + 2; x(1) - 14; -x(2) + 0.2;
         x(2) - 0.8];
    % Nonlinear equality constraints
    ceq = [];

    clc
    clear all
    warning off
    x0 = [7 0.4]; % Starting guess\
    fprintf ('The values of function value and constraints
    at starting point \ n');
    f = probofminobj (x0)
    [c, ceq] = conprobformin (x0)
    options = optimset ('LargeScale', 'off');
    [x, fval] = fmincon (@probofminobj, x0, [], [], [], [], [],
    [], @conprobformin, options)
    fprintf('The values of constraints at optimum solution\n'); [c, ceq] = conprobformin(x) % Check the constraint values at x
