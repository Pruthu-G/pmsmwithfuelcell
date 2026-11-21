function in = myResetFunction(in)
    %% Randomise speed reference using random walk (30s episode at Tsc = 0.05s)
    t_end = 30;
    dt    = 0.05;
    t     = 0:dt:t_end;
    N     = length(t);

    RPM_nominal = 180;
    RPM_min     = 50;
    RPM_max     = 200;
    step_std    = 5;

    RPM_walk    = zeros(1, N);
    RPM_walk(1) = RPM_nominal;
    for k = 2:N
        RPM_walk(k) = RPM_walk(k-1) + step_std * randn;
        RPM_walk(k) = max(RPM_min, min(RPM_max, RPM_walk(k)));
    end

    % Pass RPM directly because PMSM controller converts rpm -> rad/s internally
    ref_signal = timeseries(RPM_walk', t');
    in = setVariable(in, 'speed_reference', ref_signal);
end