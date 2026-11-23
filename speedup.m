mdl = 'pmsm_with_fuel_cell';
if ~bdIsLoaded(mdl), load_system(mdl); end

% Turn off Fast Restart temporarily so block parameters can be edited
set_param(mdl, 'FastRestart', 'off');

% 1. Lower nonlinear iterations from 10 to 4 (safe now that t=0 is stable)
set_param([mdl '/Solver Configuration'], 'MaxNonlinIter', '4');

% 2. Disable thermal networks on all 6 inverter IGBTs and Diode
allBlks = find_system(mdl, 'LookUnderMasks', 'all', 'FollowLinks', 'on');
for k = 1:length(allBlks)
    try
        if any(strcmp(fieldnames(get_param(allBlks{k}, 'DialogParameters')), 'no_thermal_network'))
            set_param(allBlks{k}, 'no_thermal_network', '1');
        end
    catch
    end
end

% 3. Switch Diode fidelity to simplest piecewise-linear mode
try
    fidEnums = enumeration('ee.enum.diode.fidelity');
    set_param([mdl '/Diode'], 'fidelity_level', ['ee.enum.diode.fidelity.' char(fidEnums(1))]);
catch
end

% 4. Disable Simscape data logging & compile to C code (Accelerator + Fast Restart)
set_param(mdl, ...
    'SimscapeLogType', 'none', ...
    'SimulationMode',  'accelerator', ...
    'FastRestart',     'on');

disp('Speedup settings applied and model compiled in Accelerator mode.');