%% train_rl_2D.m
% 1. System Sample Times
Ts  = 2e-6;   % Physical Simscape solver step (2 microseconds)
Tsc = 0.05;   % RL Agent sample time (50 ms = 20 Hz control loop)

% PMSM Parameters 
PM = 0.03; Ld = 2e-4; Lq = 2e-4; L0 = 2e-4; Rs = 0.013; N = 6; J = 0.2;
Imax = 100; Tmax = 1.5*N*PM*Imax; fnom = 100;
rpm_nom = 60*fnom/N; omegam_nom = 2*pi*fnom/N; Pmax = omegam_nom*Tmax;

%% 2. Open Simulink Model
modelName = 'pmsm_with_fuel_cell';
open_system(modelName);

%% 3. Observation Spec — 11 Signals (Added 2nd Action Feedback)
numObs = 11;
obsInfo = rlNumericSpec([numObs 1], ...
    'LowerLimit', -inf*ones(numObs,1), ...
    'UpperLimit',  inf*ones(numObs,1));
obsInfo.Name = 'observations';

%% 4. Action Spec — 2D Action Space
% Action 1: Duty cycle shift [-0.05, 0.05]
% Action 2: Bus voltage setpoint delta [-5V, +5V]
numAct = 2;
actInfo = rlNumericSpec([numAct 1], ...
    'LowerLimit', [-0.05; 300], ...
    'UpperLimit',  [ 0.05;  500]);
actInfo.Name = 'actions';

%% 5. Create Simulink Environment
env = rlSimulinkEnv(modelName, ...
    [modelName '/RL Agent'], ...
    obsInfo, actInfo);
env.ResetFcn = @myResetFunction;

%% 6. Critic Networks (Double Q-Functions for SAC)
createCritic = @() ...
    layerGraph([
        featureInputLayer(numObs, 'Name', 'obs')
        fullyConnectedLayer(64, 'Name', 'obs_fc1')
        reluLayer('Name', 'obs_relu1')
    ]) ...
    .addLayers([
        featureInputLayer(numAct, 'Name', 'act')
        fullyConnectedLayer(64, 'Name', 'act_fc1')
        reluLayer('Name', 'act_relu1')
    ]) ...
    .addLayers([
        concatenationLayer(1, 2, 'Name', 'concat')
        fullyConnectedLayer(64, 'Name', 'common_fc1')
        reluLayer('Name', 'common_relu1')
        fullyConnectedLayer(1, 'Name', 'output')
    ]) ...
    .connectLayers('obs_relu1', 'concat/in1') ...
    .connectLayers('act_relu1', 'concat/in2');

% Initialize two Q-Value representations
criticNet1 = createCritic();
criticNet2 = createCritic();

critic1 = rlQValueFunction(criticNet1, obsInfo, actInfo, ...
    'ObservationInputNames', 'obs', 'ActionInputNames', 'act');
critic2 = rlQValueFunction(criticNet2, obsInfo, actInfo, ...
    'ObservationInputNames', 'obs', 'ActionInputNames', 'act');

critics = [critic1, critic2];

%% 7. Actor Network — 2D Gaussian Outputs (Mean and Std)
commonActorPath = [
    featureInputLayer(numObs, 'Name', 'obs')
    fullyConnectedLayer(64, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')];

meanPath = fullyConnectedLayer(numAct, 'Name', 'mean');
stdPath  = [
    fullyConnectedLayer(numAct, 'Name', 'std_fc')
    softplusLayer('Name', 'std')];

actorNetwork = layerGraph(commonActorPath) ...
    .addLayers(meanPath) ...
    .addLayers(stdPath) ...
    .connectLayers('relu2', 'mean') ...
    .connectLayers('relu2', 'std_fc');

actor = rlContinuousGaussianActor(actorNetwork, obsInfo, actInfo, ...
    'ActionMeanOutputNames',              'mean', ...
    'ActionStandardDeviationOutputNames', 'std');

%% 8. SAC Agent Options
agentOpts = rlSACAgentOptions(...
    'SampleTime',                Tsc, ...
    'TargetSmoothFactor',        1e-3, ...
    'ExperienceBufferLength',    1e6, ...
    'MiniBatchSize',             256, ...
    'NumWarmStartSteps',         1000);

agent = rlSACAgent(actor, critics, agentOpts);

%% 9. Training Options
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',                1000, ...
    'MaxStepsPerEpisode',         600, ...  % 600 steps * 0.05s = 30 seconds per episode
    'ScoreAveragingWindowLength', 20, ...
    'StopTrainingCriteria',       'AverageReward', ...
    'StopTrainingValue',          -100, ...
    'Verbose',                    true, ...
    'Plots',                      'training-progress', ...
    'SaveAgentCriteria',          'EpisodeReward', ...
    'SaveAgentValue',             -500, ...
    'SaveAgentDirectory',         'saved_agents');

%% 10. Train
trainingStats = train(agent, env, trainOpts);

%% 11. Save Final Agent
save('trained_fc_2D_agent.mat', 'agent', 'trainingStats');