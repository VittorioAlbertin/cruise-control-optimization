%% Receding-horizon car-following simulation (MPC + GUROBI)
% Real-time updates with slack variable plotting and 80% regen efficiency.

clear; clc; close all;
yalmip('clear');

%% 1. Unified Parameters
dt = 0.1; N = 20; Tsim = 40; tvec = 0:dt:Tsim; nSteps = numel(tvec);

amin = -9.0; amax = 3.0; vmin = 0; vmax = 46.9; 
d0 = 4.0; h = 1.2; smax = 15;
w_traction = 10; w_regen = 2; w_v = 5; w_d = 20; w_s = 10000;

x_f = 0; v_f = 25; v_target = 36.1;

%% 2. True Leader Trajectory (Includes Hard Braking)
vL_true = 30 * ones(size(tvec));
vL_true(tvec >= 10 & tvec < 15) = 10; % Hard brake at t=10
vL_true(tvec >= 25) = 35;             

xL_true = zeros(size(tvec)); xL_true(1) = 40; 
for k = 2:nSteps, xL_true(k) = xL_true(k-1) + vL_true(k-1)*dt; end

%% 3. Build MPC Optimizer 
disp('Building MPC Controller...');
a_plus = sdpvar(N, 1); a_minus = sdpvar(N, 1);
v_opt = sdpvar(N+1, 1); p_opt = sdpvar(N+1, 1); s_opt = sdpvar(N, 1);
a_net = a_plus - a_minus;

p_init = sdpvar(1,1); v_init = sdpvar(1,1); p_lead_traj = sdpvar(N+1, 1);

constraints = [p_opt(1) == p_init, v_opt(1) == v_init, ...
               amin <= a_net <= amax, vmin <= v_opt <= vmax, ...
               0 <= s_opt <= smax, a_plus >= 0, a_minus >= 0];
obj = 0;
for k = 1:N
    constraints = [constraints, v_opt(k+1) == v_opt(k) + a_net(k)*dt];
    constraints = [constraints, p_opt(k+1) == p_opt(k) + v_opt(k)*dt + 0.5*a_net(k)*dt^2];
    constraints = [constraints, p_lead_traj(k+1) - p_opt(k+1) + s_opt(k) >= d0 + h*v_opt(k+1)];
    
    obj = obj + w_traction*a_plus(k)^2 + w_regen*a_minus(k)^2 + ...
          w_v*(v_opt(k+1) - v_target)^2 + w_s*s_opt(k)^2 + ...
          w_d*((p_lead_traj(k+1) - p_opt(k+1)) - (d0 + h*v_opt(k+1)))^2;
end

opts = sdpsettings('solver','gurobi','verbose',0);
MPC_Controller = optimizer(constraints, obj, opts, ...
                           {p_init, v_init, p_lead_traj}, ...
                           {a_net, a_plus, a_minus, s_opt(1)});

%% 4. Main Simulation Loop
x_f_log = zeros(nSteps,1); v_f_log = zeros(nSteps,1);
anet_log = zeros(nSteps,1); ap_log = zeros(nSteps,1); am_log = zeros(nSteps,1);
gap_log = zeros(nSteps,1); desired_gap_log = zeros(nSteps,1); slack_log = zeros(nSteps,1);

disp('Simulating...');
for idx = 1:nSteps-1
    x_f_log(idx) = x_f; v_f_log(idx) = v_f;
    xL_now = xL_true(idx); vL_now = vL_true(idx);
    
    gap_log(idx) = xL_now - x_f;
    desired_gap_log(idx) = d0 + h * v_f;
    
    assumed_lead_traj = xL_now + vL_now * (0:N)' * dt;
    
    [solutions, err] = MPC_Controller({x_f, v_f, assumed_lead_traj});
    
    if err == 0
        a_cmd = solutions{1}(1); ap_cmd = solutions{2}(1);
        am_cmd = solutions{3}(1); slack_log(idx) = solutions{4};
    else
        a_cmd = max(amin, -3.0); ap_cmd = 0; am_cmd = 3.0; slack_log(idx) = NaN;
    end
    
    anet_log(idx) = a_cmd; ap_log(idx) = ap_cmd; am_log(idx) = am_cmd;
    
    v_prev = v_f;
    v_f = max(0, v_f + a_cmd*dt);
    x_f = x_f + v_prev*dt + 0.5*a_cmd*dt^2;
end
gap_log(end) = xL_true(end) - x_f; desired_gap_log(end) = d0 + h * v_f;

%% 5. Presentation Plots
figure('Name','MPC Dashboard','Position',[100 100 1400 800]);

subplot(2,3,1); hold on; grid on;
plot(tvec, gap_log, 'b', 'LineWidth', 2, 'DisplayName', 'Actual Gap');
plot(tvec, desired_gap_log, 'r--', 'LineWidth', 2, 'DisplayName', 'Desired Gap');
ylabel('Distance (m)'); title('Gap Dynamics'); legend('Location','best');

subplot(2,3,2); hold on; grid on;
plot(tvec, vL_true, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Leader Speed');
plot(tvec, v_f_log, 'b-', 'LineWidth', 2, 'DisplayName', 'AV Speed');
ylabel('Speed (m/s)'); title('Velocity Tracking'); legend('Location','best');

subplot(2,3,3); hold on; grid on;
plot(tvec, ap_log, 'g', 'LineWidth', 1.5, 'DisplayName', 'Traction (a+)');
plot(tvec, -am_log, 'r', 'LineWidth', 1.5, 'DisplayName', 'Regen (a-)');
plot(tvec, anet_log, 'k--', 'LineWidth', 1, 'DisplayName', 'Net Accel');
ylabel('Acceleration (m/s^2)'); title('Actuation Command'); legend('Location','best');

subplot(2,3,4); hold on; grid on;
energy_spent = cumsum(ap_log.^2 * dt);
energy_recovered = cumsum(0.8 * am_log.^2 * dt); % 80% efficiency applied
plot(tvec, energy_spent, 'g', 'LineWidth', 2, 'DisplayName', 'Energy Spent');
plot(tvec, energy_recovered, 'r', 'LineWidth', 2, 'DisplayName', 'Energy Recovered (80%)');
ylabel('Cumulative Cost'); xlabel('Time (s)'); title('Energy Dynamics'); legend('Location','best');

% NEW SLACK PLOT
subplot(2,3,5); hold on; grid on;
plot(tvec, slack_log, 'm', 'LineWidth', 2, 'DisplayName', 'Slack Variable (s)');
ylabel('Slack (m)'); xlabel('Time (s)'); title('Safety Margin Violation'); legend('Location','best');