%% Car-Following Optimization with YALMIP + GUROBI (Open-Loop)
% Constant leader speed, gap plotting, and 80% regen efficiency visualization.

clear; clc; close all;
yalmip('clear');

%% 1. Unified Parameters
dt   = 0.1;          
N    = 300;          

amin = -8.0; amax = 3.0; 
vmin = 0; vmax = 46.9;         
d0   = 4.0; h = 1.2; smax = 15;           

v0   = 10;           % Follower starts fast (30 m/s)
x0   = 0;            
v_target = 36.1;     

%% 2. Leader Trajectory (Constant Speed)
v_lead_const = 20;   % Leader is cruising slower (20 m/s)
v_ref_lead = v_lead_const * ones(N+1, 1);

xL = zeros(N+1, 1);
xL(1) = x0 + 80;     % Leader starts 80m ahead
for k = 1:N
    xL(k+1) = xL(k) + v_ref_lead(k)*dt;
end

%% 3. Weights & Variables
w_traction = 20; w_regen = 2; w_v = 5; w_d = 20; w_s = 10000;  

a_plus  = sdpvar(N, 1); 
a_minus = sdpvar(N, 1); 
v       = sdpvar(N+1, 1);
x       = sdpvar(N+1, 1);
s       = sdpvar(N, 1);
a_net   = a_plus - a_minus; 

%% 4. Constraints & Objective
constraints = [v(1) == v0, x(1) == x0, ...
               amin <= a_net <= amax, vmin <= v <= vmax, ...
               0 <= s <= smax, a_plus >= 0, a_minus >= 0];

objective = w_traction * sum(a_plus.^2) + w_regen * sum(a_minus.^2) + ...
            w_v * sum((v - v_target).^2) + w_s * sum(s.^2);

for k = 1:N
    constraints = [constraints, v(k+1) == v(k) + a_net(k)*dt];
    constraints = [constraints, x(k+1) == x(k) + v(k)*dt + 0.5*a_net(k)*dt^2];
    constraints = [constraints, -x(k+1) + h*v(k+1) + s(k) <= xL(k+1) - d0];
    
    dist_err = xL(k+1) - x(k+1) - d0 - h*v(k+1);
    objective = objective + w_d * dist_err^2;
end

%% 5. Solve
opts = sdpsettings('solver','gurobi','verbose',1);
sol = optimize(constraints, objective, opts);

%% 6. Extract & Plot
ap_opt = value(a_plus); am_opt = value(a_minus); anet_opt = value(a_net);
v_opt = value(v); x_opt = value(x);
time = (0:N)*dt;

% Calculate Gaps
actual_gap = xL - x_opt;
desired_gap = d0 + h * v_opt;

figure('Name', 'Open-Loop ACC Profile', 'Position', [100, 100, 1200, 800]);

subplot(2,2,1); hold on; grid on;
plot(time, actual_gap, 'b', 'LineWidth', 2, 'DisplayName', 'Actual Gap');
plot(time, desired_gap, 'r--', 'LineWidth', 2, 'DisplayName', 'Desired Gap (d0+hv)');
ylabel('Distance (m)'); title('Gap Dynamics'); legend('Location','best');

subplot(2,2,2); hold on; grid on;
plot(time, v_ref_lead, 'k--', 'LineWidth', 1.5, 'DisplayName', 'Leader Speed');
plot(time, v_opt, 'b-', 'LineWidth', 2, 'DisplayName', 'Follower Speed');
ylabel('Velocity (m/s)'); title('Velocity Tracking'); legend('Location','best');

subplot(2,2,3); hold on; grid on;
plot(time(1:end-1), ap_opt, 'g', 'LineWidth', 1.5, 'DisplayName', 'Traction (a+)');
plot(time(1:end-1), -am_opt, 'r', 'LineWidth', 1.5, 'DisplayName', 'Regen (a-)');
plot(time(1:end-1), anet_opt, 'k--', 'LineWidth', 1, 'DisplayName', 'Net Accel');
ylabel('Accel (m/s^2)'); title('Actuation Profile'); legend('Location','best');

subplot(2,2,4); hold on; grid on;
energy_spent = cumsum(ap_opt.^2 * dt);
energy_recovered = cumsum(0.8 * am_opt.^2 * dt); % 80% EFFICIENCY APPLIED HERE
plot(time(1:end-1), energy_spent, 'g', 'LineWidth', 2, 'DisplayName', 'Energy Spent');
plot(time(1:end-1), energy_recovered, 'r', 'LineWidth', 2, 'DisplayName', 'Energy Recovered (80%)');
xlabel('Time (s)'); ylabel('Cumulative Cost'); title('Energy Tracking (Proxy)'); legend('Location','best');