% dem_linear_system.m
%
% Validation script for DEM (Dynamic Expectation Maximization) using MATLAB + SPM12
% 3-phase design matching the Python version:
%
%   Phase 1 — D-step (a=0 fixed):    spm_DEM with pC=0
%   Phase 2 — E-step (param est):    spm_nlsi_GN (Gauss-Newton)
%   Phase 3 — D-step (estimated a):  spm_DEM with pC=0
%
% Target system: damped linear system
%   dx/dt = a * x + v    (true a = -1.0, external input v = 0)
%   y     = x + epsilon  (observation noise)
%
% Output: results/comparison_matlab.csv
%   Columns: t, x_true, x_estimated, a_estimated, vfe
%
% Correspondence with the Python version:
%   experiments/comparison/python/dem_linear_system.py

clear; clc;
addpath('/home/jkoba/SynologyDrive_private/SynologyDrive/spm12');
fprintf('=== DEM Linear System: MATLAB + SPM12 ===\n\n');

try
    v = spm('version');
    fprintf('SPM version: %s\n\n', v);
catch
    fprintf('Warning: SPM not found.\n');
end

%% ========== 1. Set project root ==========
script_dir   = fileparts(mfilename('fullpath'));
project_root = fullfile(script_dir, '..', '..', '..');
results_dir  = fullfile(project_root, 'results');
if ~exist(results_dir, 'dir'), mkdir(results_dir); end

%% ========== 2. Common parameters (match Python version exactly) ==========
a_true    = -1.0;
x0_true   = 1.0;
v_input   = 0.0;
dt        = 0.1;
T_end     = 4.0;
N         = round(T_end / dt);   % 40

noise_std = exp(-2);
rng(42);

pi_y      = 2.0;
pi_x      = 8.0;
n_embed   = 4;
s_smooth  = 1.0;
a_init    = 0.0;
pC_a      = 1000;

fprintf('Parameter settings:\n');
fprintf('  a_true=%.2f  a_init=%.2f  pi_y=%.2f  pi_x=%.2f  pC=%.0f\n', ...
    a_true, a_init, pi_y, pi_x, pC_a);
fprintf('  n=%d  s=%.2f  dt=%.2f  T=%.1f  N=%d\n\n', n_embed, s_smooth, dt, T_end, N);

%% ========== 3. Generate data ==========
fprintf('Step 1: Generate data\n');

t_vec  = (0:N-1)' * dt;
x_true = zeros(N, 1);
x = x0_true;
for k = 1:N
    x_true(k) = x;
    x = x + dt * (a_true * x + v_input);
end
x_analytic = x0_true * exp(a_true * t_vec);
noise  = noise_std * randn(N, 1);
y_obs  = x_true + noise;

fprintf('  State range: [%.4f, %.4f]  Noise std: %.4f\n\n', ...
    min(x_true), max(x_true), std(y_obs - x_true));

%% ========== 4. Helper: build spm_DEM model (fixed parameter) ==========
function M = build_dem_model(a_val, pi_y, pi_x, n_embed, s_smooth, x0)
    M(1).f = @(x,v,P) P.a * x{1};
    M(1).g = @(x,v,P) x{1};
    M(1).x = {x0};
    M(1).V = pi_y;
    M(1).W = pi_x;
    M(1).n = n_embed;
    M(1).l = 1;
    M(1).pE.a = a_val;
    M(1).pC   = sparse(0, 0);    % no parameter update
    M(1).E.s  = s_smooth;
    M(1).E.n  = n_embed;
    M(1).E.d  = 1;
    M(1).E.nD = 1;
    M(1).E.nE = 1;               % single E-step pass (no param update)
    M(2).v = 0;
    M(2).V = exp(16);
end

%% ========== 5. Phase 1: D-step with a=0 ==========
fprintf('Step 2: Phase 1 — D-step (a=0, fixed)\n');

M1 = build_dem_model(a_init, pi_y, pi_x, n_embed, s_smooth, x0_true);
DEM1.M = M1;
DEM1.Y = y_obs';
DEM1.U = zeros(1, N);
DEM1.C = zeros(1, N);

DEM1_out = spm_DEM(DEM1);

x_phase1 = DEM1_out.qU.x{1}(1, 1:N)';
rmse1 = sqrt(mean((x_phase1 - x_true).^2));
fprintf('  State RMSE (a=0): %.4f\n\n', rmse1);

%% ========== 6. Phase 2: E-step — spm_nlsi_GN ==========
fprintf('Step 3: Phase 2 — E-step (spm_nlsi_GN)\n');
fprintf('  a_init=%.2f -> a_true=%.2f\n', a_init, a_true);

% Model for spm_nlsi_GN (uses f(x,u,P,M) interface, NOT cell arrays)
MG.f  = @(x, u, P, M) P.a * x;
MG.g  = @(x, u, P, M) x;
MG.x  = x0_true;
MG.m  = 1;       % number of inputs
MG.n  = 1;       % number of states
MG.l  = 1;       % number of outputs
MG.IS = 'spm_int_J';
MG.pE.a = a_init;
MG.pC   = struct('a', pC_a);    % prior covariance

% Input
UG.u  = zeros(N, 1);
UG.dt = dt;

% Observations
YG.y  = y_obs;
YG.dt = dt;
YG.Q  = {speye(1) / noise_std^2};

[Ep, Cp, ~, F_nlsi] = spm_nlsi_GN(MG, UG, YG);
a_est = Ep.a;
a_error = abs(a_est - a_true);

fprintf('  a_estimated = %.4f (true=%.4f, error=%.4f)\n\n', a_est, a_true, a_error);

%% ========== 7. Phase 3: D-step with estimated a ==========
fprintf('Step 4: Phase 3 — D-step (a=%.4f, fixed)\n', a_est);

M3 = build_dem_model(a_est, pi_y, pi_x, n_embed, s_smooth, x0_true);
DEM3.M = M3;
DEM3.Y = y_obs';
DEM3.U = zeros(1, N);
DEM3.C = zeros(1, N);

DEM3_out = spm_DEM(DEM3);

x_estimated = DEM3_out.qU.x{1}(1, 1:N)';
rmse3 = sqrt(mean((x_estimated - x_true).^2));
fprintf('  State RMSE (a=%.4f): %.4f\n\n', a_est, rmse3);

%% ========== 8. Build output arrays ==========
% VFE from Phase 3
if isfield(DEM3_out, 'F') && ~isempty(DEM3_out.F)
    F_val = DEM3_out.F(end);
    vfe_history = repmat(-F_val / N, N, 1);
else
    vfe_history = nan(N, 1);
end

% a_estimated history: step function (0 → a_est)
a_history = [repmat(a_init, N, 1)];   % Phase 1 used a_init
% For CSV: use a_est for all steps (final estimate)
a_history_csv = repmat(a_est, N, 1);

%% ========== 9. Accuracy evaluation ==========
fprintf('Step 5: Accuracy evaluation\n');
rmse_analytic = sqrt(mean((x_estimated - x_analytic).^2));
fprintf('  State RMSE (vs true trajectory):   %.4f\n', rmse3);
fprintf('  State RMSE (vs analytic solution): %.4f\n', rmse_analytic);
fprintf('  Parameter error |a_est - a_true|:  %.4f\n\n', a_error);

%% ========== 10. Save CSV ==========
fprintf('Step 6: Save to CSV\n');
output_path = fullfile(results_dir, 'comparison_matlab.csv');
fid = fopen(output_path, 'w');
if fid == -1, error('Cannot open: %s', output_path); end
fprintf(fid, 't,x_true,x_estimated,a_estimated,vfe\n');
for k = 1:N
    fprintf(fid, '%.6f,%.6f,%.6f,%.6f,%.6f\n', ...
        t_vec(k), x_true(k), x_estimated(k), a_history_csv(k), vfe_history(k));
end
fclose(fid);
fprintf('  Saved: %s\n\n', output_path);

%% ========== 11. Metadata ==========
meta_path = fullfile(results_dir, 'comparison_matlab_meta.txt');
fid = fopen(meta_path, 'w');
fprintf(fid, 'a_true=%.4f\n',      a_true);
fprintf(fid, 'a_estimated=%.4f\n', a_est);
fprintf(fid, 'rmse_state=%.6f\n',  rmse3);
fprintf(fid, 'n_order=%d\n',       n_embed);
fprintf(fid, 'pi_y=%.4f\n',        pi_y);
fprintf(fid, 'pi_x=%.4f\n',        pi_x);
fprintf(fid, 's=%.4f\n',           s_smooth);
fprintf(fid, 'dt=%.4f\n',          dt);
fprintf(fid, 'N=%d\n',             N);
fprintf(fid, 'noise_std=%.6f\n',   noise_std);
fclose(fid);

%% ========== 12. Plot ==========
fprintf('Step 7: Plot\n');
figure('Name', 'DEM MATLAB Result', 'NumberTitle', 'off', 'Visible', 'off');

subplot(3,1,1);
plot(t_vec, x_true,     'b-',  'LineWidth', 2, 'DisplayName', 'True x(t)');
hold on;
plot(t_vec, x_analytic, 'b--', 'LineWidth', 1, 'DisplayName', 'Analytic exp(-t)');
plot(t_vec, y_obs,      'k.',  'MarkerSize', 6, 'DisplayName', 'Obs y(t)');
plot(t_vec, x_phase1,   'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('Phase1 (a=%.1f)', a_init));
plot(t_vec, x_estimated,'r-',  'LineWidth', 2, 'DisplayName', sprintf('Phase3 (a=%.3f)', a_est));
xlabel('Time (s)'); ylabel('State');
title('State estimation'); legend('Location','best'); grid on;

subplot(3,1,2);
bar([a_init, a_est, a_true], 'FaceColor', [0.5 0.7 1]);
set(gca, 'XTickLabel', {'a\_init', 'a\_est (nlsi)', 'a\_true'});
ylabel('Value');
title(sprintf('Parameter estimation: a_est=%.4f, a_true=%.4f', a_est, a_true));
grid on;

subplot(3,1,3);
valid = ~isnan(vfe_history);
if any(valid)
    plot(t_vec(valid), vfe_history(valid), 'm-', 'LineWidth', 1.5);
    xlabel('Time (s)'); ylabel('VFE'); title('VFE (Phase 3)'); grid on;
end

sgtitle('DEM Comparison: MATLAB + SPM12 (3-phase)');
fig_path = fullfile(results_dir, 'comparison_matlab_plot.png');
saveas(gcf, fig_path);
fprintf('  Saved: %s\n', fig_path);

%% ========== Completion ==========
fprintf('\n===========================================\n');
fprintf('MATLAB DEM (3-phase) complete\n');
fprintf('===========================================\n');
fprintf('  Phase 1 RMSE (a=0):        %.4f\n', rmse1);
fprintf('  Phase 2 a_est (nlsi):       %.4f\n', a_est);
fprintf('  Phase 3 RMSE (a=%.4f): %.4f\n', a_est, rmse3);
fprintf('Output: %s\n', output_path);
fprintf('===========================================\n');
