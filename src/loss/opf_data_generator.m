clear;
clc;

rng(42);

generate_data();

function generate_data()
    % Define parameter sets
    topo_ids = {'case39'};
    sigma_values = [0.01, 0.03, 0.05];
    bias_values = [0];

    % Loop over all combinations of topo_id, sigma, and bias

    for k = 1:numel(topo_ids)
        topo_id = topo_ids{k}; 
        mpc = loadcase(topo_id);
        
        output_dir = fullfile(pwd, 'data', topo_id);
        if ~exist(output_dir, 'dir')
            mkdir(output_dir);
        end

        save_env(mpc, topo_id, output_dir);

        for sigma = sigma_values
            for bias = bias_values
                fprintf('Processing topo_id=%s, sigma=%.2f, bias=%.2f\n', topo_id, sigma, bias);
                run_opf(topo_id, sigma, bias, output_dir);
            end
        end
    end
end


function run_opf(topo_id, sigma, bias, output_dir)
    BUS_I=1; BUS_TYPE=2; PD=3; QD=4; VMAX=12; VMIN=13;
    F_BUS=1; T_BUS=2; BR_R=3; BR_X=4; RATE_A=6;
    GEN_BUS=1; PMAX=9; PMIN=10; QMAX=4; QMIN=5;
    PG=2; QG=3; VM=8; VA=9; LAM_P=14; LAM_Q=15; MU_VMAX=16; MU_VMIN=17;
    MU_SF=18; MU_ST=19; MU_PMAX=22; MU_PMIN=23; MU_QMAX=24; MU_QMIN=25;

    % ---- Initialization ----
    mpc = loadcase(topo_id);

    % Load original Pd/Qd
    Pd_original = mpc.bus(:, PD);
    Qd_original = mpc.bus(:, QD);
    S_original = Pd_original + 1i * Qd_original;
    pf_angle = angle(S_original);

    % Settings
    num_scenarios = 5000;
    num_buses = size(mpc.bus, 1);
    bus_ids = mpc.bus(:, 1);
    bus_map = containers.Map(bus_ids, 1:num_buses);  % Map: bus_id → row index

    T_rows = cell(1, num_scenarios);
    scenario_count = 0;

    % ---- Main Loop ----
    for i = 1:num_scenarios
        mpc_tmp = mpc;

        % Generate noise vector
        noise = 1 + sigma * randn(num_buses, 1);
        
        % Create mask for buses with original Pd > 0
        mask = Pd_original > 0;
        
        % Initialize noisy Pd/Qd
        Pd_noisy = Pd_original;
        Qd_noisy = Qd_original;
        
        % Only modify buses with Pd > 0
        Pd_noisy(mask) = Pd_original(mask) .* noise(mask) + bias;
        Pd_noisy = max(Pd_noisy, 0);
        Qd_noisy(mask) = Pd_noisy(mask) .* tan(pf_angle(mask));
        
        % Apply back to case
        mpc_tmp.bus(:, PD) = Pd_noisy;
        mpc_tmp.bus(:, QD) = Qd_noisy;

        % Run OPF
        results = runopf(mpc_tmp, mpoption('verbose',0,'out.all',0));

        if results.success
            scenario_count = scenario_count + 1;

            row_data = struct();

            % Add Pd/Qd entries
            for row_id = 1:num_buses
                bus_id = bus_ids(row_id);
                row_data.(sprintf('p_d_%d', bus_id)) = results.bus(row_id, PD);
                row_data.(sprintf('q_d_%d', bus_id)) = results.bus(row_id, QD);
            end

            % Add Pg/Qg entries
            Pg_full = zeros(num_buses, 1);
            Qg_full = zeros(num_buses, 1);
            gen_buses = results.gen(:, GEN_BUS);
            for gen_row_id = 1:length(gen_buses)
                bus_id = gen_buses(gen_row_id);
                row_id = bus_map(bus_id);
                Pg_full(row_id) = results.gen(gen_row_id, PG);
                Qg_full(row_id) = results.gen(gen_row_id, QG);
            end
            for row_id = 1:num_buses
                bus_id = bus_ids(row_id);
                row_data.(sprintf('p_g_%d', bus_id)) = Pg_full(row_id);
                row_data.(sprintf('q_g_%d', bus_id)) = Qg_full(row_id);
            end

            % Add V entris
            for row_id = 1:num_buses
                bus_id = bus_ids(row_id);
                v = results.bus(row_id, VM);
                theta = results.bus(row_id, VA); %deg
                vr = v * cosd(theta);
                vi = v * sind(theta);
                row_data.(sprintf('v_r_%d', bus_id)) = vr;
                row_data.(sprintf('v_i_%d', bus_id)) = vi;
            end
            
            % Add lam_p and lam_q
            for row_id = 1:num_buses
                bus_id = bus_ids(row_id);
                row_data.(sprintf('lam_p_%d', bus_id)) = results.bus(row_id, LAM_P);
                row_data.(sprintf('lam_q_%d', bus_id)) = results.bus(row_id, LAM_Q);
            end

            % Add mu_v_u and mu_v_d
            for row_id = 1:num_buses
                bus_id = bus_ids(row_id);
                row_data.(sprintf('mu_v_u_%d', bus_id)) = results.bus(row_id, MU_VMAX);
                row_data.(sprintf('mu_v_d_%d', bus_id)) = results.bus(row_id, MU_VMIN);
            end

            % Add mu_p_u, mu_p_d, mu_q_u, and mu_p_d
            mu_p_u_full = zeros(num_buses, 1);
            mu_p_d_full = zeros(num_buses, 1);
            mu_q_u_full = zeros(num_buses, 1);
            mu_q_d_full = zeros(num_buses, 1);
            for gen_row_id = 1:length(gen_buses)
                bus_id = gen_buses(gen_row_id);
                row_id = bus_map(bus_id);
                mu_p_u_full(row_id) = results.gen(gen_row_id, MU_PMAX);
                mu_p_d_full(row_id) = results.gen(gen_row_id, MU_PMIN);
                mu_q_u_full(row_id) = results.gen(gen_row_id, MU_QMAX);
                mu_q_d_full(row_id) = results.gen(gen_row_id, MU_QMIN);
            end
            for row_id = 1:num_buses
                bus_id = bus_ids(row_id);
                row_data.(sprintf('mu_p_u_%d', bus_id)) = mu_p_u_full(row_id);
                row_data.(sprintf('mu_p_d_%d', bus_id)) = mu_p_d_full(row_id);
                row_data.(sprintf('mu_q_u_%d', bus_id)) = mu_q_u_full(row_id);
                row_data.(sprintf('mu_q_d_%d', bus_id)) = mu_q_d_full(row_id);
            end

            % Add mu_sf and mu_st todo
            num_branch = size(results.branch, 1);
            for br_row_id = 1:num_branch
                f_bus_id = results.branch(br_row_id, F_BUS);
                t_bus_id = results.branch(br_row_id, T_BUS);
                row_data.(sprintf('mu_sf_line%d_f%d_t%d', br_row_id, f_bus_id, t_bus_id)) = results.branch(br_row_id, MU_SF);
                row_data.(sprintf('mu_st_line%d_f%d_t%d', br_row_id, f_bus_id, t_bus_id)) = results.branch(br_row_id, MU_ST);
            end

            % Add objective
            row_data.Objective = results.f;

            T_rows{scenario_count} = row_data;
        else
            fprintf('Scenario %d failed to solve.\n', i);
        end
    end

    % ---- Convert to table ----
    T_structs = [T_rows{1:scenario_count}];
    T_all = struct2table(T_structs);

    % ---- Export ----
    filename = fullfile(output_dir, sprintf('results_%s_sigma%.2f_bias%.2f.csv', ...
        topo_id, sigma, bias));
    writetable(T_all, filename);
    fprintf('Saved %d scenarios to %s\n', scenario_count, filename);

end


function save_env(mpc, topo_id, output_dir)
    BUS_I=1; BUS_TYPE=2; PD=3; QD=4; VMAX=12; VMIN=13;
    F_BUS=1; T_BUS=2; BR_R=3; BR_X=4; RATE_A=6;
    GEN_BUS=1; PMAX=9; PMIN=10; QMAX=4; QMIN=5;
    PG=2; QG=3; VM=8; VA=9; LAM_P=14; LAM_Q=15; MU_VMAX=16; MU_VMIN=17;
    MU_SF=18; MU_ST=19; MU_PMAX=22; MU_PMIN=23; MU_QMAX=24; MU_QMIN=25;

    % ---- Ybus / G / B ----
    [Ybus, ~, ~] = makeYbus(mpc.baseMVA, mpc.bus, mpc.branch);
    G = full(real(Ybus));
    B = full(imag(Ybus));

    writematrix(mpc.branch(:,1:2), fullfile(output_dir, sprintf('%s_branches.csv', topo_id)));
    writematrix(G, fullfile(output_dir, sprintf('%s_g.csv', topo_id)));
    writematrix(B, fullfile(output_dir, sprintf('%s_b.csv', topo_id)));

    % ---- Branch & line admittance ----
    f = mpc.branch(:, F_BUS) - 1;
    t = mpc.branch(:, T_BUS) - 1;
    R = mpc.branch(:, BR_R);
    X = mpc.branch(:, BR_X);
    y_line = 1 ./ (R + 1i*X);
    G_line = real(y_line);
    B_line = imag(y_line);

    % ---- Limits ----
    S_max = mpc.branch(:, RATE_A);
    V_min = mpc.bus(:, VMIN);
    V_max = mpc.bus(:, VMAX);
    P_g_min = mpc.gen(:, PMIN);
    P_g_max = mpc.gen(:, PMAX);
    Q_g_min = mpc.gen(:, QMIN);
    Q_g_max = mpc.gen(:, QMAX);

    % ---- Index sets ----
    gen_bus_idx   = mpc.gen(:, GEN_BUS) - 1;
    load_bus_idx  = find(mpc.bus(:, PD) ~= 0 | mpc.bus(:, QD) ~= 0) - 1;
    slack_bus_idx = find(mpc.bus(:, BUS_TYPE) == 3) - 1;

    baseMVA = mpc.baseMVA;

    % ---- Save .mat ----
    env = struct();
    env.G = G; env.B = B;
    env.branches = [f, t];
    env.G_line = G_line; env.B_line = B_line;
    env.S_max = S_max;
    env.V_min = V_min; env.V_max = V_max;
    env.P_g_min = P_g_min; env.P_g_max = P_g_max;
    env.Q_g_min = Q_g_min; env.Q_g_max = Q_g_max;
    env.gen_bus_idx = gen_bus_idx;
    env.load_bus_idx = load_bus_idx;
    env.slack_bus_idx = slack_bus_idx;
    env.baseMVA = baseMVA;

    mat_name = fullfile(output_dir, sprintf('%s_env.mat', topo_id));
    save(mat_name, '-struct', 'env');

    fprintf('  -> env saved: %s\n', mat_name);
end