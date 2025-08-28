clc;
clear;

generate_data();

function generate_data()

    case_name = 'case39'; 
    output_dir = fullfile(pwd, 'data', case_name);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    mpc = loadcase(case_name);
    save_env(mpc, case_name, output_dir);

    pattern = load_or_make_pattern('sorted_profile_data.csv');

    for scenario_idx = 0:10
        run_opf(case_name, output_dir, scenario_idx, pattern);
    end

end


function pattern = load_or_make_pattern(csv_path)
    raw_table = readtable(csv_path);
    A = table2array(raw_table);
    base = mean(A, 2);
    base = max(base, 1e-6);
    pattern = A ./ base;
    shrink_factor = 0.5;
    pattern = 1 + shrink_factor * (pattern - 1);
    pattern = min(max(pattern, 0.5), 1.5);
end


function run_opf(case_name, output_dir, scenario_idx, pattern)
    
    BUS_I=1; BUS_TYPE=2; PD=3; QD=4; VMAX=12; VMIN=13;
    F_BUS=1; T_BUS=2; BR_R=3; BR_X=4; RATE_A=6;
    GEN_BUS=1; PMAX=9; PMIN=10; QMAX=4; QMIN=5;
    PG=2; QG=3; VM=8; VA=9; LAM_P=14; LAM_Q=15; MU_VMAX=16; MU_VMIN=17;
    MU_SF=18; MU_ST=19; MU_PMAX=22; MU_PMIN=23; MU_QMAX=24; MU_QMIN=25;
    
    rng(1000 + scenario_idx);
    
    mpc = loadcase(case_name); 
    num_gens = size(mpc.gen, 1);
    num_buses = size(mpc.bus, 1);
    bus_ids = mpc.bus(:, 1);
    bus_map = containers.Map(bus_ids, 1:num_buses);
    
    % generate load data
    num_samples = size(pattern, 2);
    pattern = pattern(1:num_buses, :);
    
    pd_base = mpc.bus(:, PD);
    qd_base = mpc.bus(:, QD);

    data_p = pd_base .* pattern;
    
    if scenario_idx > 0
        pd_rel_sigma = 0.05;
        pd_abs_sigma = 0.02;
        pd_std_dev   = pd_rel_sigma * abs(data_p) + pd_abs_sigma;
        pd_noise     = pd_std_dev .* randn(size(data_p));
        data_p    = data_p + pd_noise;
    end
    
    data_p = max(data_p, 0);

    data_q = qd_base .* pattern;
    
    qd_rel_sigma = 0.02;
    qd_abs_sigma = 0;
    qd_std_dev   = qd_rel_sigma * abs(data_q) + qd_abs_sigma;
    qd_noise     = qd_std_dev .* randn(size(data_q));
    data_q    = data_q + qd_noise;
      
    T_rows = cell(1, num_samples);
    sample_count = 0;  
    
    for i = 1:num_samples

        sample_count = sample_count + 1;

        mpc = loadcase(case_name);

        mpc.bus(:, PD) = data_p(:, i);
        mpc.bus(:, QD) = data_q(:, i);

        % Run OPF
        results = runopf(mpc, mpoption('verbose',0,'out.all',0));

        if results.success
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

            T_rows{sample_count} = row_data;
        else
            fprintf('Time series %d failed to solve.\n', i);
        end
    end

    % ---- Convert to table ----
    T_structs = [T_rows{1:sample_count}];
    T_all = struct2table(T_structs);

    % ---- Export ----
    filename = fullfile(output_dir, sprintf('results_%s_scenario_%02d.csv', ...
        case_name, scenario_idx));
    writetable(T_all, filename);
    fprintf('  -> Saved %d time series of scenario %d to %s\n', sample_count, scenario_idx, filename);

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

    P_g_base = mpc.gen(:, PG);
    Q_g_base = mpc.gen(:, QG);
    P_d_base = mpc.bus(:, PD);
    Q_d_base = mpc.bus(:, QD);
    V_m_base = mpc.bus(:, VM);
    V_angle_base = mpc.bus(:, VA);
    V_r_base = V_m_base .* cosd(V_angle_base);
    V_i_base = V_m_base .* sind(V_angle_base);

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

    env.P_g_base = P_g_base;
    env.Q_g_base = Q_g_base;
    env.P_d_base = P_d_base;
    env.Q_d_base = Q_d_base;
    env.V_m_base = V_m_base;
    env.V_angle_base = V_angle_base;
    env.V_r_base = V_r_base;
    env.V_i_base = V_i_base;

    mat_name = fullfile(output_dir, sprintf('%s_env.mat', topo_id));
    save(mat_name, '-struct', 'env');

    fprintf('  -> env saved: %s\n', mat_name);
end