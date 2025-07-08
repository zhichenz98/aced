clear;
clc;

output_dir = fullfile(pwd, 'data');
if ~exist(output_dir, 'dir')
    mkdir('data');
end

generate_data();

function generate_data()
    % Define parameter sets
    topo_ids = {'case30_topo0'};
    sigma_values = [0.01, 0.03, 0.05];
    bias_values = [0, 0.01];

    % Loop over all combinations of topo_id, sigma, and bias

    for topo_id = topo_ids
        for sigma = sigma_values
            for bias = bias_values
                fprintf('Processing topo_id=%s, sigma=%.2f, bias=%.2f\n', topo_id{1}, sigma, bias);
                run_opf(topo_id{1}, sigma, bias);
            end
        end
    end
end


%% === Runing OPF Function ===
function run_opf(topo_id, sigma, bias)
    rng(42);

    %% === Initialization ===
    mpc = loadcase(topo_id);

    % Load original Pd/Qd
    Pd_original = mpc.bus(:,3);
    Qd_original = mpc.bus(:,4);
    S_original = Pd_original + 1i * Qd_original;
    pf_angle = angle(S_original);

    % Settings
    num_scenarios = 1000;
    num_buses = size(mpc.bus, 1);
    bus_ids = mpc.bus(:, 1);
    bus_map = containers.Map(bus_ids, 1:num_buses);  % Map: bus_id → row index

    T_rows = {};
    scenario_count = 0;

    %% === Main Loop ===
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
        Qd_noisy(mask) = Pd_noisy(mask) .* tan(pf_angle(mask));
        
        % Apply back to case
        mpc_tmp.bus(:,3) = Pd_noisy;
        mpc_tmp.bus(:,4) = Qd_noisy;

        % Run OPF
        results = runopf(mpc_tmp);

        if results.success
            scenario_count = scenario_count + 1;

            row_data = struct();

            % Add Pd/Qd entries
            for row_id = 1:num_buses
                bus_id = bus_ids(row_id);
                row_data.(sprintf('Pd_bus%d', bus_id)) = results.bus(row_id, 3);
                row_data.(sprintf('Qd_bus%d', bus_id)) = results.bus(row_id, 4);
            end

            % Add Pg/Qg entries
            Pg_full = zeros(num_buses, 1);
            Qg_full = zeros(num_buses, 1);
            gen_buses = results.gen(:, 1);
            for gen_row_id = 1:length(gen_buses)
                bus_id = gen_buses(gen_row_id);
                row_id = bus_map(bus_id);
                Pg_full(row_id) = results.gen(gen_row_id, 2);
                Qg_full(row_id) = results.gen(gen_row_id, 3);
            end
            for row_id = 1:num_buses
                bus_id = bus_ids(row_id);
                row_data.(sprintf('Pg_bus%d', bus_id)) = Pg_full(row_id);
                row_data.(sprintf('Qg_bus%d', bus_id)) = Qg_full(row_id);
            end

            % Add V entris
            for row_id = 1:num_buses
                bus_id = bus_ids(row_id);
                v = results.bus(row_id, 8);
                theta = results.bus(row_id, 9);
                vr = v * cosd(theta);
                vi = v * sind(theta);
                row_data.(sprintf('Vr_bus%d', bus_id)) = vr;
                row_data.(sprintf('Vi_bus%d', bus_id)) = vi;
            end


            % Add objective
            row_data.Objective = results.f;

            T_rows{end+1} = row_data; %#ok<AGROW>
        else
            fprintf('Scenario %d failed to solve.\n', i);
        end
    end

    %% === Convert to table ===
    T_structs = [T_rows{:}];
    T_all = struct2table(T_structs);

    %% === Export ===
    output_dir = "data";
    filename = fullfile(output_dir, sprintf('results_%s_sigma%.2f_bias%.2f.csv', ...
        topo_id, sigma, bias));
    writetable(T_all, filename);
    fprintf('Saved %d scenarios to %s\n', scenario_count, filename);

    % Save Ybus and branch info
    edge_name = fullfile(output_dir, sprintf('%s_branches.csv', topo_id));
    g_name    = fullfile(output_dir, sprintf('%s_g.csv', topo_id));
    b_name    = fullfile(output_dir, sprintf('%s_b.csv', topo_id));

    Ybus_sparse = makeYbus(mpc);
    Ybus = full(Ybus_sparse);
    branch_set = mpc.branch(:, 1:2);

    writematrix(branch_set, edge_name);
    writematrix(real(Ybus), g_name);
    writematrix(imag(Ybus), b_name);
end