function SimulatedPLs(OPC, opc_micro_contact, Real_distance, OPC_ran, numSamples, num_subset)

if nargin < 6 || isempty(num_subset) %the number of simulated PLs/ sample
    num_subset = 50;
end

if nargin < 5 || isempty(numSamples) %How many time you are going to simulate 
    numSamples = 100;
end

if nargin < 4 || isempty(OPC_ran) % Random positions on the OPC surface
    % Frequency of the grating in cycles per pixel: Here 0.01 cycles per pixel:
    OPC_ran = 1;
end

if nargin < 3 || isempty(Real_distance) % Random positions on the OPC surface
    % Frequency of the grating in cycles per pixel: Here 0.01 cycles per pixel:
    Real_distance = 1;
end

if nargin < 2 || isempty(opc_micro_contact) % input data of where microglia and OPCs are contacting eachother
    % Tilt angle of the grating:
    opc_micro_contact = 0;
end

if nargin < 1 || isempty(opc_micro_contact) % input data of where microglia and OPCs are contacting eachother
    % Tilt angle of the grating:
    OPC = 0;
end

% Initialize the storage for minimum distances for each sample
sampled_min_distances = zeros(numSamples, size(opc_micro_contact, 1));

% Sampling and distance calculation process
for j = 1:numSamples
    % Randomly sample 50 positions from random_PL_position
    indices = randperm(size(OPC_ran, 1), num_subset);
    subRanPoints = OPC_ran(indices, :);

    % Calculate the minimum distances for this subset
    for i = 1:size(opc_micro_contact, 1)
        diffs = bsxfun(@minus, subRanPoints, opc_micro_contact(i, :));
        distances = sqrt(sum(diffs.^2, 2));
        sampled_min_distances(j, i) = min(distances);
    end
end

fileName = append('OPC_',string(OPC),'_simulatedDistances.csv');
data=reshape(sampled_min_distances,size(sampled_min_distances,2)*size(sampled_min_distances,1),1);

%Saving and plotting data
writematrix(data,fileName);

avg_real_distance = mean(Real_distance)/1000;
Samp_data = data/1000;

histogram(Samp_data,'Normalization','cdf');
hold;
line([avg_real_distance, avg_real_distance], ylim, 'Color', 'r');
line([mean(Samp_data), mean(Samp_data)], ylim, 'Color', 'b');
histogram(Real_distance/1000,'Normalization','cdf');
title('Distribution of Minimum Distances');
xlabel('Minimum Distance (um)');
ylabel('Cumulative  Frequency');
legend('Simulated Data', 'Observed Data');
