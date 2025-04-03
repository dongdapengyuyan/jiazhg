clc; clear; close all;

%% 1. 加载 "Subjective Test Scores.xlsx" 并解析标签
xlsFile = 'D:\2025fighting\声评价问题\TCD-VOIP\Subjective Test Scores.xlsx';
[~, ~, raw] = xlsread(xlsFile);

% 提取文件名和 MOS 评分
fileNames = raw(2:end, 1); % 第一列是文件名
mosScores = cell2mat(raw(2:end, 3)); % 第三列是 MOS 评分

%% 2. 遍历音频文件并提取特征
% 定义主目录及其子目录（包含五个文件夹）
baseDir = 'D:\2025fighting\声评价问题\TCD-VOIP\Test Set';
subDirs = {'chop', 'clip', 'compspkr', 'echo', 'noise'};

filePaths = {};
mosLabels = [];
featureMatrix = [];

for i = 1:length(fileNames)
    fileName = fileNames{i};
    fileFound = false;
    
    % 遍历所有子目录寻找目标文件
    for j = 1:length(subDirs)
        filePath = fullfile(baseDir, subDirs{j}, fileName);
        if exist(filePath, 'file')
            fileFound = true;
            break;
        end
    end
    
    if fileFound
        [y, Fs] = audioread(filePath);
        
        % 统一采样率为16kHz
        targetFs = 16000;
        if Fs ~= targetFs
            y = resample(y, targetFs, Fs);
        end
        
        % 若为多声道，则取第一通道
        if size(y,2) > 1
            y = y(:,1);
        end
        
        % 提取 MFCC 特征（13维均值+标准差构成26维特征）
        coeffs = mfcc(y, targetFs, 'NumCoeffs', 13);
        avgCoeffs = mean(coeffs, 1);
        stdCoeffs = std(coeffs, 0, 1);
        featureVector = [avgCoeffs, stdCoeffs];
        
        % 存储特征和标签
        featureMatrix = [featureMatrix; featureVector];
        mosLabels = [mosLabels; mosScores(i)];
        filePaths{end+1,1} = filePath;
    else
        warning('未找到文件: %s', fileName);
    end
end

fprintf('共处理 %d 个音频样本。\n', numel(filePaths));

%% 3. 归一化特征
featureMatrix = normalize(featureMatrix, 'range');

%% 4. 划分训练集和测试集（80%/20%）
numSamples = size(featureMatrix, 1);
trainRatio = 0.8;
randIdx = randperm(numSamples);
trainIdx = randIdx(1:round(trainRatio * numSamples));
testIdx = randIdx(round(trainRatio * numSamples) + 1:end);

XTrain = featureMatrix(trainIdx, :);
YTrain = mosLabels(trainIdx);

XTest = featureMatrix(testIdx, :);
YTest = mosLabels(testIdx);

%% 5. 构建并训练回归模型
layers = [
    featureInputLayer(size(XTrain,2), 'Name', 'input')
    
    fullyConnectedLayer(32, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    
    fullyConnectedLayer(16, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    
    fullyConnectedLayer(1, 'Name', 'fc3') % 回归输出
    regressionLayer('Name', 'output')
];

options = trainingOptions('adam', ...
    'MaxEpochs', 100, ...
    'MiniBatchSize', 8, ...
    'InitialLearnRate', 0.001, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

net = trainNetwork(XTrain, YTrain, layers, options);

%% 6. 评估模型
YPred = predict(net, XTest);

% 计算均方误差 (MSE)
mse = mean((YTest - YPred).^2);
fprintf('均方误差 (MSE): %.4f\n', mse);

% 计算相关系数
correlation = corr(YTest, YPred);
fprintf('预测与真实 MOS 评分相关系数: %.4f\n', correlation);

% 绘制散点图
figure;
scatter(YTest, YPred, 'filled');
xlabel('真实 MOS 评分');
ylabel('预测 MOS 评分');
title('MOS 预测 vs 真实值');
grid on;
