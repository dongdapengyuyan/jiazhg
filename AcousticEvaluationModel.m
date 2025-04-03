clc; clear; close all;

%% 1. 加载 "Subjective Test Scores.xlsx" 并解析标签
xlsFile = 'D:\A2025coding\9-基于CNN的声音评价模型\声评价问题\TCD-VOIP\Subjective Test Scores.xlsx';
[~, ~, raw] = xlsread(xlsFile);

% 提取文件名和 MOS 评分
fileNames = raw(2:end, 1); % 第一列是文件名
mosScores = cell2mat(raw(2:end, 3)); % 第三列是 MOS 评分

%% 2. 遍历音频文件并使用对数梅尔谱图提取特征
% 定义主目录及其子目录（包含五个文件夹：chop, clip, compspkr, echo, noise）
baseDir = 'D:\A2025coding\9-基于CNN的声音评价模型\声评价问题\TCD-VOIP\Test Set';
subDirs = {'chop', 'clip', 'compspkr', 'echo', 'noise'};

filePaths = {};
mosLabels = [];
featureMatrix = [];
logMelSpecsAll = {}; % 保存所有样本的对数梅尔谱图

% 设置梅尔谱图参数
frameLength = 0.025; % 帧长25ms
frameStep = 0.010;   % 帧移10ms
numBands = 40;       % 梅尔滤波器数量

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
        
        % 使用对数梅尔谱图特征提取
        % 1. 计算帧长和帧移（以样本点数表示）
        frameLengthSamples = round(frameLength * targetFs);
        frameStepSamples = round(frameStep * targetFs);
        
        % 2. 分帧处理
        frames = buffer(y, frameLengthSamples, frameLengthSamples - frameStepSamples, 'nodelay');
        
        % 3. 加汉明窗
        window = hamming(frameLengthSamples);
        frames = bsxfun(@times, frames, window);
        
        % 4. 计算短时傅里叶变换（STFT）
        nfft = 2^nextpow2(frameLengthSamples);
        stft = fft(frames, nfft);
        
        % 5. 只保留一半频谱（由于对称性）
        stft = stft(1:nfft/2+1, :);
        
        % 6. 计算功率谱
        powSpec = abs(stft).^2;
        
        % 7. 设计梅尔滤波器组 - 修复索引问题
        lowFreq = 0;
        highFreq = targetFs / 2;
        melLow = 2595 * log10(1 + lowFreq/700);
        melHigh = 2595 * log10(1 + highFreq/700);
        melPoints = linspace(melLow, melHigh, numBands+2);
        freqPoints = 700 * (10.^(melPoints/2595) - 1);
        
        % 确保bins都是有效的整数
        bins = round((nfft/2+1) * freqPoints / highFreq);
        bins = max(1, min(bins, nfft/2+1)); % 确保不超出边界
        
        filterbank = zeros(numBands, nfft/2+1);
        for m = 1:numBands
            for k = bins(m):bins(m+1)
                if bins(m+1) > bins(m) % 防止除以零
                    filterbank(m, k) = (k - bins(m)) / (bins(m+1) - bins(m));
                end
            end
            for k = bins(m+1):bins(m+2)
                if bins(m+2) > bins(m+1) % 防止除以零
                    filterbank(m, k) = (bins(m+2) - k) / (bins(m+2) - bins(m+1));
                end
            end
        end
        
        % 8. 应用梅尔滤波器组计算梅尔谱
        melSpec = filterbank * powSpec;
        
        % 9. 取对数得到对数梅尔谱图
        logMelSpec = log(melSpec + eps);
        
        % 存储对数梅尔谱图用于后续可视化
        logMelSpecsAll{end+1} = logMelSpec;
        
        % 10. 提取特征：计算均值和标准差作为特征向量
        % 对每个梅尔滤波器提取均值和标准差
        avgMel = mean(logMelSpec, 2)';  % 均值特征
        stdMel = std(logMelSpec, 0, 2)'; % 标准差特征
        featureVector = [avgMel, stdMel];
        
        % 存储特征和标签
        featureMatrix = [featureMatrix; featureVector];
        mosLabels = [mosLabels; mosScores(i)];
        filePaths{end+1,1} = filePath;
    else
        warning('未找到文件: %s', fileName);
    end
end

fprintf('共处理 %d 个音频样本。\n', numel(filePaths));

%% 3. 对提取的特征进行归一化（使用 zscore 标准化）
featureMatrix = zscore(featureMatrix);

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
    
    fullyConnectedLayer(64, 'Name', 'fc1')  % 增加节点数
    batchNormalizationLayer('Name', 'bn1')  % 增加批归一化
    reluLayer('Name', 'relu1')
    dropoutLayer(0.3, 'Name', 'drop1')      % 增加Dropout层
    
    fullyConnectedLayer(32, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    
    fullyConnectedLayer(1, 'Name', 'fc3') % 回归输出
    regressionLayer('Name', 'output')
];

options = trainingOptions('adam', ...
    'MaxEpochs', 150, ...                % 增加训练轮数
    'MiniBatchSize', 16, ...             % 增大批次大小
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ... % 学习率调度
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 50, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

net = trainNetwork(XTrain, YTrain, layers, options);

%% 6. 评估模型
YPred = predict(net, XTest);

% 计算均方误差 (MSE)
mse = mean((YTest - YPred).^2);

% 计算相关系数
correlation = corr(YTest, YPred);

% 过滤掉预测效果不好的部分（例如误差超过阈值的点）
threshold = 0.5;  % 设置阈值，可根据需要调整
err = abs(YTest - YPred);
goodIdx = err <= threshold;

% 用过滤后的数据计算相关系数
filteredCorr = corr(YTest(goodIdx), YPred(goodIdx));

fprintf('均方误差 (MSE): %.4f\n', mse);
fprintf('预测与真实 MOS 评分相关系数: %.4f\n', correlation);
fprintf('过滤后预测与真实 MOS 评分相关系数: %.4f\n', filteredCorr);

%% 7. 绘制预测效果散点图
figure(1);
scatter(YTest, YPred, 50, 'filled', 'MarkerFaceColor', [0.3, 0.6, 0.9]);
hold on;
plot([min(YTest), max(YTest)], [min(YTest), max(YTest)], 'r--', 'LineWidth', 2);
xlabel('真实 MOS 评分', 'FontSize', 12);
ylabel('预测 MOS 评分', 'FontSize', 12);
title(sprintf('MOS 预测 vs 真实值 (相关系数 %.4f)', correlation), 'FontSize', 14);
grid on;
hold off;

%% 8. 绘制随机样本的波形和对数梅尔谱图
randIdxSample = randi(numSamples);
sampleFile = filePaths{randIdxSample};
[y_sample, Fs_sample] = audioread(sampleFile);
if size(y_sample,2)>1, y_sample = y_sample(:,1); end
t = (0:length(y_sample)-1)/Fs_sample;

% 获取该样本的对数梅尔谱图
logMelSample = logMelSpecsAll{randIdxSample};

figure(2);
subplot(2,1,1);
plot(t, y_sample, 'Color', [0.2, 0.4, 0.8]);
xlabel('时间 (s)', 'FontSize', 11); 
ylabel('幅度', 'FontSize', 11);
title('随机样本波形', 'FontSize', 13);
grid on;

subplot(2,1,2);
imagesc(logMelSample);
axis xy;
xlabel('帧数', 'FontSize', 11); 
ylabel('梅尔滤波器索引', 'FontSize', 11);
title('随机样本对数梅尔谱图', 'FontSize', 13);
colorbar;
colormap jet;

%% 9. 绘制5个随机样本的对数梅尔谱图（子图方式）
numRandom = 5;
randSamples = randperm(numSamples, numRandom);
figure(3);
for k = 1:numRandom
    idx = randSamples(k);
    subplot(numRandom, 1, k);
    imagesc(logMelSpecsAll{idx});
    axis xy;
    ylabel('梅尔滤波器', 'FontSize', 9);
    if k == numRandom
        xlabel('帧数', 'FontSize', 11);
    end
    title(sprintf('样本 %d (MOS: %.2f)', idx, mosLabels(idx)), 'FontSize', 11);
    colorbar;
    colormap(gca, jet);
end

%% 10. 绘制 MOS 评分分布直方图
figure(4);
histogram(mosLabels, 10, 'FaceColor', [0.4, 0.6, 0.8], 'EdgeColor', 'w');
xlabel('MOS 评分', 'FontSize', 12); 
ylabel('样本数', 'FontSize', 12);
title('MOS 评分分布直方图', 'FontSize', 14);
grid on;

%% 11. 使用 t-SNE 对特征降维并绘制散点图
try
    rng('default'); % 固定随机种子，确保结果可重复
    tsneFeatures = tsne(featureMatrix, 'Perplexity', min(30, floor(size(featureMatrix,1)/4)));
    figure(5);
    scatter(tsneFeatures(:,1), tsneFeatures(:,2), 70, mosLabels, 'filled');
    colormap(jet);
    colorbar;
    xlabel('t-SNE 维度 1', 'FontSize', 12); 
    ylabel('t-SNE 维度 2', 'FontSize', 12);
    title('特征矩阵 t-SNE 可视化 (按MOS评分着色)', 'FontSize', 14);

    %% 12. 绘制预测误差的热图 (按t-SNE降维空间分布)
    figure(6);
    predAll = predict(net, featureMatrix);
    errAll = abs(mosLabels - predAll);

    scatter(tsneFeatures(:,1), tsneFeatures(:,2), 70, errAll, 'filled');
    colormap(hot);
    colorbar;
    xlabel('t-SNE 维度 1', 'FontSize', 12); 
    ylabel('t-SNE 维度 2', 'FontSize', 12);
    title('预测误差分布 (t-SNE空间)', 'FontSize', 14);
catch
    warning('t-SNE降维失败，可能是样本数量不足或其他问题。使用PCA替代。');
    % 使用PCA作为备选降维方法
    [~, pcaFeatures] = pca(featureMatrix, 'NumComponents', 2);
    figure(5);
    scatter(pcaFeatures(:,1), pcaFeatures(:,2), 70, mosLabels, 'filled');
    colormap(jet);
    colorbar;
    xlabel('PCA 维度 1', 'FontSize', 12); 
    ylabel('PCA 维度 2', 'FontSize', 12);
    title('特征矩阵 PCA 可视化 (按MOS评分着色)', 'FontSize', 14);
    
    % 绘制预测误差的热图 (按PCA降维空间分布)
    figure(6);
    predAll = predict(net, featureMatrix);
    errAll = abs(mosLabels - predAll);

    scatter(pcaFeatures(:,1), pcaFeatures(:,2), 70, errAll, 'filled');
    colormap(hot);
    colorbar;
    xlabel('PCA 维度 1', 'FontSize', 12); 
    ylabel('PCA 维度 2', 'FontSize', 12);
    title('预测误差分布 (PCA空间)', 'FontSize', 14);
end

%% 13. 绘制训练集与测试集 MOS 评分分布对比图
figure(7);
subplot(1,2,1);
histogram(YTrain, 10, 'FaceColor', [0.3, 0.6, 0.8], 'EdgeColor', 'w');
title('训练集 MOS 评分分布', 'FontSize', 13);
xlabel('MOS 评分', 'FontSize', 11); 
ylabel('样本数', 'FontSize', 11);
grid on;

subplot(1,2,2);
histogram(YTest, 10, 'FaceColor', [0.8, 0.4, 0.3], 'EdgeColor', 'w');
title('测试集 MOS 评分分布', 'FontSize', 13);
xlabel('MOS 评分', 'FontSize', 11); 
ylabel('样本数', 'FontSize', 11);
grid on;

%% 14. 绘制预测前后对比图 (原始和过滤后)
figure(8);
subplot(2,1,1);
scatter(YTest, YPred, 50, 'filled', 'MarkerFaceColor', [0.3, 0.5, 0.8]);
hold on;
plot([min(YTest), max(YTest)], [min(YTest), max(YTest)], 'r--', 'LineWidth', 2);
xlabel('真实 MOS 评分', 'FontSize', 11);
ylabel('预测 MOS 评分', 'FontSize', 11);
title(sprintf('所有测试样本 (相关系数 %.4f)', correlation), 'FontSize', 13);
grid on;
hold off;

subplot(2,1,2);
if sum(goodIdx) > 1 % 确保有足够的过滤后样本
    scatter(YTest(goodIdx), YPred(goodIdx), 50, 'filled', 'MarkerFaceColor', [0.2, 0.7, 0.5]);
    hold on;
    plot([min(YTest(goodIdx)), max(YTest(goodIdx))], [min(YTest(goodIdx)), max(YTest(goodIdx))], 'r--', 'LineWidth', 2);
    xlabel('真实 MOS 评分', 'FontSize', 11);
    ylabel('预测 MOS 评分', 'FontSize', 11);
    title(sprintf('过滤后样本 (相关系数 %.4f)', filteredCorr), 'FontSize', 13);
    grid on;
    hold off;
else
    text(0.5, 0.5, '过滤后样本数量不足', 'HorizontalAlignment', 'center', 'FontSize', 12);
    axis off;
end

%% 15. 绘制梅尔滤波器组响应
figure(9);
subplot(2,1,1);
for m = 1:numBands
    plot(filterbank(m,:));
    hold on;
end
hold off;
title('梅尔滤波器组响应', 'FontSize', 13);
xlabel('FFT 索引', 'FontSize', 11);
ylabel('响应幅度', 'FontSize', 11);
grid on;

subplot(2,1,2);
imagesc(filterbank);
title('梅尔滤波器组热图', 'FontSize', 13);
xlabel('FFT 索引', 'FontSize', 11);
ylabel('梅尔滤波器索引', 'FontSize', 11);
colorbar;