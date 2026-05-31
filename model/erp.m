%% ERP Stock Risk Classification Project
% This script classifies ERP inventory products into three risk levels:
% Low, Medium, and High.
%
% Required dataset:
%   stock_risk_dataset.csv
%
% Main steps:
%   1. Read and summarize the dataset
%   2. Clean missing values
%   3. Split data, then create engineered features
%   4. Train SVM, Decision Tree, and Random Forest models
%   5. Evaluate and compare the models
%   6. Save all generated figures in matlab_results

clear; clc; close all;
rng(42); % Makes the train/test split repeatable

%% 1. Read the CSV file
scriptFolder = fileparts(mfilename("fullpath"));
csvFile = fullfile(scriptFolder, "stock_risk_dataset.csv");

if ~isfile(csvFile)
    csvFile = fullfile(scriptFolder, "python", "stock_risk_dataset.csv");
end

if ~isfile(csvFile)
    error("Dataset file not found. Please place stock_risk_dataset.csv next to this script or inside the python folder.");
end

data = readtable(csvFile);

fprintf("\nERP Stock Risk Classification\n");
fprintf("Dataset loaded from: %s\n", csvFile);
fprintf("Rows: %d | Columns: %d\n\n", height(data), width(data));

%% 2. Display dataset summary
disp("First 5 rows of the dataset:");
disp(data(1:min(5, height(data)), :));

disp("Dataset summary:");
summary(data);

disp("Risk level distribution:");
disp(groupcounts(data, "risk_level"));

%% 3. Clean missing values if any
fprintf("\nCleaning missing values...\n");

for i = 1:width(data)
    colName = data.Properties.VariableNames{i};
    col = data.(colName);

    if isnumeric(col)
        if any(ismissing(col))
            replacementValue = median(col, "omitnan");
            data.(colName) = fillmissing(col, "constant", replacementValue);
            fprintf("Filled missing numeric values in %s with median %.4f\n", colName, replacementValue);
        end
    elseif iscategorical(col)
        if any(ismissing(col))
            modeValue = mode(col(~ismissing(col)));
            data.(colName) = fillmissing(col, "constant", modeValue);
            fprintf("Filled missing categorical values in %s with mode %s\n", colName, string(modeValue));
        end
    elseif isstring(col)
        if any(ismissing(col))
            knownValues = col(~ismissing(col));
            if isempty(knownValues)
                replacementValue = "Unknown";
            else
                replacementValue = string(mode(categorical(knownValues)));
            end
            data.(colName) = fillmissing(col, "constant", replacementValue);
            fprintf("Filled missing text values in %s\n", colName);
        end
    elseif iscellstr(col)
        missingRows = ismissing(col);
        if any(missingRows)
            knownValues = col(~missingRows);
            if isempty(knownValues)
                replacementValue = "Unknown";
            else
                replacementValue = string(mode(categorical(knownValues)));
            end
            data.(colName)(missingRows) = cellstr(replacementValue);
            fprintf("Filled missing text values in %s\n", colName);
        end
    end
end

disp("Missing value cleaning completed.");

%% 4. Target and split FIRST (before any feature engineering)
y = categorical(data.risk_level);
cv = cvpartition(y, "HoldOut", 0.20);

trainIdx = training(cv);
testIdx  = test(cv);

dataTrain = data(trainIdx, :);
dataTest  = data(testIdx, :);

yTrain = y(trainIdx);
yTest  = y(testIdx);

fprintf("\nTraining samples: %d\n", sum(trainIdx));
fprintf("Testing samples:  %d\n", sum(testIdx));

%% 5. Feature engineering on each split separately
% NOTE: min_stock_level is excluded from features to avoid leakage.
% stock_quantity - min_stock_level = reorder_gap which directly encodes
% the risk label. Keeping both stock_quantity AND min_stock_level lets
% any model reconstruct that rule trivially (100% accuracy is an artifact,
% not real generalisation). sales_speed_category is kept because it is
% derived only from weekly_sales and adds a useful discretised signal.
dataTrain = addFeatures(dataTrain);
dataTest  = addFeatures(dataTest);

%% 6. Build X matrices (min_stock_level intentionally omitted)
featureNames = {"stock_quantity", "weekly_sales", "supplier_delay_days", ...
                "price", "last_sale_days", "seasonality", ...
                "sales_speed_category"};

XTrain = extractX(dataTrain, featureNames);
XTest  = extractX(dataTest,  featureNames);

%% 7. Normalize using ONLY training stats (avoid test leakage)
[XTrain, mu, sigma] = zscore(XTrain);
% Replace zero std columns (constant features) to avoid NaN
sigma(sigma == 0) = 1;
XTest = (XTest - mu) ./ sigma;

%% 7b. Correlation heatmap (computed on training data only)
featureLabels = cellstr(featureNames);
corrMatrix = corr(XTrain);

figHeatmap = figure("Name", "Feature Correlation Heatmap");
hm = heatmap(featureLabels, featureLabels, corrMatrix);
hm.Title = "Feature Correlation Heatmap";

%% 8. Train SVM, Decision Tree, and Random Forest
fprintf("\nTraining models...\n");

% Decision Tree
treeModel = fitctree(XTrain, yTrain);

% Random Forest (Bagged Trees)
rfModel = fitcensemble(XTrain, yTrain, "Method", "Bag");

% SVM (multiclass, one-vs-all, RBF kernel)
svmTemplate = templateSVM("KernelFunction", "rbf", "Standardize", true);
svmModel    = fitcecoc(XTrain, yTrain, "Learners", svmTemplate, "Coding", "onevsall");

fprintf("Model training completed.\n");

%% 9. 5-Fold Cross Validation
cvTree = crossval(treeModel, "KFold", 5);
treeLoss = kfoldLoss(cvTree);

cvRF = crossval(rfModel, "KFold", 5);
rfLoss = kfoldLoss(cvRF);

cvSVM = crossval(svmModel, "KFold", 5);
svmLoss = kfoldLoss(cvSVM);

fprintf("\n===== 5-Fold Cross Validation (on training set) =====\n");
fprintf("Decision Tree Accuracy: %.2f%%\n", (1 - treeLoss) * 100);
fprintf("Random Forest Accuracy: %.2f%%\n", (1 - rfLoss)  * 100);
fprintf("SVM Accuracy:           %.2f%%\n", (1 - svmLoss) * 100);

%% 10. Predict on held-out test set
yPredTree = predict(treeModel, XTest);
yPredRF   = predict(rfModel,   XTest);
yPredSVM  = predict(svmModel,  XTest);

%% 11. Calculate Accuracy, Precision, Recall, F1
classNames = categories(y);

metricsTree = calculateMetrics(yTest, yPredTree, classNames);
metricsRF   = calculateMetrics(yTest, yPredRF,   classNames);
metricsSVM  = calculateMetrics(yTest, yPredSVM,  classNames);

%% 12. Comparison table
modelNames     = {'Decision Tree'; 'Random Forest'; 'SVM'};
accuracyValues = [metricsTree.Accuracy;  metricsRF.Accuracy;  metricsSVM.Accuracy];
precisionValues= [metricsTree.Precision; metricsRF.Precision; metricsSVM.Precision];
recallValues   = [metricsTree.Recall;    metricsRF.Recall;    metricsSVM.Recall];
f1Values       = [metricsTree.F1Score;   metricsRF.F1Score;   metricsSVM.F1Score];

comparisonTable = table(modelNames, accuracyValues, precisionValues, recallValues, f1Values, ...
    'VariableNames', {'Model','Accuracy','Precision','Recall','F1_Score'});

fprintf("\nModel comparison table (held-out test set):\n");
disp(comparisonTable);

%% 13. Save figures
resultsFolder = fullfile(scriptFolder, "matlab_results");
if ~exist(resultsFolder, "dir")
    mkdir(resultsFolder);
end

% Heatmap (created earlier, save now)
saveas(figHeatmap, fullfile(resultsFolder, "correlation_heatmap.png"));

% Accuracy bar chart
figAccuracy = figure("Name", "Accuracy Comparison");
bar(comparisonTable.Accuracy);
set(gca, "XTickLabel", comparisonTable.Model);
ylim([0 1]);
ylabel("Accuracy");
title("Accuracy Comparison of Classification Models");
grid on;
saveas(figAccuracy, fullfile(resultsFolder, "accuracy_comparison.png"));

% Confusion matrices
figTree = figure("Name", "Decision Tree Confusion Matrix");
confusionchart(yTest, yPredTree);
title("Confusion Matrix - Decision Tree");
saveas(figTree, fullfile(resultsFolder, "confusion_matrix_decision_tree.png"));

figRF = figure("Name", "Random Forest Confusion Matrix");
confusionchart(yTest, yPredRF);
title("Confusion Matrix - Random Forest");
saveas(figRF, fullfile(resultsFolder, "confusion_matrix_random_forest.png"));

figSVM = figure("Name", "SVM Confusion Matrix");
confusionchart(yTest, yPredSVM);
title("Confusion Matrix - SVM");
saveas(figSVM, fullfile(resultsFolder, "confusion_matrix_svm.png"));

% Feature importance (Random Forest)
figImp = figure("Name", "Feature Importance");
imp = predictorImportance(rfModel);
bar(imp);
set(gca, 'XTick', 1:numel(featureNames));
set(gca, 'XTickLabel', featureNames);
xtickangle(45);
ylabel('Importance');
title('Random Forest Feature Importance');
saveas(figImp, fullfile(resultsFolder, "feature_importance_random_forest.png"));

fprintf("\nFigures saved in: %s\n", resultsFolder);

%% 14. Best model
[bestAccuracy, bestIndex] = max(comparisonTable.Accuracy);
bestModel = comparisonTable.Model{bestIndex};
fprintf("\nBest model based on test Accuracy: %s\n", bestModel);
fprintf("Best Accuracy: %.2f%%\n", bestAccuracy * 100);

%% ---- Helper functions ----

function metrics = calculateMetrics(yTrue, yPred, classNames)
    confusionMat = confusionmat(yTrue, yPred, "Order", categorical(classNames));
    tp = diag(confusionMat);
    fp = sum(confusionMat, 1)' - tp;
    fn = sum(confusionMat, 2) - tp;
    precPerClass = tp ./ (tp + fp);
    recPerClass  = tp ./ (tp + fn);
    f1PerClass   = 2 .* (precPerClass .* recPerClass) ./ (precPerClass + recPerClass);
    precPerClass(isnan(precPerClass)) = 0;
    recPerClass(isnan(recPerClass))   = 0;
    f1PerClass(isnan(f1PerClass))     = 0;
    metrics.Accuracy  = mean(yTrue == yPred);
    metrics.Precision = mean(precPerClass);
    metrics.Recall    = mean(recPerClass);
    metrics.F1Score   = mean(f1PerClass);
end

function d = addFeatures(d)
    d.daily_sales = d.weekly_sales ./ 7;
    safe = d.daily_sales;
    safe(safe <= 0) = eps;
    d.stock_coverage_days = d.stock_quantity ./ safe;
    d.reorder_gap = d.stock_quantity - d.min_stock_level;
    d.sales_speed_category = zeros(height(d), 1);
    d.sales_speed_category(d.weekly_sales <= 5)  = 1;
    d.sales_speed_category(d.weekly_sales > 5 & d.weekly_sales <= 15) = 2;
    d.sales_speed_category(d.weekly_sales > 15)  = 3;
end

function X = extractX(d, featureNames)
    X = zeros(height(d), numel(featureNames));
    for i = 1:numel(featureNames)
        col = d.(featureNames{i});
        if isnumeric(col)
            X(:, i) = col;
        else
            X(:, i) = grp2idx(categorical(col));
        end
    end
end

save("best_model.mat", ...
    "rfModel", ...
    "mu", ...
    "sigma", ...
    "featureNames");