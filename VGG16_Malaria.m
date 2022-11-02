clc;
clear all;
close all;
tic;
%Load the pretrained Network

vg16 = vgg16;
layers = vg19.Layers;


% Modify the Last Layers
layers(39) = fullyConnectedLayer(2);
layers(41) = classificationLayer;


 matlabroot  = 'AcrimaTrainingRotated'
 Datasetpath = fullfile(matlabroot)
 FinalTrain  = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')
% 
 matlabroot  = 'AcrimaTestingRotated'
 Datasetpath = fullfile(matlabroot)
 FinalTest   = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')

% Finding number of images in each category for training
labelCountTrain = countEachLabel(FinalTrain)

% Finding number of images in each category for testing
labelCountTest = countEachLabel(FinalTest)


% Resize the images to the input size of first layer
inputSize = [224,224,3];

augimdsTest = augmentedImageDatastore(inputSize(1:2),FinalTest);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),FinalTrain);

% Retrain the network
options = trainingOptions('adam', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',10, ...   
    'InitialLearnRate',1e-4,'Shuffle','every-epoch');

MyNet = trainNetwork(augimdsTrain,layers,options);
save('VGG16Data.mat')
 toc;
 
 
 %% Program for Testing
 tic;
 
% % Classification validation
[YPred,scores] = classify(MyNet,augimdsTest);

%Accuracy calculation
YValidation = FinalTest.Labels;
accuracy = mean(YPred == YValidation)


% Plot confusion matrix
figure, plotconfusion(YValidation,YPred)
% 
toc;
%
%%
writematrix(YPred,'VGG19Data_YPred.txt','Delimiter','tab')
writematrix(scores,'VGG1Data_scores.txt','Delimiter','tab')
%%
 
