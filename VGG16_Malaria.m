clc;
clear all;
close all;
tic;
%Load the pretrained Network
% vg16 = vgg16;
% layers = vg16.Layers;

 vg19 = vgg19;
layers = vg19.Layers;

layers(45) = fullyConnectedLayer(3);
layers(47) = classificationLayer;

% Modify the Last Layers
% layers(39) = fullyConnectedLayer(3);
% layers(41) = classificationLayer;

matlabroot = 'Dataset_3'
Datasetpath = fullfile(matlabroot)
Data  = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')


%Split the glaucoma images from Data in the ratio 70:30 and normal images also 
[Data_GA7, Data_GA3] = splitEachLabel(Data,0.7,'Include','glaucoma_A1')
[Data_GE7, Data_GE3] = splitEachLabel(Data,0.7,'Include','glaucoma_E1')
[Data_N7, Data_N3] = splitEachLabel(Data,0.7,'Include','normal1')
% 

%Final Training set
FinalTrain = imageDatastore(cat(1,Data_GA7.Files,Data_GE7.Files,Data_N7.Files))
FinalTrain.Labels = cat(1,Data_GA7.Labels,Data_GE7.Labels,Data_N7.Labels)

% Final Testing set
FinalTest = imageDatastore(cat(1,Data_GA3.Files,Data_GE3.Files,Data_N3.Files))
FinalTest.Labels = cat(1,Data_GA3.Labels,Data_GE3.Labels,Data_N3.Labels)

%  matlabroot  = 'AcrimaTrainingRotated'
% Datasetpath = fullfile(matlabroot)
% FinalTrain  = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')
% 
% matlabroot  = 'AcrimaTestingRotated'
% Datasetpath = fullfile(matlabroot)
% FinalTest   = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')

%  matlabroot = 'D:\poonguzhali\3cell_imagesblu'
% Datasetpath = fullfile(matlabroot)
% Data  = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')
% 
% % % 
% %Split the glaucoma images from Data in the ratio 80:20 and normal images also 
% [Data_G80, Data_G20] = splitEachLabel(Data,0.8,'Include','Parasitized')
% [Data_N80, Data_N20] = splitEachLabel(Data,0.8,'Include','Uninfected')
% 
% FinalTrain = imageDatastore(cat(1,Data_G80.Files,Data_N80.Files))
% FinalTrain.Labels = cat(1,Data_G80.Labels,Data_N80.Labels)
% 
% % Final Testing set
% FinalTest = imageDatastore(cat(1,Data_G20.Files,Data_N20.Files));
% FinalTest.Labels = cat(1,Data_G20.Labels,Data_N20.Labels)


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
save('VGG19Dataset3.mat')
 toc;
 
 
 %%
 tic;
 %load('VGG16Malaria.mat')
% 
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
writematrix(YPred,'VGG19Dataset3_YPred.txt','Delimiter','tab')
writematrix(scores,'VGG1Dataset3_scores.txt','Delimiter','tab')
%%
 