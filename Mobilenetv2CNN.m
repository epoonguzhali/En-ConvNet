

clc;
clear all;
close all;


matlabroot  = 'LAGTRAINING'
Datasetpath = fullfile(matlabroot)
FinalTrain  = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')

matlabroot  = 'LAGTESTING'
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

% Load the pre-trained network

 net = mobilenetv2
 
analyzeNetwork(net)

%net.Layers(1)

% Replace the Final Layers

if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 

[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer] 

numClasses = 2;

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

% Replace the classification Layer with new
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

% Analyze the network for new Layers
 analyzeNetwork(lgraph)

% Specify the training Options for Network
options = trainingOptions('adam', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',10, ...     
    'InitialLearnRate',1e-4,'Shuffle','every-epoch');

MyNet = trainNetwork(augimdsTrain,lgraph,options);
save('MOBILELAG.mat')
 toc;
%% Program for Testing
  tic;
[YPredTrain,scoresTrain] = classify(MyNet,augimdsTrain);

% Plot confusion matrix
[YPred,scores] = classify(MyNet,augimdsTest);

%Accuracy calculation
YValidation = FinalTest.Labels;
accuracy = mean(YPred == YValidation)

 xlswrite('MobileLAG',scores,1);
 writematrix(YPred,'YPredSMobile.txt','Delimiter','tab')
%%

%%%
% Grad cam visualizati0
Img1 = readimage(FinalTest,1);
Img1 = imresize(Img1,[227,227]);
figure(1),imshow(Img1)


label_Img1 = classify(MyNet,Img1);

% Visulaization using Grad-CAM
scoreMap_Img1 = gradCAM(MyNet,Img1,label_Img1);
%tic;

figure(2)
imshow(Img1)
hold on
imagesc(scoreMap_Img1,'AlphaData',0.5)
colormap jet
%%
%  

