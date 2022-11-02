%%

clc;
clear all;
close all;

tic;

 matlabroot = 'D:\Poonguzhali\pre-trianed\LAGDatabase'
Datasetpath = fullfile(matlabroot)
Data  = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')

% 
%Split the glaucoma images from Data in the ratio 70:30 and normal images also 
[Data_G80, Data_G20] = splitEachLabel(Data,0.7,'Include','glaucoma')
[Data_N80, Data_N20] = splitEachLabel(Data,0.7,'Include','normal')

FinalTrain = imageDatastore(cat(1,Data_G80.Files,Data_N80.Files))
FinalTrain.Labels = cat(1,Data_G80.Labels,Data_N80.Labels);

% Final Testing set
FinalTest = imageDatastore(cat(1,Data_G20.Files,Data_N20.Files));
FinalTest.Labels = cat(1,Data_G20.Labels,Data_N20.Labels)

% Finding number of images in each category for training
labelCountTrain = countEachLabel(FinalTrain)

% Finding number of images in each category for testing
labelCountTest = countEachLabel(FinalTest)


% Resize the images to the input size of first layer
inputSize = [227,227,3];

augimdsTest = augmentedImageDatastore(inputSize(1:2),FinalTest);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),FinalTrain);

      
% % Load the pre-trained model

    net=squeezenet;
    lgraph = layerGraph(net);
     analyzeNetwork(lgraph)
    
    % Number of categories
    numClasses = numel(categories(FinalTrain.Labels));
    
    % New Learnable Layer
    newLearnableLayer = convolution2dLayer(1,2,'Name','newConv10','Padding','same','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    
    % Replacing the last layers with new layersnet
    lgraph = replaceLayer(lgraph,'conv10',newLearnableLayer);
    newsoftmaxLayer = softmaxLayer('Name','new_softmax');
    lgraph = replaceLayer(lgraph,'prob',newsoftmaxLayer);
    newClassLayer = classificationLayer('Name','new_classoutput');
    lgraph = replaceLayer(lgraph,'ClassificationLayer_predictions',newClassLayer);
    
    lgraph = layerGraph(net);
     analyzeNetwork(lgraph)
     
     
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
    
% Retrain the network
options = trainingOptions('adam', ...
    'MiniBatchSize',32, ...
    'MaxEpochs',20, ...   
    'InitialLearnRate',1e-4,'Shuffle','every-epoch');

MyNet = trainNetwork(augimdsTrain,lgraph,options);
save('SqueezenetLAGAdamEp20.mat')
 toc;
 
 %% Program for Testing
  tic;

% 
% % Classification validation
[YPred,scores] = classify(MyNet,augimdsTest);

%Accuracy calculation
YValidation = FinalTest.Labels;
accuracy = mean(YPred == YValidation)


% Plot confusion matrix
figure, plotconfusion(YValidation,YPred)

writematrix(scores,'SquuzeResultsscores')
writematrix(YPred,'SqueezeResultsPred')
% 
toc;
%  

 % % % Extract the features from fc7 layer
layer = 'conv10';
featuresTrainR = activations(MyNet,augimdsTrain,layer,'OutputAs','rows');
featuresTestR = activations(MyNet,augimdsTest,layer,'OutputAs','rows');
% 
% % featuresTrainC = activations(alex,augimdsTrain,layer,'OutputAs','columns');
% % featuresTestC = activations(alex,augimdsTest,layer,'OutputAs','columns');
% % 
% % 
 YTrain = FinalTrain.Labels;
 YTest = FinalTest.Labels;
 
 % Classifiaction using SVM classifier
%Create a template for SVM classfier and use Gaussian kernel funcion
  %
  tic;
t = templateSVM('KernelFunction','rbf')
 
 mdl = fitcecoc(featuresTrainR,YTrain,'Learners',t);
 
 %mdl = fitcecoc(featuresTrainR,YTrain);
 
 YPred = predict(mdl,featuresTestR);
 
 accuracy = mean(YPred ==YTest);
 
 figure(1), plotconfusion(YTest,YPred)
 toc;
 
 % KNN CLassifier
 tic;
 
 mdl =  fitcknn(featuresTrainR,YTrain);
 
 YPred = predict(mdl,featuresTestR);
 
 accuracy = mean(YPred ==YTest);
 
 figure(1), plotconfusion(YTest,YPred)
 
 toc;
 
% Naive Bayes CLassifier
 tic;
 
 mdl =  fitcnb(featuresTrainR,YTrain);
 
 YPred = predict(mdl,featuresTestR);
 
 accuracy = mean(YPred ==YTest);
 
 figure(1), plotconfusion(YTest,YPred)
 
 toc;
 
 % Decision tree
 tic;
 
 mdl =  fitctree(featuresTrainR,YTrain);
 
 YPred = predict(mdl,featuresTestR);
 
 accuracy = mean(YPred ==YTest);
 
 figure(1), plotconfusion(YTest,YPred)
 
 toc;
 
