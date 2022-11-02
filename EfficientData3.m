clc;
clear all;
close all;

 matlabroot = 'LAGDatabase'
 Datasetpath = fullfile(matlabroot)
 Data  = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')
 
%Split the images from Data in the ratio 80:20 
[Data_G80, Data_G20] = splitEachLabel(Data,0.7,'Include','glaucoma')
[Data_N80, Data_N20] = splitEachLabel(Data,0.7,'Include','normal')
% 
% % Final Training set
 FinalTrain = imageDatastore(cat(1,Data_G80.Files,Data_N80.Files))
 FinalTrain.Labels = cat(1,Data_G80.Labels,Data_N80.Labels)
% % 
% % % Final Testing set
 FinalTest = imageDatastore(cat(1,Data_G20.Files,Data_N20.Files));
 FinalTest.Labels = cat(1,Data_G20.Labels,Data_N20.Labels)


% Finding number of images in each category for training
labelCountTrain = countEachLabel(FinalTrain)

% Finding number of images in each category for testing
labelCountTest = countEachLabel(FinalTest)



% Resize the images to the input size of first layer
inputSize = [224,224,3];

augimdsTest = augmentedImageDatastore(inputSize(1:2),FinalTest);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),FinalTrain);

% Load the pre-trained network

net = efficientnetb0;
% 
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
save('EfficientNet.mat')
 toc;
 
  %% Program for Testing
  tic;

% % Classification validation
%[YPred,scores] = classify(MyNet,augimdsTest);
[YPred,scores] = classify(MyNet,augimdsTest);
[YPredTrain,scoresTrain] = classify(MyNet,augimdsTrain);
xlswrite('Resnet18Rim3TrainScores', scoresTrain,1);

%Accuracy calculation
YValidation = FinalTest.Labels;
accuracy = mean(YPred == YValidation)


% Plot confusion matrix
figure, plotconfusion(YValidation,YPred)
%%
writematrix(YPredTrain,'EfficientDataset3RmsPRop_YPred.txt','Delimiter','tab')
writematrix(scoresTrain,'EfficientDataset3RmsProp_scores.txt','Delimiter','tab')

%%


% Grad cam visualizati0
Img1 = readimage(FinalTest,30);
%Img1 = imresize(Img1,[64,64]);
Img1 = imresize(Img1,[224,224]);
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


writematrix(YPred,'YPredRimoneDLEp5.txt','Delimiter','tab')
writematrix(scores,'scoresRimoneDLEp5.txt','Delimiter','tab')
% 
toc;
%  

% Extract the features from fc7 layer
layer = 'efficientnet-b0|model|head|global_average_pooling2d|GlobAvgPool'

featuresTrainR = activations(MyNet,augimdsTrain,layer,'OutputAs','rows');
featuresTestR = activations(MyNet,augimdsTest,layer,'OutputAs','rows');

 YTrain = FinalTrain.Labels;
 YTest = FinalTest.Labels;
 
 % Classifiaction using SVM classifier
%Create a template for SVM classfier and use Gaussian kernel funcion
  %
  tic;
 
 mdl = fitcecoc(featuresTrainR,YTrain);
 
 [YPred1,scores1] = predict(mdl,featuresTestR);
 
 accuracy = mean(YPred1 ==YTest);
 
 figure, plotconfusion(YTest,YPred1)
 writematrix(YPred1,'YPredRimoneDLEp5SVMSVM.txt','Delimiter','tab')
writematrix(scores1,'scoresRimoneDLEp5.txt','Delimiter','tab')
 
 %t = templateSVM('KernelFunction','rbf') 
%mdl = fitcecoc(featuresTrainR,YTrain,'Learners',t);
 toc;
 
 % KNN CLassifier
 tic;
 
 md2 =  fitcknn(featuresTrainR,YTrain);
 
 [YPred2,scores2] = predict(md2,featuresTestR);
 
 accuracy = mean(YPred2 ==YTest);
 
 figure, plotconfusion(YTest,YPred2)
 
 toc;
 writematrix(YPred2,'YPredRimoneDLEp5KNN.txt','Delimiter','tab')
writematrix(scores2,'scoresRImoneDLEp5KNN.txt','Delimiter','tab')
% Naive Bayes CLassifier
 tic;
 
 md3 =  fitcnb(featuresTrainR,YTrain);
 
 [YPred3,scores3] = predict(md3,featuresTestR);
 
 accuracy = mean(YPred3 ==YTest);
 
 figure, plotconfusion(YTest,YPred3)
 
 toc;
 writematrix(YPred3,'YPredRimoneDLEp5NB.txt','Delimiter','tab')
writematrix(scores3,'scoresRimoneDLEp5NB.txt','Delimiter','tab')
 
 % Decision tree
 tic;
 
 mdl =  fitctree(featuresTrainR,YTrain);
 
 YPred = predict(mdl,featuresTestR);
 
 accuracy = mean(YPred ==YTest);
 
 figure(1), plotconfusion(YTest,YPred)
 
 toc;
 
