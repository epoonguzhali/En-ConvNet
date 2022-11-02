

clc;
clear all;
close all;

%%
tic;
% 
% matlabroot = 'Dataset3_New'
% Datasetpath = fullfile(matlabroot)
% Data  = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')
% 
% 
% %Split the glaucoma images from Data in the ratio 70:30 and normal images also 
% [Data_GA7, Data_GA3] = splitEachLabel(Data,0.7,'Include','GA1')
% [Data_GE7, Data_GE3] = splitEachLabel(Data,0.7,'Include','GE1')
% [Data_N7, Data_N3] = splitEachLabel(Data,0.7,'Include','N1')
% % 
% 
% %Final Training set
% FinalTrain = imageDatastore(cat(1,Data_GA7.Files,Data_GE7.Files,Data_N7.Files))
% FinalTrain.Labels = cat(1,Data_GA7.Labels,Data_GE7.Labels,Data_N7.Labels)
% 
% % Final Testing set
% FinalTest = imageDatastore(cat(1,Data_GA3.Files,Data_GE3.Files,Data_N3.Files))
% FinalTest.Labels = cat(1,Data_GA3.Labels,Data_GE3.Labels,Data_N3.Labels)


matlabroot  = 'LAGTRAINING'
Datasetpath = fullfile(matlabroot)
FinalTrain  = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')

matlabroot  = 'DRISTESTING'
Datasetpath = fullfile(matlabroot)
FinalTest   = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')
% matlabroot = 'RFMiDOpticDiscALL\TrainingRot'
% Datasetpath = fullfile(matlabroot)
% FinalTrain  = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')
% 
% matlabroot = 'RFMiDOpticDiscALL\TestingRot'
% Datasetpath = fullfile(matlabroot)
% FinalTest  = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')
%  matlabroot = 'LAGDatabaseRot'
% Datasetpath = fullfile(matlabroot)
% Data  = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')
% 
% 
% %Split the images from Data in the ratio 80:20 
% [Data_G80, Data_G20] = splitEachLabel(Data,0.7,'Include','glaucoma')
% [Data_N80, Data_N20] = splitEachLabel(Data,0.7,'Include','normal')
% 
% % Final Training set
% FinalTrain = imageDatastore(cat(1,Data_G80.Files,Data_N80.Files))
% FinalTrain.Labels = cat(1,Data_G80.Labels,Data_N80.Labels)
% % 
% % % Final Testing set
% FinalTest = imageDatastore(cat(1,Data_G20.Files,Data_N20.Files));
% FinalTest.Labels = cat(1,Data_G20.Labels,Data_N20.Labels)

% Finding number of images in each category for training
labelCountTrain = countEachLabel(FinalTrain)

% Finding number of images in each category for testing
labelCountTest = countEachLabel(FinalTest)


% Resize the images to the input size of first layer
inputSize = [224,224,3];
%inputSize = [227,227,3];
%inputSize = [331,331,3];

augimdsTest = augmentedImageDatastore(inputSize(1:2),FinalTest);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),FinalTrain);

% Load the pre-trained network

 %net = mobilenetv2
 %net = googlenet
 net = squeezenet
 %net = shufflenet;
 %net = alexnet;
 %net = resnet18;
 %net = densenet201;
 %net = nasnetmobile;
 %net = nasnetlarge;
% net = efficientnetb0;
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

% classes = ["G1" "N1"];
% classWeights = [0.0008 0.001]

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
save('Res18LAGDRIS.mat')
 toc;
% 'ExecutionEnvironment','cpu', ...
  %%
  tic;
%load('Resnet18AGRotAdam1')
% 
% % Classification validation

%[YPredTrain,scoresTrain] = classify(MyNet,augimdsTrain);

%YValidationTrain = FinalTrain.Labels;
%accuracy = mean(YPredTrain == YValidationTrain)


% Plot confusion matrix
%figure, plotconfusion(YValidationTrain,YPredTrain)
 %xlswrite('Resner18LAGRotTrain',scoresTrain,1);
[YPred,scores] = classify(MyNet,augimdsTest);

%Accuracy calculation
YValidation = FinalTest.Labels;
accuracy = mean(YPred == YValidation)


% Plot confusion matrix
figure, plotconfusion(YValidation,YPred)
% 
toc;
%%
 xlswrite('ShuffleRimone3',scores,1);
 writematrix(YPred,'YPredShuffleRimone3Soft.txt','Delimiter','tab')
%%

%%%
% Grad cam visualizati0
Img1 = readimage(FinalTest,1);
%Img1 = imresize(Img1,[64,64]);
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

 % % % Extract the features from fc7 layer
   %layer = 'fire9-concat'
   %layer = 'global_average_pooling2d_1';
    %layer = 'pool5-drop_7x7_s1'
    %layer = 'pool5';
    layer = 'node_200';
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
%t = templateSVM('KernelFunction','rbf')
 
% mdl = fitcecoc(featuresTrainR,YTrain,'Learners',t);
 
 mdl = fitcecoc(featuresTrainR,YTrain);
 
 [YPred1, scores1] = predict(mdl,featuresTestR);
 
 accuracy = mean(YPred1 ==YTest);
 
 figure(1), plotconfusion(YTest,YPred1)
 toc;

 xlswrite('ShuffleRimone3',scores1,2);
writematrix(YPred1,'YPredShuffleResnet18SVM.txt','Delimiter','tab')
 %%
 % KNN CLassifier
 tic;
 
 md2 =  fitcknn(featuresTrainR,YTrain);
 
 [YPred2, scores2] = predict(md2,featuresTestR);
 
 accuracy = mean(YPred ==YTest);
 
 figure(1), plotconfusion(YTest,YPred2)
 
 toc;
 xlswrite('ShuffleRimone3',scores2,3);
 writematrix(YPred2,'YPredShuffleRimone3KNN.txt','Delimiter','tab')
 %%
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
 %%
  %%
 WaveletTrain = [];
 WaveletTest = [];
 
 [m,n] = size(featuresTrainR);
 
 [m1,n1] = size(featuresTestR);
 
 for i = 1:m
     Xtrain = featuresTrainR;
       
     % ID wavelet
     [CA1, CD1] = dwt(Xtrain(i,:),'haar');
     WaveletTrain(i,:) = CA1;
     
     % Fast walsh hadamard transform
    %FastTrain(i,:) = fwht(Xtrain(i,:));   
     
     % DCT
     %dcTrain(i,:) = dct(Xtrain(i,:));
     
 end
     
 for i = 1:m1

     Xtest = featuresTestR;
       % 1D DWT 
    [CA2, CD2] = dwt(Xtest(i,:),'haar');
     WaveletTest(i,:) = CA2;
     
      % Fast walsh hadamard transform
    %FastTest(i,:) = fwht(Xtest(i,:));   
     
     % DCT
     %dcTest(i,:) = dct(Xtest(i,:));
     
 
 end
%  
% %  % Classification using SVM classifier
% %  
 YTrain = FinalTrain.Labels;
 YTest = FinalTest.Labels;
 
 % Classifiaction using SVM classifier
 
 md3 = fitcecoc(WaveletTrain,YTrain);
 
 [YPred3, scores3] = predict(md3,WaveletTest);
 
 accuracy = mean(YPred3 ==YTest);
 
 figure(2), plotconfusion(YTest,YPred3)

xlswrite('ShuffleRimone3',scores3,4);
 writematrix(YPred3,'YPredSHuffleRimone3SVMHaartxt','Delimiter','tab')
%%

  xlswrite('ShuffleLAGRot',scores2,4);
  %%
 
  % Classifiaction using KNN classifier
 
 md4 = fitcknn(WaveletTrain,YTrain);
 
 [YPred4, scores4] = predict(md4,WaveletTest);
 
 accuracy = mean(YPred4 ==YTest);
 
 figure(3), plotconfusion(YTest,YPred4)

  xlswrite('ShuffleRimone3',scores4,5);
 writematrix(YPred4,'YPredShuffleRimone3KNNHaartxt','Delimiter','tab')
 %xlswrite('ShuffleLAGRot',scores2,4);
 %%
