clc;
clear all;
close all;


tic;

matlabroot = 'D:\poonguzhali\pre-trianed'
Datasetpath = fullfile(matlabroot,'DrishtiTrainingRotated')
FinalTrain  = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')
% 
matlabroot = 'D:\poonguzhali\pre-trianed'
Datasetpath = fullfile(matlabroot,'DrishtiTestingRotated')
FinalTest  = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')


% Finding number of images in each category for training
labelCountTrain = countEachLabel(FinalTrain)

% Finding number of images in each category for testing
labelCountTest = countEachLabel(FinalTest)

% Resize the images to the input size of first layer
inputSize = [299,299,3];

augimdsTest = augmentedImageDatastore(inputSize(1:2),FinalTest);

augimdsTrain = augmentedImageDatastore(inputSize(1:2),FinalTrain);

% Load the pre-trained network

net = inceptionv3;

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
save('Inceptionv3Data.mat')
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
%%

writematrix(YPred,'Inceptionv3Dataset3YPred.txt','Delimiter','tab')
writematrix(scores,'Inceptionv3Dataset3cores.txt','Delimiter','tab')
%%


% Grad cam visualizati0
Img1 = readimage(FinalTest,2);
%Img1 = imresize(Img1,[64,64]);
Img1 = imresize(Img1,[299,299]);
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
%%
% Extract the features from fc7 layer
layer = 'avg_pool'

featuresTrainR = activations(MyNet,augimdsTrain,layer,'OutputAs','rows');
featuresTestR = activations(MyNet,augimdsTest,layer,'OutputAs','rows');

 YTrain = FinalTrain.Labels;
 YTest = FinalTest.Labels;
 
 
 % Classification using softmax
 net = trainSoftmaxLayer(featuresTrainR,YTrain);
 
 % Classifiaction using SVM classifier
%Create a template for SVM classfier and use Gaussian kernel funcion
  %
  tic;
 
 mdl = fitcecoc(featuresTrainR,YTrain);
 
 [YPred1,scores1aa] = predict(mdl,featuresTestR);
 
 accuracy = mean(YPred1 ==YTest);
 
 figure, plotconfusion(YTest,YPred1)
 
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
 
% Naive Bayes CLassifier
 tic;
 
 md3 =  fitcnb(featuresTrainR,YTrain);
 
 [YPred3,scores3] = predict(md3,featuresTestR);
 
 accuracy = mean(YPred3 ==YTest);
 
 figure, plotconfusion(YTest,YPred3)
 
 toc;
 %%
 WaveletTrain = [];
 WaveletTest = []; 
 % %  
 [m,n] = size(featuresTrainR); 
 [m1,n1] = size(featuresTestR);
 
 for i = 1:m
     Xtrain = featuresTrainR;
       
     % ID wavelet
     [CA1, CD1] = dwt(Xtrain(i,:),'haar');
     WaveletTrain(i,:) = CA1;
  
 end
     
 for i = 1:m1

     Xtest = featuresTestR;
       % 1D DWT 
    [CA2, CD2] = dwt(Xtest(i,:),'haar');
     WaveletTest(i,:) = CA2;
   
 end
 
 % Classification using SVM classifier
 
 YTrain = FinalTrain.Labels;
 YTest = FinalTest.Labels;
 
 % Classifiaction using SVM classifier
 
 mdlw= fitcecoc(WaveletTrain,YTrain);
 
 [YPred1w,scores1w] = predict(mdlw,WaveletTest);
 
 accuracy = mean(YPred1w ==YTest);
   %xlswrite('Resnet101LAGResults',scores1w,5)
 figure, plotconfusion(YTest,YPred1w)
 
  % Classifiaction using KNN classifier
 
 md2w = fitcknn(WaveletTrain,YTrain);
 
 [YPred2w,scores2w] = predict(md2w,WaveletTest);
 
 accuracy = mean(YPred2w ==YTest);
  %xlswrite('Resnet101LAGResults',scores2w,6)
 
 figure, plotconfusion(YTest,YPred2w)
 
  % Classifiaction using NB classifier
 
 md3w = fitcknn(WaveletTrain,YTrain);
 
 [YPred3w,scores3w] = predict(md3w,WaveletTest);
 
 accuracy = mean(YPred3w ==YTest);
  %xlswrite('Resnet101LAGResults',scores2w,6)
 
 figure, plotconfusion(YTest,YPred3w)
 
  Yc = [YPred1,YPred2,YPred3,YPred1w, YPred2w, YPred3w];
 writematrix(Yc, 'XceptionRimone2Predw','Delimiter','tab');
 
 Ycs = [scores];
 writematrix(Ycs, 'XceptionRimone2score','Delimiter','tab'  );
 
 Ywm = [YPred];
 writematrix(Ywm, 'XceptionRimone2YPred','Delimiter','tab');
 
 Yswm = [scores1aa,scores2,scores3,scores1w, scores2w, scores3w];
  writematrix(Yswm, 'XceptionRimone2scoresw','Delimiter','tab'  );
 Yswm1 = [scores1m,scores2m,scores1wm, scores2wm];
 Yswm2 = [scores1m9,scores2m9,scores1m8,scores2m8];
 Yswm3  =[scores1wm8,scores2wm8,scores1m7,scores2m7];

 writematrix(Yswm, 'Inceptionresultsscores1','Delimiter','tab'  );
 writematrix(Yswm1, 'Inceptionresultsscores2','Delimiter','tab'  );
 writematrix(Yswm2, 'Inceptionresultsscores3','Delimiter','tab'  );
 writematrix(Yswm3, 'Inceptionresultsscores4','Delimiter','tab'  );
 
 YPredALL = [YPred, YPred1, YPred2, YPred1w, YPred2w];
writematrix(YPredALL, 'YPredALLResnet101.xls')
