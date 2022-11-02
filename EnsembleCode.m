% Ensemble results for Drishti 

matlabroot  = 'DrishtiTraining'
Datasetpath = fullfile(matlabroot)
FinalTrain  = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')

matlabroot  = 'DrishtiTesting'
Datasetpath = fullfile(matlabroot)
FinalTest   = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')

% Finding number of images in each category for training
labelCountTrain = countEachLabel(FinalTrain)

% Finding number of images in each category for testing
labelCountTest = countEachLabel(FinalTest)


YTest = FinalTest.Labels;
YTrain = FinalTrain.Labels;

ScoresTrain = xlsread('Resner18LAGRotTrain.xls');

ScoresEnsem = xlsread('EnsembleLAGRotTest.xlsx');

%SVM1 classifier
tic;
 md1 = fitcecoc(ScoresTrain,YTrain);
 
 [YPredEns, scoresEns]= predict( md1,ScoresEnsem);
 
 accuracy = mean(YPredEns ==YTest);
 
 figure, plotconfusion(YTest,YPredEns)
toc;

xlswrite('EnsembleLAgRotOutput',scoresEns,1);
 writematrix(YPredEns,'EnsembleLAGRotOutputYPred.txt','Delimiter','tab')

%SVM2 classifier
tic;
 %md11 = fitcsvm(ScoresTrain,YTrain);
 md12 = fitcsvm(ScoresTrain,YTrain,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');
 [YPredEns2, scoresEns2]= predict( md12,ScoresEnsem);
 
 accuracy = mean(YPredEns2 ==YTest);
 
 figure, plotconfusion(YTest,YPredEns2)
toc;
xlswrite('EnsembleLAgRotOutput',scoresEns2,2);
 writematrix(YPredEns2,'EnsembleLAGRotOutputYPredSVM2.txt','Delimiter','tab')

%KNN classifier                                              
tic;
 md3 = fitcknn(ScoresTrain,YTrain);
 
 md31 = fitcknn(ScoresTrain,YTrain,'NumNeighbors',5,'Standardize',1)
 [YPredEns3, scoresEns3]= predict( md31,ScoresEnsem);
 
 accuracy = mean(YPredEns3 ==YTest);                           
 
 figure, plotconfusion(YTest,YPredEns3)
toc;
xlswrite('EnsembleLAgRotOutput',scoresEns3,3);
 writematrix(YPredEns3,'EnsembleLAGRotOutputOutputYPredKNN.txt','Delimiter','tab')


 %%
 % RIM-ONE3

 matlabroot = 'RIM-ONE3Rot'
Datasetpath = fullfile(matlabroot)
Data  = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')

[Data_G80, Data_G20] = splitEachLabel(Data,0.7,'Include','G1')
[Data_N80, Data_N20] = splitEachLabel(Data,0.7,'Include','N1')

FinalTrain = imageDatastore(cat(1,Data_G80.Files,Data_N80.Files))
FinalTrain.Labels = cat(1,Data_G80.Labels,Data_N80.Labels)

% Final Testing set
FinalTest = imageDatastore(cat(1,Data_G20.Files,Data_N20.Files));
FinalTest.Labels = cat(1,Data_G20.Labels,Data_N20.Labels)

ScoresTrain = xlsread('Rimone3Train.xlsx');

ScoresEnsem = xlsread('Rimone3Testnew.xlsx');

YTest = FinalTest.Labels;
YTrain = FinalTrain.Labels;

%SVM1 classifier
tic;
 md1 = fitcecoc(ScoresTrain,YTrain);
 
 [YPredEns, scoresEns]= predict( md1,ScoresEnsem);
 
 accuracy = mean(YPredEns ==YTest);
 
 figure, plotconfusion(YTest,YPredEns)
toc;
%%
xlswrite('EnsembleRIM3NewOutput',scoresEns,1);
 writematrix(YPredEns,'EnsembleRIM3OutputNewYPred.txt','Delimiter','tab')

%SVM2 classifier
tic;
 %md11 = fitcsvm(ScoresTrain,YTrain);
 md12 = fitcsvm(ScoresTrain,YTrain,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');
 [YPredEns2, scoresEns2]= predict( md12,ScoresEnsem);
 
 accuracy = mean(YPredEns2 ==YTest);
 
 figure, plotconfusion(YTest,YPredEns2)
toc;
xlswrite('EnsembleLAgRotOutput',scoresEns2,2);
 writematrix(YPredEns2,'EnsembleLAGRotOutputYPredSVM2.txt','Delimiter','tab')

%KNN classifier                                              
tic;
% md3 = fitcknn(ScoresTrain,YTrain);
 
 md31 = fitcknn(ScoresTrain,YTrain,'NumNeighbors',5,'Standardize',1)
 [YPredEns3, scoresEns3]= predict( md31,ScoresEnsem);
 
 accuracy = mean(YPredEns3 ==YTest);                           
 
 figure, plotconfusion(YTest,YPredEns3)
toc;
xlswrite('EnsembleLAgRotOutput',scoresEns3,3);
 writematrix(YPredEns3,'EnsembleLAGRotOutputOutputYPredKNN.txt','Delimiter','tab')

 %% Ensemble Resulsts for ACRIMA

 matlabroot  = 'AcrimaTrainingRotated'
Datasetpath = fullfile(matlabroot)
FinalTrain  = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')

matlabroot  = 'AcrimaTestingRotated'
Datasetpath = fullfile(matlabroot)
FinalTest   = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')

YTest = FinalTest.Labels;
YTrain = FinalTrain.Labels;


ScoresTrain = xlsread('AcrimaTrain.xls');

ScoresEnsem = xlsread('AcrimaTest.xlsx');


%SVM1 classifier
tic;
 md1 = fitcecoc(ScoresTrain,YTrain);
 
 [YPredEns, scoresEns]= predict( md1,ScoresEnsem);
 
 accuracy = mean(YPredEns ==YTest);
 
 figure, plotconfusion(YTest,YPredEns)
toc;

%SVM2 classifier
tic;
 %md11 = fitcsvm(ScoresTrain,YTrain);
 md12 = fitcsvm(ScoresTrain,YTrain,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');
 [YPredEns2, scoresEns2]= predict( md12,ScoresEnsem);
 
 accuracy = mean(YPredEns2 ==YTest);
 
 figure, plotconfusion(YTest,YPredEns2)
toc;
xlswrite('EnsembleLAgRotOutput',scoresEns2,2);
 writematrix(YPredEns2,'EnsembleLAGRotOutputYPredSVM2.txt','Delimiter','tab')

%KNN classifier                                              
tic;
% md3 = fitcknn(ScoresTrain,YTrain);
 
 md31 = fitcknn(ScoresTrain,YTrain,'NumNeighbors',5,'Standardize',1)
 [YPredEns3, scoresEns3]= predict( md31,ScoresEnsem);
 
 accuracy = mean(YPredEns3 ==YTest);                           
 
 figure, plotconfusion(YTest,YPredEns3)
toc;
xlswrite('EnsembleLAgRotOutput',scoresEns3,3);
 writematrix(YPredEns3,'EnsembleLAGRotOutputOutputYPredKNN.txt','Delimiter','tab')

 %% Ensemble results for Origa

  matlabroot  = 'OrigaTrainingRotated'
Datasetpath = fullfile(matlabroot)
FinalTrain  = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')

matlabroot  = 'OrigaTestingRotated'
Datasetpath = fullfile(matlabroot)
FinalTest   = imageDatastore(Datasetpath,'IncludeSubfolders',true,'LabelSource','foldernames')

YTest = FinalTest.Labels;
YTrain = FinalTrain.Labels;


ScoresTrain = xlsread('OrigaScoresTrain.xlsx');

ScoresEnsem = xlsread('OrigaTest2.xlsx');


%SVM1 classifier
tic;
 md1 = fitcecoc(ScoresTrain,YTrain);
 
 [YPredEns, scoresEns]= predict( md1,ScoresEnsem);
 
 accuracy = mean(YPredEns ==YTest);
 
 figure, plotconfusion(YTest,YPredEns)
toc;

%SVM2 classifier
tic;
 %md11 = fitcsvm(ScoresTrain,YTrain);
 md12 = fitcsvm(ScoresTrain,YTrain,'Standardize',true,'KernelFunction','RBF',...
    'KernelScale','auto');
 [YPredEns2, scoresEns2]= predict( md12,ScoresEnsem);
 
 accuracy = mean(YPredEns2 ==YTest);
 
 figure, plotconfusion(YTest,YPredEns2)
toc;
xlswrite('EnsembleLAgRotOutput',scoresEns2,2);
 writematrix(YPredEns2,'EnsembleLAGRotOutputYPredSVM2.txt','Delimiter','tab')

%KNN classifier                                              
tic;
% md3 = fitcknn(ScoresTrain,YTrain);
 
 md31 = fitcknn(ScoresTrain,YTrain,'NumNeighbors',5,'Standardize',1)
 [YPredEns3, scoresEns3]= predict( md31,ScoresEnsem);
 
 accuracy = mean(YPredEns3 ==YTest);                           
 
 figure, plotconfusion(YTest,YPredEns3)
toc;
xlswrite('EnsembleLAgRotOutput',scoresEns3,3);
 writematrix(YPredEns3,'EnsembleLAGRotOutputOutputYPredKNN.txt','Delimiter','tab')

