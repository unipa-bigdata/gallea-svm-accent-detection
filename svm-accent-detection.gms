SET
         TrainingIndex(*)
         FeatureIndex(*);

PARAMETER
        Kargs(*,*)
        TrainingData(*,*);

$call gdxxrw.exe data/accent-mfcc-data-1.xlsx par=TrainingData rng=train!A1:O181 set=TrainingIndex rng=trainingIndex!A2:A181 rdim=1 set=FeatureIndex rng=featureIndex!A2:A13 rdim=1 par=RbfKernelArguments rng=RbfKernelArguments!A1:FY181

$GDXIN accent-mfcc-data-1.gdx
$LOAD TrainingData, TrainingIndex, FeatureIndex, Kargs=RbfKernelArguments
$GDXIN


DISPLAY Kargs, TrainingData, TrainingIndex, FeatureIndex

ALIAS(TrainingIndex,i,j);

ALIAS(FeatureIndex,n);

PARAMETER
         K(i,j)
         x(i,n)
         y(i);

x(i,n) = TrainingData(i,n);
y(i) = TrainingData(i,'language');




SCALAR
   sigma /10/;

K(i,j) = EXP(-0.5 * (Kargs(i,j)/SQR(sigma)));

$ONTEXT
K(i,j) = Kargs(i,j);
$OFFTEXT

POSITIVE VARIABLE
         lambda(i) "Variabili duali";

VARIABLE
         z "Funzione obiettivo";

EQUATION
         LagrangianEq
         DualCon;

LagrangianEq.. z=E=SUM(i, lambda(i)) - 0.5 * SUM((i,j),
                       lambda(i) * lambda(j) * y(i) * y(j) * K(i,j));

DualCon..      SUM(i, lambda(i) * y(i)) =E= 0.0;

MODEL DualSVM /ALL/;

SOLVE DualSVM MAXIMIZING z USING NLP;

DISPLAY
         lambda.L;


PARAMETERS
         numberOfData
         successValue
         b;

LOOP(i$(lambda.L(i) > 0),

         b = (1.0/y(i)) - SUM(j,lambda.L(j) * y(j) * K(i,j));
);

DISPLAY b;

PARAMETER TrainingSignal(i);

TrainingSignal(i) = y(i) * SIGN( SUM(j, lambda.L(j) * y(j) * K(i,j)) + b );

successValue = SUM(i, TrainingSignal(i));

numberOfData = CARD(i);

DISPLAY TrainingSignal;

DISPLAY successValue,numberOfData;