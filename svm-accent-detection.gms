SET
         TrainingIndex(*)
         FeatureIndex(*);

PARAMETER
        Kargs(*,*)
        TrainingData(*,*);

$ONTEXT
$call gdxxrw.exe accent-mfcc-data-1.xlsx par=TrainingData rng=train!A1:O181 set=TrainingIndex rng=trainingIndex!A2:A181 rdim=1 set=FeatureIndex rng=featureIndex!A2:A13 rdim=1 par=RbfKernelArguments rng=RbfKernelArguments!A1:FY181
$OFFTEXT

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

SCALAR
    C "Penalizzazione delle violazioni";

C = 10.0;

VARIABLE
         lambda(i) "Viariabli duali"
         ;
lambda.LO(i) = 0;
lambda.UP(i) = C;

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

SCALAR Margin;

Margin = 1.0 / SQRT(z.L);

DISPLAY lambda.L, Margin;


PARAMETER
    b,
    w(n),
    margin
    ;

w(n) = sum(i, y(i) * lambda.L(i) * x(i, n));

LOOP (i$(lambda.L(i) > 0),
    b = 1 / y(i) - sum(n, w(n) * x(i, n));
);

margin = 1 / sqrt(sum(n , sqr(w(n))));


DISPLAY w, b, margin;

PARAMETER TrainingSignal(i);

TrainingSignal(i) = y(i) * SIGN( SUM(j, lambda.L(j) * y(j) * K(i,j)) + b );


DISPLAY TrainingSignal;