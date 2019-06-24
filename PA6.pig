G = LOAD '$G' USING PigStorage(',') AS ( x:Long, y:Long );
I = GROUP G BY x;
O = FOREACH I  GENERATE ($0), COUNT ($1);
N = FOREACH O GENERATE ($1);
J = GROUP N BY ($0);
L = FOREACH J  GENERATE ($0), COUNT ($1);
STORE L INTO '$O' USING PigStorage (' ');   