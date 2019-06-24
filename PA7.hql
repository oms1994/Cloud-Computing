drop table Graph;
drop table inter;
create table Graph (
  X double,
  Y double)
  row format delimited fields terminated by ',' stored as textfile;
  load data local inpath '${hiveconf:G}' overwrite into table Graph;


CREATE TABLE inter AS
SELECT 
X as ver,	
COUNT(*) as count
FROM
  Graph	
GROUP BY X	;



SELECT 
count,
COUNT(*) as j
FROM
  inter
GROUP BY count;

