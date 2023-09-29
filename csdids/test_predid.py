from ATTgt import ATTgt



import findspark
import warnings
findspark.init()

from pyspark.sql import  SparkSession, Row
from pyspark.sql.functions import \
	lit, col, when, expr, countDistinct,\
	monotonically_increasing_id, desc

spark = SparkSession.builder.appName('test_preprocess')\
    .config('spark.master', 'local[4]')\
    .config('spark.executor.memory', '1g')\
    .config("spark.sql.shuffle.partitions", 1)\
    .config('spark.driver.memory','1g')\
    .getOrCreate()

import os
name = r'D:\Workflow\work\csdid_r\R\5g10t.csv'
# os.path.exists(name)
data = spark.read.csv(name, header=True, inferSchema=True)

yname, gname, idname, tname = 'Y', 'G', 'id', 'period'


# a = ATTgt(yname=yname, gname=gname, idname=idname, tname=tname, data=data)

yname, gname, idname, tname = 'Y', 'G', 'id', 'period'
control_group = ['nevertreated', 'notyettreated']
anticipation = 0
panel = True
allow_unbalanced_panel = True
weights_name = None

clustervar = None
xfmla = None

# a.new_data.show(4)

n_pre = data.count()
control_group = control_group[0]
columns = [idname, tname, yname, gname]

if clustervar is not None:
	columns += [clustervar]
if weights_name is not None:
	columns += [weights_name]
	data = data.withColumn('_w', data[weights_name])
else:
	columns += ['_w']
	data = data.withColumn('_w', lit(1))

data = data.withColumn('_intercept', lit(1))


print('all')
spark.stop()