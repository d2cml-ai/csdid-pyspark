
from pyspark.sql import  SparkSession, Row, functions as F
from pyspark.sql.functions import \
	lit, col, when, expr, countDistinct,\
	monotonically_increasing_id, desc

from pyspark.sql.window import Window
import numpy as np


def xform_to_strings(fmla : str = 'y ~ x + 1'):
	if fmla is None:
		return ['_intercept']
	y, x = fmla.split('~')
	x_var = x.strip().split('+')
	x_var = [x.strip() for x in x_var]
	x_var = ['_intercept' if x == '1' else x for x in x_var]
	return x_var

def x_covariates(data, cov_names):
	if len(cov_names) == 1:
		if '_intercept' in cov_names:
			n = data.count()
			return np.ones((n, 1))
	
	col_rows = data.select(cov_names).collect()
	covariates = [
		[
			getattr(row, col) 
			for col in cov_names
		] 
		for row in col_rows
	]
	return np.array(covariates)


def tlist_glist(data, tname, gname, _filter = False):
	tlist = data.select(tname).distinct().orderBy(col(tname))
	glist = data.select(gname).distinct().orderBy(col(gname))
	if _filter:
		value_expr = expr(f'max({gname})')
		value = data.select(value_expr.alias('value')).first()['value']
		glist = glist.filter(glist[gname] < value)
	return tlist, glist




def panel2cs2(data, yname, idname, tname):
	if data.select(tname).distinct().count() != 2:
		raise 'panel2cs2 only for 2 perios of panel data'
	
	data = data.orderBy([idname, tname])
	window_spec = Window.partitionBy(idname).orderBy(tname)
	data = data.withColumn(
		'y1', F.lag(col(yname).over(window_spec))
	).withColumn(
		'y0', col(yname)
	).withColumn(
		'dy', col('y1') - col('y0')
	).na.drop()

	return data
