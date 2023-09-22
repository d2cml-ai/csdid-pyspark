
from pyspark.sql import  SparkSession, Row
from pyspark.sql.functions import \
	lit, col, when, expr, countDistinct,\
	monotonically_increasing_id, desc



def xform_to_strings(fmla : str = 'y ~ x + 1'):
	y, x = fmla.split('~')
	x_var = x.strip().split('+')
	x_var = [x.strip() for x in x_var]
	x_var = ['_intercept' if x == '1' else x for x in x_var]
	return x_var
def tlist_glist(data, tname, gname, _filter = False):
	tlist = data.select(tname).distinct().orderBy(col(tname))
	glist = data.select(gname).distinct().orderBy(col(gname))
	if _filter:
		value_expr = expr(f'max({gname})')
		value = data.select(value_expr.alias('value')).first()['value']
		glist = glist.filter(glist[gname] < value)
	return tlist, glist