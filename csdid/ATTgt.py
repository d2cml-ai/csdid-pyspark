from .utils import *


from pyspark.sql import  SparkSession, Row
from pyspark.sql.functions import \
	lit, col, when, expr, countDistinct,\
	monotonically_increasing_id, desc

class ATTgt:
  def __init__(
    self, yname, tname, idname, gname, data, control_group = ['nevertreated', 'notyettreated'], 
    xfmla: str = None, panel = True, allow_unbalanced_panel = True, 
    clustervar = None, weights_name = None, anticipation = 0, 
    cband = False, biters = 1000, alp = 0.05
  ):
    self.yname = yname
    self.tname = tname
    self.idname = idname 
    self.gname = gname
    self.data = data
    self.control_group = control_group
    self.xfmla = xfmla
    self.panel = panel
    self.allow_unbalanced_panel = allow_unbalanced_panel
    self.clustervar = clustervar
    self.weights_name = weights_name
    self.anticipation = anticipation
    self.cband = cband
    self.biters = biters
    self.alp = alp



    # tlist, glist, n, nG, nT, new_data \
    #   = self._preprocess_did(
    #   yname, tname, idname, data, control_group, anticipation, xfmla,
    #   panel, allow_unbalanced_panel, cband, clustervar, weights_name
    #   )



  def _preprocess_did(self):
    
    yname = self.yname
    tname = self.tname
    idname = self.idname 
    gname = self.gname
    data = self.data
    control_group = self.control_group
    xfmla = self.xfmla
    panel = self.panel
    anticipation = self.anticipation
    allow_unbalanced_panel = self.allow_unbalanced_panel
    clustervar = self.clustervar
    weights_name = self.weights_name

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

    # xfmla = 'y ~ X + 1'
    if xfmla is None:
      x_var = ['_intercept']
      x_cov = data[x_var]
      n_cov = 1
    else:
      x_var = form_to_strings(fmla=xfmla)
      n_cov = len(x_var)

    columns += x_var

    data = data[columns].na.drop('all')
    ndiff = n_pre - data.count()

    if ndiff != 0: 
      print(f'Dropped, {ndiff}, rows from original data due to missing data')


    tlist, glist = tlist_glist(data, tname, gname, False)

    #   asif_nev_treated = data[gname] > np.max(tlist)
    #   asif_nev_treated.fillna(False, inplace=True)
    #   data.loc[asif_nev_treated, gname] = 0

    max_tlist = tlist.orderBy(desc(tname)).first()[tname]
    data = data.withColumn(
        "asif_nev_treated",
        when(
            col(gname) > max_tlist, True
        ).otherwise(False))\
        .fillna({'asif_nev_treated': False})\
        .withColumn(gname, when(col("asif_nev_treated") == True, 0).otherwise(col(gname)))

    n_glist0 = glist.filter(glist[gname] == 0).count()
    if n_glist0 == 0:
      if control_group == 'nevertreated':
        raise 'There is no avaible never-treated group'
      else:
        value_expr = expr(f'max({gname}) - {anticipation}')
        value = data.select(value_expr.alias('value')).first()['value']
        data =  data.filter(data[tname] < value)
        tlist, glist = tlist_glist(data, tname, gname, True)

    glist = glist.filter(glist[gname] > 0)
    # first_period
    fp = tlist.first()[tname]
    glist = glist.filter(glist[gname] > fp + anticipation)
    
    data = data.\
      withColumn(
        "treated_fp", (col(gname) <= fp) & ~(col(gname) == 0)
      ).\
      fillna({'treated_fp': False})
    nfirst_period = \
      data.filter(col('treated_fp') == True).count() if panel\
      else\
        data.filter(col('treated_fp') == True).\
          select(idname).distinct().count()
    # 93 - 102
    if nfirst_period > 0:
      warning_message = f"Dropped {nfirst_period} units that were already treated in the first period."
      print(warning_message)
      glist_in = glist.collect()
      glist_in = [x[gname] for x in glist_in]
      glist_in = glist_in + [0]
      data = data.filter(col(gname).isin(glist_in))
      tlist, glist = tlist_glist(data, tname=tname, gname=gname)
      glist = glist.filter(col(gname) > 0)
      fp = tlist.first()[tname]
      glist = glist.filter(col(gname) > fp + anticipation)

    true_rep_cross_section = False
    if not panel:
      true_rep_cross_section = True

    if panel: 
      if allow_unbalanced_panel: 
        panel = False
        true_rep_cross_section = False
      else:
        keep = data.na.drop('all')
        n_id = data.select(idname).distinct().count()
        n_keep = keep.select(idname).distinct().count()
        n_old_data = data.count()
        # data = makeBalancedPanel(data, idname=idname, tname=tname)
        n_new = data.count()

        if n_new == 0:
          raise "All observations dropped to convert data to balanced panel. Consider setting `panel=False` and/or revisit 'idname'."
        if n_new < n_old_data:
          warnings.warn(f"Dropped {n_old_data-n_new} observations while converting to balanced panel.")

        tn = tlist.first()[tname]
        n = data.filter(col(tname) == tn).count()
    if not panel:
      if true_rep_cross_section:
          data = data.withColumn('rowid', monotonically_increasing_id())
          idname = 'rowid'
      else:
          data = data.withColumn('rowid', col(idname))
      n = data.select(idname).distinct().count()

    data = data.orderBy([idname, tname]).\
      withColumn('_w1', col('_w'))

    if glist.count() == 0:
      raise f"No valid groups. The variable in '{gname}' should be expressed as the time a unit is first treated (0 if never-treated)."
    if tlist.count() == 2:
      cband = False


    gsize = data.groupBy(col(gname)).count()
    tlist_len = tlist.count()

    gsize = gsize.withColumn('count', gsize['count'] / tlist_len)

    req_size = n_cov + 5

    gsize = gsize.filter(col('count') < req_size)
    if gsize.count() > 0:
      gpaste_array = gsize.select(gname).collect()
      gpaste = [x[gname] for x in gpaste_array]
      gpaste = ",".join(map(str, gpaste))
      if len(gpaste) > 0:
        print(f"Be aware that there are some small groups in your dataset.\n  Check groups: {gpaste}.")
      gval_ = gsize.select(gname).collect()
      gval_ = [x[gname] for x in gval_]
      if 0 in gval_ and control_group == 'nevertreated':
        raise "Never-treated group is too small, try setting control_group='notyettreated'."

    nT, nG = tlist.count(), glist.count()

    # return tlist, glist, n, nG, nT, data


    self.tlist, self.glist, self.n, self.nG, self.nT, self.new_data = \
      tlist, glist, n, nG, nT, data

