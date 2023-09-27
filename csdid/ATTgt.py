from .utils import *


from pyspark.sql import  SparkSession, Row
from pyspark.sql.functions import \
	lit, col, when, expr, countDistinct,\
	monotonically_increasing_id, desc

import numpy as np, pandas as pd
from tqdm import tqdm


from drdid import drdid, reg_did

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


    self._preprocess_did()




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
        self.panel = panel
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


  def fit(self, est_method = 'dr', base_period='varying'):
    self.base_period = base_period


    yname = self.yname
    tname = self.tname
    idname = self.idname
    gname = self.gname
    xfmla = self.xfmla
    data = self.new_data
    weights_name = self.weights_name
    panel = self.panel
    control_group= self.control_group
    anticipation = self.anticipation

    n = self.n
    nT = self.nT
    nG = self.nG

    tlist = self.tlist.collect()
    glist = self.glist.collect()

    tlist = [x[tname] for x in tlist]
    glist = [x[gname] for x in glist]
    tlist, glist = map(np.array, [tlist, glist])
    # print(tlist, glist)

    tlist_len, tfac = len(tlist), 0

    if base_period != 'universal':
      tlist_len = tlist_len - 1
      tfac = 1

    never_treated = control_group == 'nevertreated'
    if never_treated:
      data = data.withColumn('C', when(col(gname) == 0, 1).otherwise(0))
    
    data = data.withColumn('y_main', col(yname))


    inf_func = []

    att_est, group, year, post_array = [], [], [], []

    def add_att_data(att = 0, pst = 0, inf_f = []):
      inf_func.append(inf_f)
      att_est.append(att)
      group.append(g)
      year.append(tn)
      post_array.append(pst)

    covariates_var = xform_to_strings(xfmla)

    # for _, g in tqdm(enumerate(glist)):
    for g_i in tqdm(range(len(glist))):
      g = glist[g_i]
      data = data.withColumn(
        'G_m', when(
          col(gname) == g, 1
        ).otherwise(0)
      )
      for t_i in tqdm(range(tlist_len)):
        # print(g, t_i)
        pret = t_i
        tn = tlist[t_i + tfac]

        if base_period == 'universal' or g < tn:
          try:
            pret = np.where(tlist + anticipation < g)[0][-1]
          except:
            raise f"There are no pre-treatment periods for the group first treated at {g}\nUnits from this group are dropped"

        if base_period == 'universal':
          if pret == tn:
            add_att_data()
          
        if not never_treated:
          # base
          # n1 = data[gname] == 0
          # n2 = (data[gname] > (tlist[np.max([t_i, pret]) + tfac]) + anticipation)
          # n3 = np.where(data[gname] != glist[g], True, False)
          # row_eval = n1 | n2 & n3

          data = data.withColumn('n1', col(gname) == 0)
          value_n2 = (tlist[np.max([t_i, pret]) + tfac]) + anticipation
          data = data.withColumn(
            'n2', col(gname) > value_n2
          ).withColumn(
            'n3', col(gname) != g
          )
          data = data.withColumn(
            'row_eval', (col('n1') | col('n2') & col('n3'))
          ).withColumn('C', col('row_eval'))
        
        post_treat = 1 * (g <= tn)

        disdat = data.filter(
          (col(tname) == tn) | (col(tname) == tlist[pret])
        )

        if panel:
          disdat = panel2cs2(data, yname, idname, tname)
          n = disdat.count()
          disdat = disdat.withColumns(
            'dis_idx', (col('G_m') == 1) | (col('C') == 1)
          )
          disdat = disdat.filter(col('dis_idx'))
          n1 = disdat.count()


          # .rdd.flatMap(lambda x: x).collect()
  

          G = disdat.select('G_m').rdd.flatMap(lambda x: x).collect()
          C = disdat.select('C').rdd.flatMap(lambda x: x).collect()
          w = disdat.select('_w1').rdd.flatMap(lambda x: x).collect()
          y0 = disdat.select('y0').rdd.flatMap(lambda x: x).collect()
          y1 = disdat.select('y1').rdd.flatMap(lambda x: x).collect()
          dis_idx = disdat.select('dis_idx').rdd.flatMap(lambda x: x).collect()

          ypre = y0 if tn > pret else y1
          ypost = y0 if tn < pret else y1

          covariates = x_covariates(disdat, covariates_var)

          G, C, w, dis_idx = map(np.array, [G, C, w, dis_idx])
          ypost, ypre = map(np.array, [ypost, ypre])


          if callable(est_method):
            _f_est_att = est_method
          elif est_method == 'reg':
            _f_est_att = reg_did.reg_did_panel
          elif est_method =='dr':
            _f_est_att = drdid.drdid_panel


          att_gt, att_inf_func = _f_est_att(
            ypost, ypre, G, i_weights = w, covariates = covariates
          )

          inf_zeros = np.zeros(n)
          att_inf = n / n1 * att_inf_func
          inf_zeros[dis_idx] = att_inf

          add_att_data(att_gt, inf_f=inf_zeros)

        if not panel:
          disdat = disdat.withColumn(
              'GmC', (col('G_m') == 1) | (col('C') == 1)
            ).withColumn(
              'post', col(tname) == tlist[t_i + tfac]
            ).withColumn(
              'tPret', col(tname) == tlist[pret]
            ).filter(
              col('GmC') & col('post') | col('tPret')
            )

          ref_col = [
            'G_m', 'C', yname, 'post', '_w1', 'rowid'
          ]
          var_ = x_covariates(disdat, ref_col)

          G, C, Y, post, w, right_ids = var_[:, :len(ref_col)].T
          
          covariates = x_covariates(disdat, covariates_var)

          n1 = sum(G + C)

          skip_this_att_gt_list = np.array([
            np.sum(G * post) == 0,
            np.sum(G * (1 - post)) == 0,
            np.sum(C * post) == 0,
            np.sum(C * (1 - post)) == 0
          ])

          skip_this_att_gt = any(skip_this_att_gt_list)
          if skip_this_att_gt:
            add_att_data()
          
          _t = tlist[t_i]
          mssg_errors = np.array([
            f"No units in group {g} in time period {tn}",
            f"No units in group {g} in time period {_t}",
            f"No available control units for group {g} in time period {tn}",
            f"No available control units for group {g} in time period {_t}"
            ]
          )

          print_mssg_error = "\n".join(mssg_errors[skip_this_att_gt_list]), 
          
          if callable(est_method):
            _f_est_att = est_method
          elif est_method == "reg":
            _f_est_att = reg_did.reg_did_rc
          elif est_method == "dr":
            _f_est_att = drdid.drdid_rc


          att_gt, att_inf_func = _f_est_att(y=Y, post=post, D = G, i_weights=w, covariates=covariates)


          inf_func_df = pd.DataFrame(
            {
              "inf_func": att_inf_func,
              "right_ids": right_ids
            }
          )
          inf_zeros = np.zeros(n)
          aggte_infffuc = inf_func_df.groupby('right_ids').inf_func.sum()

          data_row_id = data.select('rowid').distinct().rdd.flatMap(lambda x : x).collect()
          data_row_id = np.array(data_row_id)

          dis_idx1 = np.isin(data_row_id, aggte_infffuc.index.to_numpy())

          inf_zeros[dis_idx1] = np.array(aggte_infffuc)

          add_att_data(att_gt, pst = post_treat, inf_f=inf_zeros)
          

    output = {
      'group': group,
      'time': year,
      'att': att_est,
      'post': post_array
    }

    self.output = output
    self.inf_func = inf_func


    def sum_gt(self, n=4):
      output = pd.DataFrame(self.output)
      name_attgt_df = ['Group', 'Time', 'ATT(g, t)', 'Post', "Std. Error", "[95% Pointwise", 'Conf. Band]', '']
      output.columns = name_attgt_df
      output = output.round(n)
      self.summary2_gt = output
      return self


