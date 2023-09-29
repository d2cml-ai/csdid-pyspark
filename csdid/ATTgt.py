from .utils import *
from .mboot import *
from .utils_aggte import *
from .plots import *


from pyspark.sql import  SparkSession, Row
from pyspark.sql.functions import \
	lit, col, when, expr, countDistinct,\
	monotonically_increasing_id, desc, mean

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
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
    summary_aggte = {
      'group': None,
      'calendar': None,
      'simple': None
    }

    self.summary_aggte = summary_aggte




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
      x_var = xform_to_strings(fmla=xfmla)
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


    tlist = tlist.collect()
    glist = glist.collect()
    tlist = [x[tname] for x in tlist]
    glist = [x[gname] for x in glist]
    self.tlist, self.glist, self.n, self.nG, self.nT, self.new_data = \
      tlist, glist, n, nG, nT, data

    self.true_rep_cross_section = true_rep_cross_section


  def fit(self, est_method = 'dr', base_period='varying', bstrap=False):
    self.base_period = base_period
    self.bstrap = bstrap

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
    tlist = self.tlist
    glist = self.glist

    n = self.n
    nT = self.nT
    nG = self.nG




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
          

    l_attest = len(att_est)
    crit_val, se, v = np.zeros(l_attest), np.zeros(l_attest), np.zeros(l_attest)

    DIDparams = {}
    DIDparams['bstrap'] = bstrap
    DIDparams['alp'] = self.alp
    DIDparams['panel'] = self.panel
    DIDparams['data'] = self.new_data
    DIDparams['clustervar'] = self.clustervar
    DIDparams['biters'] = self.biters
    DIDparams['tname'] = self.tname
    DIDparams['true_repeated_cross_section'] = self.true_rep_cross_section
    DIDparams['tlist'] = tlist
    DIDparams['tname'] = tname
    DIDparams['idname'] = idname
    DIDparams['cband'] = self.cband
    DIDparams['control_group'] = self.control_group
    DIDparams['anticipation'] = self.anticipation
    DIDparams['est_method'] = est_method
    

    if bstrap:
      inf_func = np.array(inf_func)
      ref_se = mboot(
        inf_func.T, DIDparams
        )
      crit_val, se = ref_se['crit_val'], ref_se['se']


    cband_lower = att_est - crit_val * se
    cband_upper = att_est + crit_val * se
    sig = (cband_upper < 0) | (cband_lower > 0)

    sig[np.isnan(sig)] = False
    sig_text = np.where(sig, "*", "")

    output = {
      'group': group,
      'time': year,
      'att': att_est,
      'post': post_array,
      'se': se,
      'c': crit_val,
      'upper': cband_lower,
      'lower': cband_upper,
      'sig': sig_text 
    }

    self.output = output
    self.inf_func = inf_func
    self.DIDparams = DIDparams


  def sum_gt(self, n=4):
    output = pd.DataFrame(self.output)
    name_attgt_df = ['Group', 'Time', 'ATT(g, t)', 'Post', "Std. Error", "[95% Pointwise", 'Conf. Band]', '']
    output.columns = name_attgt_df
    output = output.round(n)
    if not self.bstrap:
      output = output[name_attgt_df[:4]]
    self.summary2_gt = output
    return self

  def compute_aggte(self, typec         = "group",
                      balance_e     = None,
                      min_e         = float('-inf'),
                      max_e         = float('inf'),
                      na_rm         = False,
                      bstrap        = None,
                      biters        = None,
                      cband         = None,
                      alp           = None,
                      clustervars   = None,
                      print_ = True
                      ):
    out_result = self.output
    group = out_result['group']
    t = out_result['time']
    att = out_result['att']
    tlist = self.tlist
    glist = self.glist
    data = self.new_data
    inffunc = np.array(self.inf_func).T
    n = self.n
    gname = self.gname
    tname = self.tname
    idname = self.idname
    panel = self.panel

    group, t, att, tlist, glist = \
      map(np.array, [group, t, att, tlist, glist])

    if clustervars is None:
        clustervars = self.clustervar
    if bstrap is None:
        bstrap = self.bstrap
    if biters is None:
        biters = self.biters
    if alp is None:
        alp = self.alp
    if cband is None:
        cband = self.cband

    if typec not in ["simple", "dynamic", "group", "calendar"]:
        raise "`typec` must be one of ['simple', 'dynamic', 'group', 'calendar']"



    
    if na_rm:
        notna = ~np.isnan(att)
        group = group[notna]
        t = t[notna]
        att = att[notna]
        inffunc = inffunc[:, notna]
        glist = np.sort(np.unique(group))
    
        if typec == "group":
            gnotna = []
            for g in glist:
                indices = np.where((group == g) & (g <= t))
                is_not_na = np.any(~np.isnan(att[indices]))
                gnotna.append(is_not_na)
            
            gnotna = np.array(gnotna)
            glist = glist[gnotna]
            not_all_na = np.isin(group, glist)
            group = group[not_all_na]
            t = t[not_all_na]
            att = att[not_all_na]
            inffunc = inffunc[:, not_all_na]
            glist = np.sort(np.unique(group))

    if (not na_rm) and np.any(np.isnan(att)):
        raise "Missing values at att_gt found. If you want to remove these, set `na_rm = True`."
    if panel:
      dta = data.filter(col(tname) == tlist[0])
      columns_r = ['_w1', gname]
    else:
      dta = data.groupBy(idname).agg(
        mean(col('_w1')),
        mean(col(gname))
      )
      columns_r = dta.columns[1:]

    var_cols = x_covariates(dta, columns_r)
    _w1, _gvals = var_cols[:, :2].T

    originalt = t
    originalgroup = group.copy()
    originalglist = glist.copy()
    originaltlist = tlist.copy()
    # In case g's are not part of tlist


    list_array = list(originaltlist) + list(originalglist)
    originalgtlist = np.sort(np.unique(list_array))
    uniquet = list(range(len(originalgtlist) ))

    def t2orig(t):
        return originalgtlist[uniquet.index(t) if t in uniquet else -1]
    
    # Function to switch between "original" t values and new t values
    def orig2t(orig):
        new_t = [uniquet[i] for i in range(len(originalgtlist)) if originalgtlist[i] == orig]
        out = new_t[0] if new_t else None
        return out
    
    # print(originalgtlist, uniquet)
    t     = [orig2t(orig) for orig in originalt]
    group = [orig2t(orig) for orig in originalgroup]
    glist = [orig2t(orig) for orig in originalglist]
    tlist = np.asarray(list(set(t)))
    maxT  = max(t)
        
    # Set the weights
    # return data.columns
    weights_ind = _w1
    
    # We can work in overall probabilities because conditioning will cancel out
    # since it shows up in numerator and denominator
    pg = np.array([np.mean(weights_ind * (_gvals == g)) for g in originalglist])
    pgg = pg.copy()

    pg = [pg[glist.index(g)] for g in group]  
    
    # Which group time average treatment effects are post-treatment
    keepers = [i for i in range(len(group)) if group[i] <= t[i] <= (group[i] + max_e)] ### added second condition to allow for limit on longest period included in att
    
    # n x 1 vector of group variable
    G = [orig2t(g) for g in _gvals]
    # print(G)
    dp = self.DIDparams

    if typec == "simple":
      # Simple ATT
      # Averages all post-treatment ATT(g,t) with weights given by group size
      pg = np.array(pg)
      simple_att = np.sum(att[keepers] * pg[keepers]) / np.sum(pg[keepers])
      if np.isnan(simple_att):
          simple_att = None
  
      # Get the part of the influence function coming from estimated weights
      simple_wif = wif(keepers, pg, weights_ind, G, group)
  
      # Get the overall influence function
      simple_if = get_agg_inf_func(att = att , 
                                    inffunc = inffunc , 
                                    whichones = keepers ,
                                    weights_agg = np.array(pg)[keepers]/np.sum(np.array(pg)[keepers]) , 
                                    wif = simple_wif )[:, None]
  
      # Get standard errors from the overall influence function
      simple_se = get_se(simple_if, dp)
      
      if simple_se is not None:
          if simple_se <= np.sqrt(np.finfo(float).eps) * 10:
              simple_se = None

      simple_aggte = AGGTEobj(overall_att=simple_att, 
                                  overall_se=simple_se, 
                                  typec=typec,
                                  inf_function={'simple_att': simple_if}, 
                                  DIDparams=dp, print_=print_)

      self.summary_aggte['simple'] = simple_aggte
      # print(AGGTEobj_print)



    # =============================================================================
#  GRoup
# =============================================================================

    if typec == "group":
        group = np.array(group)
        t = np.array(t) 
        pg = np.array(pg) 
        selective_att_g = [np.mean(att[( group== g) & (t >= g) & (t <= (group + max_e))]) for g in glist]
        selective_att_g = np.asarray(selective_att_g)
        selective_att_g[np.isnan(selective_att_g)] = None
    
        selective_se_inner = [None] * len(glist)
        for i, g in enumerate(glist):
            whichg = np.where(np.logical_and.reduce((group == g, g <= t, t <= (group + max_e))))[0]
            weightsg =  pg[whichg] / np.sum(pg[whichg])
            inf_func_g = get_agg_inf_func(att = att , 
                                            inffunc = inffunc , 
                                            whichones = whichg ,
                                            weights_agg = weightsg , 
                                            wif = None)[:, None]
            se_g = get_se(inf_func_g, dp)
            selective_se_inner[i] = {'inf_func': inf_func_g, 'se': se_g}
            
        # recover standard errors separately by group   
        selective_se_g = np.asarray([item['se'] for item in selective_se_inner]).T
        
        selective_se_g[selective_se_g <= np.sqrt(np.finfo(float).eps) * 10] = None
        
        selective_inf_func_g = np.column_stack([elem["inf_func"] for elem in selective_se_inner])
  
        # use multiplier bootstrap (across groups) to get critical value
        # for constructing uniform confidence bands   
        selective_crit_val = norm.ppf(1 - alp/2)
        
        if dp['cband']:
            if not dp['bstrap']:
                print("Used bootstrap procedure to compute simultaneous confidence band")
        
            selective_crit_val = mboot(selective_inf_func_g, dp)['crit_val']
        
            if np.isnan(selective_crit_val) or np.isinf(selective_crit_val):
                print("Simultaneous critical value is NA. This probably happened because we cannot compute t-statistic (std errors are NA). We then report pointwise conf. intervals.")
                selective_crit_val = norm.ppf(1 - alp/2)
                dp['cband'] = False
        
            if selective_crit_val < norm.ppf(1 - alp/2):
                print("Simultaneous conf. band is somehow smaller than pointwise one using normal approximation. Since this is unusual, we are reporting pointwise confidence intervals")
                selective_crit_val = norm.ppf(1 - alp/2)
                dp['cband'] = False
        
            if selective_crit_val >= 7:
                print("Simultaneous critical value is arguably 'too large' to be reliable. This usually happens when the number of observations per group is small and/or there is not much variation in outcomes.")
  
        # get overall att under selective treatment timing
        # (here use pgg instead of pg because we can just look at each group)            
        selective_att = np.sum(selective_att_g * pgg) / np.sum(pgg)
        
        # account for having to estimate pgg in the influence function    
        selective_wif = wif(keepers = np.arange(1, len(glist)+1)-1, 
                            pg  = pgg, 
                            weights_ind = weights_ind, 
                            G = G, 
                            group = group)
        
        # get overall influence function   
        selective_inf_func = get_agg_inf_func(att = selective_att_g, 
                                              inffunc = selective_inf_func_g,
                                              whichones = np.arange(1, len(glist)+1)-1, 
                                              weights_agg = pgg/np.sum(pgg),
                                              wif = selective_wif)[:, None]    
        
        # get overall standard error        
        selective_se = get_se(selective_inf_func, dp)
        if not np.isnan(selective_se):
            if selective_se <= np.sqrt(np.finfo(float).eps) * 10:
                selective_se = None
    
        group_aggte = AGGTEobj(overall_att = selective_att, 
                            overall_se = selective_se, 
                            typec = typec,
                            egt = originalglist,
                            att_egt = selective_att_g,
                            se_egt = selective_se_g,
                            crit_val_egt = selective_crit_val,
                            inf_function = {'selective_inf_func_g': selective_inf_func_g, 
                                            'selective_inf_func': selective_inf_func},
                            DIDparams = dp, print_=print_)

        self.summary_aggte['group'] = group_aggte

        # =============================================================================
#  Calendar
# =============================================================================

 # np.array(group)
    if typec == "calendar":
        minG = min(group)
        calendar_tlist = tlist[tlist >= minG]
        pg = np.array(pg)
        calendar_att_t = []
        group = np.array(group)
        t = np.array(t)
        for t1 in calendar_tlist:
            whicht = np.where((t == t1) & (group <= t))[0]
            attt = att[whicht]
            pgt = pg[whicht] / np.sum(pg[whicht])
            calendar_att_t.append(np.sum(pgt * attt))
            
        # get standard errors and influence functions
        # for each time specific att
        calendar_se_inner = []
        for t1 in calendar_tlist:
            which_t = np.where((t == t1) & (group <= t))[0]
            pgt = pg[which_t] / np.sum(pg[which_t])
            wif_t = wif(keepers=which_t, 
                        pg=pg, 
                        weights_ind=weights_ind, 
                        G=G, 
                        group=group)
            inf_func_t = get_agg_inf_func(att=att, 
                                            inffunc=inffunc, 
                                            whichones=which_t, 
                                            weights_agg=pgt, 
                                            wif=wif_t)[:, None]
            se_t = get_se(inf_func_t, dp)
            calendar_se_inner.append({"inf_func": inf_func_t, "se": se_t})
    
    
    
        # recover standard errors separately by time
        calendar_se_t = np.array([se["se"] for se in calendar_se_inner]).T
        calendar_se_t[calendar_se_t <= np.sqrt(np.finfo(float).eps) * 10] = np.nan
        
        # recover influence function separately by time
        calendar_inf_func_t = np.column_stack([se["inf_func"] for se in calendar_se_inner])
    
        # use multiplier boostrap (across groups) to get critical value
        # for constructing uniform confidence bands
        calendar_crit_val = norm.ppf(1 - alp/2)
        
        if dp['cband']:
            if not dp['bstrap']:
                warnings.warn('Used bootstrap procedure to compute simultaneous confidence band')
        
            # mboot function is not provided, please define it separately
            calendar_crit_val = mboot(calendar_inf_func_t, dp)['crit_val']
        
            if np.isnan(calendar_crit_val) or np.isinf(calendar_crit_val):
                warnings.warn('Simultaneous critical value is NA. This probably happened because we cannot compute t-statistic (std errors are NA). We then report pointwise conf. intervals.')
                calendar_crit_val = norm.ppf(1 - alp/2)
                dp['cband'] = False
        
            if calendar_crit_val < norm.ppf(1 - alp/2):
                warnings.warn('Simultaneous conf. band is somehow smaller than pointwise one using normal approximation. Since this is unusual, we are reporting pointwise confidence intervals.')
                calendar_crit_val = norm.ppf(1 - alp/2)
                dp['cband'] = False
        
            if calendar_crit_val >= 7:
                warnings.warn("Simultaneous critical value is arguably 'too large' to be reliable. This usually happens when the number of observations per group is small and/or there is not much variation in outcomes.")
    
    
        # get overall att under calendar time effects
        # this is just average over all time periods
        calendar_att = np.mean(calendar_att_t)
        
        # get overall influence function
        calendar_inf_func = get_agg_inf_func(att=calendar_att_t,
                                             inffunc=calendar_inf_func_t,
                                             whichones=range(len(calendar_tlist)),
                                             weights_agg=np.repeat(1/len(calendar_tlist), len(calendar_tlist)),
                                             wif=None)[:, None]
        calendar_inf_func = np.array(calendar_inf_func)
        
        # get overall standard error
        calendar_se = get_se(calendar_inf_func, dp)
        if not np.isnan(calendar_se):
            if calendar_se <= np.sqrt(np.finfo(float).eps) * 10:
                calendar_se = np.nan
        
        calendar_aggte = AGGTEobj(overall_att=calendar_att,
                                overall_se=calendar_se,
                                typec=typec,
                                egt=list(map(t2orig, calendar_tlist)),
                                att_egt=calendar_att_t,
                                se_egt=calendar_se_t,
                                crit_val_egt=calendar_crit_val,
                                inf_function={"calendar_inf_func_t": calendar_inf_func_t,
                                              "calendar_inf_func": calendar_inf_func},
  
                                DIDparams=dp)

        self.summary_aggte['calendar'] = calendar_aggte

  
  def plot_attgt(self, ylim=None, 
                xlab=None, 
                ylab=None, 
                title="Group",
                xgap=1, 
                ncol=1, 
                legend=True, 
                group=None, 
                ref_line=0,
                theming=True, 
                grtitle="Group"
                ):

    did_object = self.output

    grp = did_object['group']
    t_i = did_object['time']

    G = len(np.unique(grp))
    Y = len(np.unique(t_i))
    g = np.unique(grp)[np.argsort(np.unique(grp))].astype(int)
    y = np.unique(t_i)

    results = pd.DataFrame({'year': np.tile(y, G)})
    results['group'] = np.repeat(g, Y)
    results['grtitle'] = grtitle + ' ' + results['group'].astype(str)
    results['att'] = did_object['att']
    results['att_se'] = did_object['se']
    results['post'] = np.where(results['year'] >= grp, 1, 0)
    results['year'] = results['year']
    results['c'] = did_object['c']

    self.results_plot_df_attgt = results

    if group is None:
      group = g
      if any(group not in g for group in group):
        raise "Some of the specified groups do not exist in the data. Reporting all available groups."


    legend_1 = False    # for multiple subplots, legend outside 
    fig, axes = plt.subplots(nrows=len(group), ncols=1, figsize=(10, 5))  # Adjust the figsize as needed
    handles = []
    labels = []
    for i, group_cat in enumerate(group):
        group_data = results.loc[results['group'] == group_cat]
        title = group_data['grtitle'].unique()[0]
        ax = axes[i]
        ax = gplot(group_data, ax, ylim, xlab, ylab, title, xgap, legend_1, ref_line, theming)
    plt.tight_layout()
    if legend is True:
        handles_ax, labels_ax = ax.get_legend_handles_labels()
        handles.extend(handles_ax)
        labels.extend(labels_ax)
        fig.legend(handles, labels, loc='lower center', fontsize='small', bbox_to_anchor=(0.545, -0.075), ncol=2)
    
    plt.show()
    return fig 
  def plot_aggte(self, 
          typec = 'group',
          ylim=None, 
                    xlab=None, 
                    ylab=None, 
                    title="", 
                    xgap=1, 
                    legend=True, 
                    ref_line=0, 
                    theming=True,
                    **kwargs):

    did_object = self.summary_aggte['group']

    post_treat = 1 * (np.asarray(did_object["egt"]).astype(int) >= 0)
    
    results = {
        "year": list(map(int, did_object["egt"])),
        "att": did_object["att_egt"],
        "att_se": did_object["se_egt"][0],
        "post": post_treat
    }
    
    results = pd.DataFrame(results)
    self.results_plot_df_aggte = results
    
    if did_object['crit_val_egt'] is None:
        results['c'] = abs(norm.ppf(0.025))
    else:
        results['c'] = did_object['crit_val_egt']

    if title == "":
        title = "Average Effect by Group" if\
          did_object["type"] == "group" else\
            "Average Effect by Length of Exposure"


    if did_object["type"] == "group":
        fig, ax = plt.subplots(figsize=(10, 5))
        p = splot(results, ax, ylim, xlab, ylab, title, legend, ref_line, theming)
        plt.tight_layout()
        plt.show()

    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        p = gplot(results, ax, ylim, xlab, ylab, title, xgap, legend, ref_line, theming)
        plt.tight_layout()
        plt.show() 
        
    return p