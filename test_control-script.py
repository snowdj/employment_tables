# to put into perspective how much more efficient it is to use Python, and functions in your code, the previous process to produce these tables used 15 SPSS scripts and 3 excel files with a total of 50 sheets. This makes the process considerably more: transparent, less prone to errors. It also allows products such as the interactive tool to be built which would have not previously been possible.

# as it stands, I have removed SIC as a breakdown option, to simplify anonymising the data

import pandas as pd
import numpy as np
import itertools
import pytest
import platform
import pickle




# this section creates the aggregated and anonymised data used for analysis. If you don't have access to the raw data, you can move straight on to the analysis section which uses saved aggregated data (CURRENTLY DUMMY DATA).

if platform.system() == 'Darwin':
    shared_drive = '/Volumes/'
elif platform.system() == 'Windows':
    shared_drive = 'g drive path here'

raw_data_dir = shared_drive + 'Data/EAU/Statistics/Economic Estimates/Employment - Helen/max-csv-data/'
raw_data_dir = '~/data/'

years = range(2011, 2017 + 1)

#raw_data = {y:pd.read_csv(raw_data_dir +  'raw_' + str(y) + "_df.csv") for y in years}
#
#pickle.dump(raw_data, open("raw_data.p", "wb"))
raw_data = pickle.load(open("raw_data.p", "rb"))

# finish cleaning data post R cleaning =========================================

regionlookupdata = pd.read_csv('region-lookup.csv')
regionlookdict = {row[0]: row[1] for index, row in regionlookupdata.iterrows()}

sic_mappings = pd.read_csv("sic_mappings.csv")
sic_mappings = sic_mappings[sic_mappings.sic != 62.011]
sic_mappings.sic = round(sic_mappings.sic * 100, 0)
sic_mappings.sic = sic_mappings.sic.astype(int)


# make time series tables ======================================================


# we start with df which has a single row for each person which contains their main and second jobs, so main and second sic, and main and second emptype (INECAC05 and SECJMBR) and a weighted count

# we need to sum together the main and second jobs. so we subset for mainemp, secondemp, mainsemp, secondsemp.

# for each subset, we add a sector column and create some new levels - each sector subet (including all dcms), totaluk (which is entire original subset), civil society, overlap. so we are left with a nice big data set with every combination of sic, sector, and cat (e.g. region). 

# however, we need to join the 4 subsets together so that the main and second jobs counts can be added up. 

# currently we create sector and cat base columns (agg) to outer join into (thereby adding sics) so that each subset is aligned and can be added. We 

# maybe it is best to not bother with the base columns and just outer join everything - and do the aggregating in the next stage - would need to make sure that sic, sector, region (not region, just cat, one of which is region!) do not have any missing values - for now include sic, sector and region in agg then try reducing after the for loop???



def expand_grid(data_dict):
   rows = itertools.product(*data_dict.values())
   return pd.DataFrame.from_records(rows, columns=data_dict.keys())


# debug to find loc warning
#@profile
def clean_data(year):
    #region = False
    
    # find weighting column name for given year    
    if year < 2016:
        weightedcountcol = 'PWTA14'
    if year == 2016:
        weightedcountcol = 'PWTA16'
    if year == 2017:
        weightedcountcol = 'PWTA17'

    df = raw_data[year]
    
    df['regionmain'] = df.GORWKR.map(regionlookdict)
    df['regionsecond'] = df.GORWK2R.map(regionlookdict)
#    df['qualification'] = df['qualification'].astype(str)
    df['ftpt'] = df['ftpt'].astype(str)
    df['nssec'] = df['nssec'].astype(str)
    
    catuniques = []
    for caty in demographics + ['region']:
        if caty == 'region':
            catuniques.append(np.unique(regionlookupdata.mapno))
        else:
            catuniques.append(np.unique(df[caty]))
    
    x = pd.Series(np.unique(sic_mappings.sector))
    y = pd.Series(["civil_society", "total_uk", "overlap"])
    x = x.append(y)
    
    aggdict = {}
    aggdict['sector'] = x
    for caty in demographics + ['region']:
        if caty == 'region':
            aggdict[caty] = np.unique(regionlookupdata.mapno)
        else:
            aggdict[caty] = np.unique(df[caty])

    #'sic': np.unique(sic_mappings.sic),
    agg = expand_grid(aggdict)

    for subset in ['mainemp', 'secondemp', 'mainselfemp', 'secondselfemp']:
        if subset == 'mainemp':
            sicvar = "INDC07M"
            emptype = "INECAC05"
            emptypeflag = 1
            regioncol = 'regionmain'
    
        if subset == 'secondemp':
            sicvar = "INDC07S"
            emptype = "SECJMBR"
            emptypeflag = 1
            regioncol = 'regionsecond'
    
        if subset == 'mainselfemp':
            sicvar = "INDC07M"
            emptype = "INECAC05"
            emptypeflag = 2
            regioncol = 'regionmain'
    
        if subset == 'secondselfemp':
            sicvar = "INDC07S"
            emptype = "SECJMBR"
            emptypeflag = 2
            regioncol = 'regionsecond'

        # create subset for each of 4 groups
        df['region'] = df[regioncol]
        df['region'] = df['region'].fillna('missing region')
        dftemp = df[[sicvar, emptype, weightedcountcol, 'cs_flag'] + demographics + ['region']].copy()
        dftemp = dftemp.loc[dftemp[emptype] == emptypeflag]
        # need separate sic column to allow merging - I think
        dftemp.rename(columns={sicvar : 'sic'}, inplace=True)

        # total uk includes missing sics, so take copy before removing missing sics
        dftemp_totaluk = dftemp.copy()
        
        # remove rows from subset with missing sic
        dftemp = dftemp.loc[np.isnan(dftemp.sic) == False]
        
        # add sector column and further subset to all sectors excluding all_dcms
        dftemp_sectors = pd.merge(dftemp, sic_mappings.loc[:,['sic', 'sector']], how = 'inner')
        dftemp_sectors = dftemp_sectors.loc[dftemp_sectors['sector'] != 'all_dcms']
        
        # subset civil society
        dftemp_cs = dftemp.loc[dftemp['cs_flag'] == 1].copy()
        dftemp_cs['sector'] = 'civil_society'
        dftemp_cs = dftemp_cs[dftemp_sectors.columns.values]
        
        # subset all_dcms (still need to add cs and remove overlap)
        dftemp_all_dcms = pd.merge(dftemp, sic_mappings.loc[:,['sic', 'sector']], how = 'inner')
        dftemp_all_dcms = dftemp_all_dcms.loc[dftemp_all_dcms['sector'] == 'all_dcms']
        
        # subset overlap between sectors
        dftemp_all_dcms_overlap = pd.merge(dftemp, sic_mappings.loc[:,['sic', 'sector']], how = 'inner')
        dftemp_all_dcms_overlap = dftemp_all_dcms_overlap.loc[dftemp_all_dcms_overlap['sector'] == 'all_dcms']
        dftemp_all_dcms_overlap = dftemp_all_dcms_overlap.loc[dftemp_all_dcms_overlap['cs_flag'] == 1]
        dftemp_all_dcms_overlap['sector'] = 'overlap'
        
        # subset uk total
        dftemp_totaluk['sector'] = 'total_uk'
        # reorder columns
        dftemp_totaluk = dftemp_totaluk[dftemp_sectors.columns.values]
        
        # append different subsets together
        dftemp = dftemp_totaluk.append(dftemp_sectors).append(dftemp_cs).append(dftemp_all_dcms).append(dftemp_all_dcms_overlap)
#        dftemp = pd.concat(['dftemp_totaluk', 'dftemp_sectors', 'dftemp_cs', 'dftemp_all_dcms', 'dftemp_all_dcms_overlap'])
        
        # groupby ignores NaN so giving region NaNs a value
        
        # this converts sic back to numeric
        dftemp = dftemp.infer_objects()
        
        # only total_uk sector has nan sics so groupby is dropping data - setting missing values to 'missing'
        dftemp['sic'] = dftemp['sic'].fillna(value=-1)
        dftemp.head()
        # create column with unique name (which is why pd.DataFrame() syntax is used) which sums the count by sector
        aggtemp = pd.DataFrame({subset : dftemp.groupby( ['sector', 'sic'] + demographics + ['region'])[weightedcountcol].sum()}).reset_index()
        
        # merge final stacked subset into empty dataset containing each sector and category level combo
        # should be able to just use aggtemp for first agg where subset=='mainemp', but gave error, need to have play around. checking that agg has all the correct sectors and cat levels should be a separate piece of code.
        agg = pd.merge(agg, aggtemp, how='outer')
     
    agg['year'] = year
    return agg


# ------------------------------------------------------------------------------
# now we sum together the emp columns and melt the data so that emptype is a column

def clean_data2(df):
    
    # fill in missing values to avoid problems with NaN
    agg = df.fillna(0)
    
    # sum main and second jobs counts together
    agg['emp'] = agg['mainemp'] + agg['secondemp']
    agg['selfemp'] = agg['mainselfemp'] + agg['secondselfemp']
    agg.drop(['mainemp', 'secondemp', 'mainselfemp', 'secondselfemp'], axis=1, inplace=True)
    
    # melt 'emp' and 'selfemp' into 'emptype'
    agg = agg[['sector', 'sic', 'year', 'emp', 'selfemp', 'region'] + demographics]
    melted = pd.melt(agg, id_vars=['sector', 'sic', 'year', 'region'] + demographics, var_name='emptype', value_name='count')  
    
    # need to aggregate before we can add civil society to all_dcms?
    
    # reduce down to desired aggregate
    aggfinal = melted.drop(['sic'], axis=1)
    aggfinal = aggfinal.groupby(['sector', 'emptype', 'year', 'region'] + demographics).sum()
    aggfinal = aggfinal.reset_index(['sector', 'emptype', 'year', 'region'] + demographics)
    
    # add civil society to all_dcms and remove overlap from all_dcms
    aggfinaloverlap = aggfinal.copy()
#    aggfinaloverlap = aggfinaloverlap.reset_index(drop=True)
    
    alldcmsindex = aggfinaloverlap[aggfinaloverlap['sector'] == 'all_dcms'].index
    csindex = aggfinaloverlap[aggfinaloverlap['sector'] == 'civil_society'].index
    overlapindex = aggfinaloverlap[aggfinaloverlap['sector'] == 'overlap'].index
    newalldcms = aggfinaloverlap.loc[alldcmsindex, ['count']].reset_index(drop=True) + aggfinaloverlap.loc[csindex, ['count']].reset_index(drop=True) - aggfinaloverlap.loc[overlapindex, ['count']].reset_index(drop=True)
    newalldcms2 = newalldcms['count']
    newalldcms3 = np.array(newalldcms2)
    aggfinaloverlap.loc[alldcmsindex, ['count']] = newalldcms3
    
    return aggfinaloverlap


# need clean_data2() to include year to be able to reconcile figures

# check figures match doing it this way then make clean_data2() more flexible to not strip out sic.


# qualification is only in 2016 and 2017 so exclude
demographics = ['sex', 'ethnicity', 'dcms_ageband', 'ftpt', 'nssec']
other_vars = ['sector', 'emptype', 'year', 'region']

cleaned_data = [clean_data(i) for i in years]
agg = pd.concat(cleaned_data, ignore_index=True)

#pickle.dump(agg, open("agg.p", "wb"))
#agg = pickle.load(open("agg.p", "rb"))

#mycat = demographics + ['region']
aggfinal = clean_data2(agg)
aggfinal.columns



# stack demographic cols
#aggfinalmelt = aggfinal.melt(
#    other_vars + ['count'],
#    demographics,
#    value_name='level', var_name='demographic'
#).groupby(other_vars + ['demographic', 'level'], as_index=False).sum()
#
#aggfinalmelt = aggfinalmelt.fillna(0)
#check = aggfinalmelt.loc[(aggfinalmelt.demographic == 'sex') & (aggfinalmelt.year == 2016)]





# anonymisation
#tb[tb < 6000] = 0
#check = aggfinal.copy()
#check.loc[pd.isnull(check.count)] = 0
#check['count'] = check['count'].astype(int)
#check = aggfinal.loc[aggfinal.count < 0]
#aggfinal.dtypes

# make tables

# anonymisation - to reduce the amount of anonymisation needed, the data has been structured to not allow comparison of multiple demographics - this is conistent with the current excel publication.

def make_table(index, columns, sub_col=None, sub_value=None, raw=False):
    
    # create flag for whether data needs to be subsetted to all_dcms
    if 'sector' not in index and 'sector' not in columns and 'sector' not in [sub_col]:
        all_dcms = True
    else:
        all_dcms = False
        
    if raw:
        agg_temp = aggfinal
        
    else:
        # user specified subset data
        if sub_col:
            all_cols = index + columns + [sub_col]
        else:
            all_cols = index + columns
        if all_dcms:
            all_cols = all_cols + ['sector']
            
        agg_temp = pd.read_csv('agg_csvs/' + '_'.join(sorted(all_cols)) + '.csv')

    
    # subset data if subset arguments provided
    if sub_col:
            agg_temp = agg_temp.loc[agg_temp[sub_col] == sub_value]
        
    # for non sector breakdowns, subset data to only inlcude 'all_dcms' sector
    if all_dcms:
        agg_temp = agg_temp.loc[agg_temp.sector == 'all_dcms']
    
#    idemo = [d for d in index if d in demographics]
#    cdemo = [d for d in columns if d in demographics]
#    index2 = index
#    columns2 = columns
#    if idemo and cdemo:
#        print('tooooo many demographics')
#    if idemo:
#        agg_temp = agg_temp.loc[agg_temp.demographic == idemo[0]]
#        index2 = ['level' if d==idemo[0] else d for d in index]
#        
#    if cdemo:
#        agg_temp = agg_temp.loc[agg_temp.demographic == cdemo[0]]
#        columns2 = ['level' if d==cdemo[0] else d for d in columns]

    
    # pd.crosstab() only accepts lists of series not subsetted dataframes
#    sindex = [agg_temp[col] for col in index2]
#    scolumns = [agg_temp[col] for col in columns2]
    
    # pd.crosstab() only accepts lists of series not subsetted dataframes
    sindex = [agg_temp[col] for col in index]
    scolumns = [agg_temp[col] for col in columns]    
    # create table
    tb = pd.crosstab(index=sindex, columns=scolumns, values=agg_temp['count'], aggfunc='sum')
    
    # reorder columns and index
    orderings = {
        'sector': ["civil_society", "creative", "culture", "digital", "gambling", "sport", "telecoms", "all_dcms", "total_uk"],
        'sex': ['Male', 'Female'],
        'region': ['North East', 'North West', 'Yorkshire and the Humber', 'East Midlands', 'West Midlands', 'East of England', 'London', 'South East', 'South West', 'Wales', 'Scotland', 'Northern Ireland'],
    }
    for i in [i for i in index if i in orderings]:
        if isinstance(tb.index, pd.core.index.MultiIndex):
            tb = tb.reindex(orderings[i], axis=0, level=i)
        else:
            tb = tb.reindex(orderings[i], axis=0)

    for i in [i for i in columns if i in orderings]:
        if isinstance(tb.columns, pd.core.index.MultiIndex):
            tb = tb.reindex(orderings[i], axis=1, level=i)
        else:
            tb = tb.reindex(orderings[i], axis=1)
    
    
    # round and convert to 000's
    tb = round(tb / 1000, 0).astype(int)
    
    return tb


make_table(['region'], ['emptype'], 'year', 2017)


#demographics
#other_vars
#demographics + other_vars
#combo_vars = ['sector', 'emptype', 'year', 'region', 'demographic', 'gap']
#combo_vars = ['emptype', 'year', 'sector']
#len(combo_vars) # 6. 7 is 5k+, 8 is 40k+

from itertools import chain, combinations
# find all powersets - from https://docs.python.org/3/library/itertools.html#itertools-recipes
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


make_table_combos = [list(x) for x in powerset(demographics + other_vars)]
len(make_table_combos)
combo = ['sector', 'ethnicity', 'year']
def reduce_and_save_data(combo):
    mydf = aggfinal.groupby(combo)['count'].sum().reset_index()
    mydf.loc[mydf['count'] < 6000, 'count'] = 0    
    mydf.to_csv('agg_csvs/' + '_'.join(sorted(combo)) + '.csv')


#for com in make_table_combos[1:]:
#    reduce_and_save_data(com)

aggfinal.columns
make_table(['sector', 'ethnicity'], ['year'])
make_table(['sector', 'ethnicity'], ['emptype', 'sex'], 'year', 2017)
make_table(['sector', 'ethnicity'], ['emptype', 'sex'], 'year', 2017, True)


#demo = [d for d in ['emptype', 'sex'] if d in demographics]
#columns = ['emptype', 'sex']
#columns = ['level' if d==demo[0] else d for d in columns]
#if demo:
#    print(5)
#demo[0]
#one = []
#two = []
#three = one + two
#if three:
#    print(5)

#tb_back = make_table(['sector'], ['sex'], 'year', 2016)
#tb = make_table(['sector'], ['emptype'], 'year', 2016)
#tb = make_table(['region', 'sector'], ['emptype'], 'year', 2016)


# read in data directly from 2016 excel publication downloaded from gov.uk
from openpyxl import load_workbook, Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import string

def read_xl_pub(wsname, startrow, finishrow, cols):
    ws = wb[wsname]
    col_nos = [string.ascii_lowercase.index(myletter.lower()) for myletter in cols]
    
    exceldata = ws.values
    exceldata = list(exceldata)
    newdata = []
    
    # copy data into list
    for row in range(startrow - 1, finishrow):
        listrow = [exceldata[row][i] for i in col_nos]
    
        # code anonymised as 0s
        listrow = [0 if x == '-' else x for x in listrow]
        
        # code NA as 999999s
        listrow = [999999 if x == 'N/A' else x for x in listrow]
        
        listrow = [-999999 if pd.isnull(x) else x for x in listrow]
    
        newdata.append(listrow)
    
    exceldataframe = pd.DataFrame(newdata)
    return exceldataframe


# add tourism
#data = data.append(tourism)


# testing
# https://www.gov.uk/government/statistics/dcms-sectors-economic-estimates-2017-employment-and-trade
# https://www.gov.uk/government/uploads/system/uploads/attachment_data/file/636391/DCMS_Sectors_Economic_Estimates_Employment_2016_tables.xlsx
wb = load_workbook('DCMS_Sectors_Economic_Estimates_Employment_2016_tables.xlsx')
py_tbs = dict(
    sex = make_table(['sector'], ['sex'], 'year', 2016)[0:7],
#    region = make_table(['region'], ['emptype'], 'year', 2016),
)
xl_tbs = dict(
    sex = read_xl_pub(wsname = "3.5 - Gender (000's)", startrow = 9, finishrow = 15, cols = ['l', 'n']),
    region = read_xl_pub(wsname = "3.3 - Region (000's)", startrow = 8, finishrow = 19, cols = ['b', 'd']),
)


# marks=pytest.mark.xfail
@pytest.mark.parametrize('test_input,expected', [
    pytest.param('sex', True, marks=pytest.mark.basic),
])
def test_datamatches(test_input, expected):
    assert (py_tbs[test_input].values == xl_tbs[test_input].values).all() == expected


# for region the total won't = the sum anyway, so don't need to do annonymisation
