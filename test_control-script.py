# to put into perspective how much more efficient it is to use Python, and functions in your code, the previous process to produce these tables used 15 SPSS scripts and 3 excel files with a total of 50 sheets. This makes the process considerably more: transparent, less prone to errors. It also allows products such as the interactive tool to be built which would have not previously been possible.

# as it stands, I have removed SIC as a breakdown option, to simplify anonymising the data

import pandas as pd
import numpy as np
import itertools
import pytest
import platform

if platform.system() == 'Darwin':
    shared_drive = '/Volumes/'
elif platform.system() == 'Windows':
    shared_drive = 'g drive path here'

raw_data_dir = shared_drive + 'Data/EAU/Statistics/Economic Estimates/Employment - Helen/max-csv-data/'
raw_data_dir = '~/data/'

years = range(2011, 2016 + 1)

raw_data = {}
for year in years:
    raw_data[year] = pd.read_csv(raw_data_dir +  'raw_' + str(year) + "_df.csv")
    #print("~/data/cleaned_" + str(year) + "_df.csv")


# df2016 = pd.read_csv("~/data/cleaned_2016_df.csv")
#df2016 = allyears[current_year]

# finish cleaning data post R cleaning =========================================

regionlookupdata = pd.read_csv('region-lookup.csv')
regionlookdict = {}
for index, row in regionlookupdata.iterrows():
    regionlookdict.update({row[0] : row[1]})


def clean_raw_data(df):
        
    df['regionmain'] = df.GORWKR.map(regionlookdict)
    df['regionsecond'] = df.GORWK2R.map(regionlookdict)
    
    return df

cleaned_data = {k: clean_raw_data(v) for k, v in raw_data.items()}

#df = allyears[current_year]    
#df['qualification'] = df['qualification'].astype(str)
#df['ftpt'] = df['ftpt'].astype(str)
#df['nssec'] = df['nssec'].astype(str)

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

    df = cleaned_data[year]
    
    catuniques = []
    for caty in mycat:
        if caty == 'region':
            catuniques.append(np.unique(regionlookupdata.mapno))
        else:
            catuniques.append(np.unique(df[caty]))
    
    x = pd.Series(np.unique(sic_mappings.sector))
    y = pd.Series(["civil_society", "total_uk", "overlap"])
    x = x.append(y)
    
    aggdict = {}
    aggdict['sector'] = x
    for caty in mycat:
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
        dftemp = df[[sicvar, emptype, weightedcountcol, 'cs_flag'] + mycat].copy()
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
        dftemp = dftemp_totaluk.append(dftemp_sectors)
        dftemp = dftemp.append(dftemp_cs)
        dftemp = dftemp.append(dftemp_all_dcms)
        dftemp = dftemp.append(dftemp_all_dcms_overlap)
        
        # groupby ignores NaN so giving region NaNs a value
        
        # this converts sic back to numeric
        dftemp = dftemp.infer_objects()
        
        # only total_uk sector has nan sics so groupby is dropping data - setting missing values to 'missing'
        dftemp['sic'] = dftemp['sic'].fillna(value=-1)
        dftemp.head()
        # create column with unique name (which is why pd.DataFrame() syntax is used) which sums the count by sector
        aggtemp = pd.DataFrame({subset : dftemp.groupby( ['sector', 'sic'] + mycat)[weightedcountcol].sum()}).reset_index()
        
        # merge final stacked subset into empty dataset containing each sector and category level combo
        # should be able to just use aggtemp for first agg where subset=='mainemp', but gave error, need to have play around. checking that agg has all the correct sectors and cat levels should be a separate piece of code.
        agg = pd.merge(agg, aggtemp, how='outer')
     
    agg['year'] = year
    return agg


# ------------------------------------------------------------------------------
# now we sum together the emp columns and melt the data so that emptype is a column

def clean_data2(df):
    
    agg = df.copy()
    
    # fill in missing values to avoid problems with NaN
    agg = agg.fillna(0)
    
    # sum main and second jobs counts together
    agg['emp'] = agg['mainemp'] + agg['secondemp']
    agg['selfemp'] = agg['mainselfemp'] + agg['secondselfemp']
    agg.drop(['mainemp', 'secondemp', 'mainselfemp', 'secondselfemp'], axis=1, inplace=True)
    
    agg.head()
    melted = pd.melt(agg, id_vars=['sector', 'sic', 'year'] + mycat, var_name='emptype', value_name='count')
    melted.head()    
    
    # need to aggregate before we can add civil society to all_dcms?
    
    # reduce down to desired aggregate
    aggfinal = melted.drop(['sic'], axis=1)
    aggfinal = aggfinal.groupby(['sector', 'emptype', 'year'] + mycat).sum()
    aggfinal = aggfinal.reset_index(['sector', 'emptype', 'year'] + mycat)
    aggfinal.head()
    
    # add civil society to all_dcms and remove overlap from all_dcms
    aggfinaloverlap = aggfinal.copy()
    aggfinaloverlap = aggfinaloverlap.reset_index(drop=True)
    aggfinaloverlap.head()
    
    alldcmsindex = aggfinaloverlap[aggfinaloverlap['sector'] == 'all_dcms'].index
    csindex = aggfinaloverlap[aggfinaloverlap['sector'] == 'civil_society'].index
    overlapindex = aggfinaloverlap[aggfinaloverlap['sector'] == 'overlap'].index
    newalldcms = aggfinaloverlap.loc[alldcmsindex, ['count']].reset_index(drop=True) + aggfinaloverlap.loc[csindex, ['count']].reset_index(drop=True) - aggfinaloverlap.loc[overlapindex, ['count']].reset_index(drop=True)
    type(newalldcms)
    newalldcms2 = newalldcms['count']
    type(newalldcms2)
    newalldcms3 = np.array(newalldcms2)
    type(newalldcms3)
    aggfinaloverlap.loc[alldcmsindex, ['count']] = newalldcms3
    
    aggfinal = aggfinaloverlap.copy()
    
    return aggfinal


# need clean_data2() to include year to be able to reconcile figures


# check figures match doing it this way then make clean_data2() more flexible to not strip out sic.
table_levels = ['sex', 'region', 'sector']
mycat = ['sex', 'region']
mycat = ['sex']
mycat = ['region']
demographics = ['sex', 'ethnicity', 'dcms_ageband', 'qualification', 'ftpt', 'nssec']
other_vars = ['sector', ]

cleaned_data2 = [clean_data(i) for i in years]
agg = pd.concat(cleaned_data2, ignore_index=True)
agg.columns

aggfinal = clean_data2(agg)


# make tables

# anonymisation - to reduce the amount of anonymisation needed, the data has been structured to not allow comparison of multiple demographics - this is conistent with the current excel publication.

def make_table(index, columns, sub_col, sub_value):
    
    #subset data
    agg_temp = aggfinal.loc[aggfinal[sub_col] == sub_value]
    
    
    # pd.crosstab() only accepts lists of series not subsetted dataframes
    sindex = [agg_temp[col] for col in index]
    scolumns = [agg_temp[col] for col in columns]
    
    # create table
    tb = pd.crosstab(index=sindex, columns=scolumns, values=agg_temp['count'], aggfunc='sum')
    
    # reorder columns and index
    orderings = {
        'sector': ["civil_society", "creative", "culture", "digital", "gambling", "sport", "telecoms", "all_dcms", "total_uk"],
        'sex': ['Male', 'Female'],
        'region': ['North East', 'North West', 'Yorkshire and the Humber', 'East Midlands', 'West Midlands', 'East of England', 'London', 'South East', 'South West', 'Wales', 'Scotland', 'Northern Ireland', 'All regions'],
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
    
    # anonymise
    tb[tb < 6000] = 0
    
    # round and convert to 000's
    tb = round(tb / 1000, 0).astype(int)
    
    return tb


tb = make_table(['sector'], ['emptype', 'sex'], 'year', 2016)
tb = make_table(['sector'], ['emptype'], 'year', 2016)
tb = make_table(['region', 'sector'], ['emptype'], 'year', 2016)


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

wb = load_workbook('DCMS_Sectors_Economic_Estimates_Employment_2016_tables.xlsx')
xl_gender = read_xl_pub(
        wsname = "3.5 - Gender (000's)",
        startrow = 9,
        finishrow = 17,
        cols = ['l', 'n'])


(tb.values == xl_gender.values).all()





if mycat == 'qualification':
    dfcopy = dfcopy[dfcopy.qualification != 'dont know']
    dfcopy = dfcopy[dfcopy.qualification != 'nan']
    
# add tourism
#data = data.append(tourism)
        
# rounding
data = round(data / 1000, 0).astype(int)
        
                                            
mask = emptable.loc[:, (slice(None), colsforrounding)] < 6000
emptable[mask] = 0
emptable.loc[:, (slice(None), colsforrounding)] = round(emptable.loc[:, (slice(None), colsforrounding)] / 1000, 0).astype(int)
final = final.fillna(0)
        
        

# marks=pytest.mark.xfail
import pytest
@pytest.mark.parametrize('test_input,expected', [
    pytest.param('sex', 0, marks=pytest.mark.basic),
    pytest.param('ethnicity', 0, marks=pytest.mark.basic),
    pytest.param('dcms_ageband', 0, marks=pytest.mark.basic),
    pytest.param('qualification', 0, marks=pytest.mark.basic), # publication numbers dont add up - go through with penny - turn's out there is an extra column which is hidden by the publication called don't know which explains all this
    pytest.param('ftpt', 0, marks=pytest.mark.basic),
    pytest.param('nssec', 0, marks=pytest.mark.basic),
    pytest.param('region', 0, marks=pytest.mark.basic),
    pytest.param('cs', 0, marks=pytest.mark.basic),
    pytest.param('ci', 0, marks=pytest.mark.basic),
    pytest.param('culture', 0, marks=pytest.mark.basic),
    pytest.param('digital', 0, marks=pytest.mark.basic),
    pytest.param('gambling', 0, marks=pytest.mark.basic),
    pytest.param('sport', 0, marks=pytest.mark.basic),
    pytest.param('telecoms', 0, marks=pytest.mark.basic),
])
def test_datamatches(test_input, expected):
    assert sum((differencelist[test_input] < -0.05).any()) == expected
    assert sum((differencelist[test_input] > 0.05).any()) == expected


# for region the total won't = the sum anyway, so don't need to do annonymisation
