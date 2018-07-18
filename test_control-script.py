import pandas as pd
import numpy as np
import itertools
import pytest

current_year = 2016

raw_data = {}
for year in range(2011, current_year + 1):
    raw_data[year] = pd.read_csv("~/data/cleaned_" + str(year) + "_df.csv")
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

mycat = ['sex', 'region']

agg_2016 = clean_data(year = 2016)
agg_2015 = clean_data(year = 2015)
agg_2014 = clean_data(year = 2014)
agg_2013 = clean_data(year = 2013)
agg_2012 = clean_data(year = 2012)
agg_2011 = clean_data(year = 2011)


cleaned_data2 = [clean_data(i) for i in range(2011, current_year + 1)]
for check in cleaned_data2:
    print(check.dtypes)
final = pd.concat(cleaned_data2, ignore_index=True)

    
    import aggregate_data_func
    allyears[k] = aggregate_data_func.aggregate_data(allyears[k], {'mycat': 'region'})
    
    mycol = allyears[k][allyears[k]['emptype'] == 'total']
    mycol = mycol.groupby(['sector']).sum()
    mycol = mycol.rename(columns={'count': k})
    data.append(mycol)






timeseries = pd.concat(data, axis=1)


    dfcopy = df.copy()
    if mycat == 'qualification':
        dfcopy = dfcopy[dfcopy.qualification != 'dont know']
        dfcopy = dfcopy[dfcopy.qualification != 'nan']
    
    if time_series:
        data = timeseries
        
        if current_year == 2016:
            tourism = pd.DataFrame(columns=data.columns)
            tourism.loc['tourism'] = [1, 2, 3, 4, 5, 6]
    
            percuk = pd.DataFrame(columns=data.columns)
            percuk.loc['percuk'] = [1, 2, 3, 4, 5, 6]
        
        if current_year == 2017:
            tourism = pd.DataFrame(columns=data.columns)
            tourism.loc['tourism'] = [1, 2, 3, 4, 5, 6, 7]
    
            percuk = pd.DataFrame(columns=data.columns)
            percuk.loc['percuk'] = [1, 2, 3, 4, 5, 6, 7]

        # add tourism
        data = data.append(tourism)
        data = data.append(percuk)
        
        # rounding
        data = round(data / 1000, 0).astype(int)
        
        
        # reorder rows
        myroworder = ["civil_society", "creative", "culture", "digital", "gambling", "sport", "telecoms", 'tourism', "all_dcms", 'percuk', "total_uk"]
        data = data.reindex(myroworder)
        
        
        # CHECK DATA MATCHES PUBLICATION
        # store anonymised values as 0s for comparison and data types
        import check_data_func
        from openpyxl import load_workbook, Workbook
        from openpyxl.utils.dataframe import dataframe_to_rows
#        exceldataframe = check_data_func.check_data(data, wsname, startrow, startcol, finishrow, finishcol)
#        # compare computed and publication data
#        
#        difference = data - exceldataframe
#        
#        if sum((difference > 1).any()) != 0:
#            print(table + ': datasets dont match')
            
    # MAIN GROUP OF FUNCTIONS
    else:        
        cat = mycat
        # CLEANING DATA - adding up main and second jobs, calculating some totals, columns for sector, cat, region, count
        # there doesn't appear to be any tables which use both region and a demographic category, so simply remove region or replace cat column with it.
        sic_level = False
        
        import clean_data_func
        agg = clean_data_func.clean_data(dfcopy, table_params, sic_mappings, regionlookupdata, weightedcountcol)
        agg = agg[['sector', mycat, 'sic', 'mainemp', 'secondemp', 'mainselfemp', 'secondselfemp']]
        
        spsslist = """
        1820, 2611, 2612, 2620, 2630, 2640, 2680, 3012, 3212, 3220, 3230, 4651, 4652, 4763, 4764, 4910, 4932, 4939, 5010, 5030, 5110, 5510, 5520, 5530, 5590, 5610, 5621, 5629, 5630, 5811, 5812, 5813, 5814, 
        5819, 5821, 5829, 5911, 5912, 5913, 5914, 5920, 6010, 6020, 6110, 6120, 6130, 6190, 6201, 6202, 6203, 6209, 6311, 6312, 6391, 6399, 6820, 7021, 7111, 7311, 7312, 7410, 7420, 7430, 7711, 7721, 
        7722, 7729, 7734, 7735, 7740, 7911, 7912, 7990, 8230, 8551, 8552, 9001, 9002, 9003, 9004, 9101, 9102, 9103, 9104, 9200, 9311, 9312, 9313, 9319, 9321, 9329, 9511, 9512 """
        spsslist = spsslist.replace('\n', '')
        spsslist = spsslist.replace('\t', '')
        spsslist = spsslist.replace(' ', '')
        mylist = np.array(spsslist.split(","))
        
        import aggregate_data_func
        aggfinal = aggregate_data_func.aggregate_data(agg, table_params)
            
        if emptypecats == False:
            aggfinal = aggfinal[aggfinal['emptype'] == 'total']
                            
        # SUMMARISING DATA
        if cat == 'region':
            import region_summary_table_func
            final = region_summary_table_func.region_summary_table(aggfinal, table_params)
        else:
            import summary_table_func
            final = summary_table_func.summary_table(aggfinal, cat, perc, cattotal, catorder) 
        
        # ANONYMISING DATA
        import anonymise_func
        data = anonymise_func.anonymise(final, emptypecats, anoncats, cat, sector)
        
        # add extra anonymisation to match publication
        if cat == 'sex':
            data.loc['telecoms', ('employed', 'Total')] = 0
            data.loc['telecoms', ('self employed', 'Total')] = 0
    
        if table == 'cs':
            data.loc['Northern Ireland', 'total'] = 0
            data.loc['Northern Ireland', 'perc_of_all_regions'] = 0
        
        # CHECK DATA MATCHES PUBLICATION
        # store anonymised values as 0s for comparison and data types
        import check_data_func
        from openpyxl import load_workbook, Workbook
        from openpyxl.utils.dataframe import dataframe_to_rows
        exceldataframe = check_data_func.check_data(data, wsname, startrow, startcol, finishrow, finishcol)
        # compare computed and publication data
        
        difference = data - exceldataframe
        
        if sum((difference > 1).any()) != 0:
            print(cat + ': datasets dont match')
        
        #mylist = make_cat_data()
        #data = mylist[1]
        #difference = mylist[0]

    differencelist.update({table : difference})
    
    ws = wb[wsname]
    rows = dataframe_to_rows(data, index=False, header=False)
    
    for r_idx, row in enumerate(rows, 1):
        for c_idx, value in enumerate(row, 1):
             ws.cell(row=r_idx + startrow - 1, column=c_idx + 1, value=value)
 
  
"""
wsname = "3.5 - Gender (000's)"
startrow = 9
finishrow = 17
finishcol = 16
"""

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


wb.save('employment_pub_2017_python_unfinished.xlsx')

# get final table with hierarchical indexes which I can check against those read in from excel (including order of rows etc), but then just output the values to the formatted excel templates

# for anonymisation, it seems something quite simple will work initially. Only gender, age, and nnsec seem like they will require rules

# for region the total won't = the sum anyway, so don't need to do annonymisation

# use openpyxl initially and move on to xlwings if necessary. xlsxwriter cannot read workbooks but should be considered if producing workbooks from scratch.













































