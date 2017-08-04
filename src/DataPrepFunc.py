import pandas as pd
from DataPrepClasses import *

# Name Definitions
fpes = 'full_parttime_employment_stat'
cs = 'country_self'
cf = 'country_father'
cm = 'country_mother'
cz = 'citizenship'
wph = 'wage_per_hour'
edu = 'education'
na = ['?',' ?',' ? ','? ','NA','nan']

def mode(df):
    md = df.mode()
    if not md.empty:
        return md[0]
    else:
        return 'unknown'
        
def FixData1(src_path,temp_path,ch_sz):
    # Drop unneeded columns and group some feuture classes together
    FullX = pd.read_csv(src_path,chunksize=ch_sz,na_values=na) 
    count = 0   
    for X in FullX:
        # Drop Unwanted Columns
        ## unemployment: the other category 'employment_stat' has all info needed
        ## migration: I felt it was mostly repeated, I kept migration_reg
        ## year: just the year the data was collected
        ## enrollment: we already have the 'education' and 'job status'
        ## occupation/industry code: we already have similar features
        ## state/region of previous residence: most data is missing
        ## family_stats: family summary, age and job status are enough
        ## veteran: veteran benefits will do
        ## business_or_self_employed: we have class of worker
        col2drop = ['reason_for_unemployment', 'migration_msa', 
                    'migration_within_reg','migration_sunbelt','year',
                    'enrolled_in_edu_inst_lastwk', 'major_occupation_code',
                    'major_industry_code', 'state_of_previous_residence',
                    'region_of_previous_residence', 'd_household_family_stat',
                    'fill_questionnaire_veteran_admin', 
                    'business_or_self_employed']
        X.drop(col2drop, axis=1, inplace=True)
                   
        # Group Some Categories
        ## education
        school = ['Children', '11th grade','7th and 8th grade', '10th grade',
                  '1st 2nd 3rd or 4th grade','9th grade','5th or 6th grade', 
                  '12th grade no diploma', 'Less than 1st grade']
        post_school = ['Some college but no degree', 'High school graduate']
        college = ['Bachelors degree(BA AB BS)', 
                   'Associates degree-occup /vocational',
                   'Associates degree-academic program']
        graduate = ['Masters degree(MA MS MEng MEd MSW MBA)', 
                    'Doctorate degree(PhD EdD)',
                    'Prof school degree (MD DDS DVM LLB JD)']
        h=(X[edu].str.strip().isin(school))
        X.loc[h,edu]='school'
        h=(X[edu].str.strip().isin(post_school))
        X.loc[h,edu]='post_school'
        h=(X[edu].str.strip().isin(college))
        X.loc[h,edu]='college'
        h=(X[edu].str.strip().isin(graduate))
        X.loc[h,edu]='graduate'
        ## employment status
        unemployed = ['Children or Armed Forces', 'Not in labor force', 
                      'Unemployed full-time']
        PT = ['PT for econ reasons usually PT','Unemployed part- time',
              'PT for econ reasons usually FT']
        FT = ['Full-time schedules', 'PT for non-econ reasons usually FT']
        h=(X[fpes].str.strip().isin(unemployed))
        X.loc[h,fpes]='unemployed'
        h=(X[fpes].str.strip().isin(PT))
        X.loc[h,fpes]='PT'
        h=(X[fpes].str.strip().isin(FT))
        X.loc[h,fpes]='FT'
        ## d_household_summary fix
        h=(X['d_household_summary'].str.strip()=='Child under 18 ever married')
        X.loc[h,'d_household_summary']='Child under 18 never married'
        ## Group countries together
        South_America = ['Columbia','Peru','Ecuador',]
        Central_America = ['Mexico','Trinadad&Tobago','Jamaica','Puerto-Rico',
                           'Dominican-Republic','Outlying-U S (Guam USVI etc)', 
                           'Guatemala','El-Salvador','Cuba','Nicaragua',
                           'Honduras','Haiti','Panama']
        East_Asia = ['South Korea','Taiwan','Japan','China','Hong Kong']
        South_Asia = ['Philippines','Vietnam', 'Cambodia','India',
                      'Thailand','Laos']
        West_Europe = ['Germany','Italy','England','Ireland',
                       'France','Scotland','Portugal','Holand-Netherlands']
        East_Europe = [' Poland','Yugoslavia','Greece','Hungary']      
        tokens = [cs, cf, cm]
        for tok in tokens:
            h=(X[tok].str.strip().isin(South_America))
            X.loc[h,tok]='South America'
            h=(X[tok].str.strip().isin(Central_America))
            X.loc[h,tok]='Central America'
            h=(X[tok].str.strip().isin(East_Asia))
            X.loc[h,tok]='East Asia'
            h=(X[tok].str.strip().isin(South_Asia))
            X.loc[h,tok]='South Asia'
            h=(X[tok].str.strip().isin(West_Europe))
            X.loc[h,tok]='West Europe'
            h=(X[tok].str.strip().isin(East_Europe))
            X.loc[h,tok]='East_Europe'
        
        # Fix labels
        if X['income_level'].dtype == 'object':    
            h=(X['income_level']=='-50000')
            X.loc[h]='0'
            X.loc[~h]='1'
        else:
            h=(X['income_level']==-50000)
            X.loc[h]=0
            X.loc[~h]=1

        # Saving The Changes
        if count==0:
            X.to_csv(temp_path, index=False)
            count+=1
        else:
            X.to_csv(temp_path, header=None, index=False, mode='a')
            count+=1
            
def FindAvgWage(temp_path,ch_sz):
    FullX = pd.read_csv(temp_path,chunksize=ch_sz,na_values=na) 
    avg_wage=[]
    sz_count = []   
    for X in FullX:
        nz = (X[wph]!=0)
        avg_wage.append(pd.pivot_table(X.loc[nz], values=wph, columns=[edu,fpes,'age']))
        sz_count.append(len(X))
    Tot_avg_wage = sz_count[0]*avg_wage[0]
    for j in range(1,len(avg_wage)):
        Tot_avg_wage = Tot_avg_wage.add(sz_count[j]*avg_wage[j],fill_value=0)
    Tot_avg_wage/=sum(sz_count)  
    return Tot_avg_wage.astype(int)   
    
def FixData2(temp_path,mod_path,ch_sz,Tot_avg_wage):
    # Impute missing data and add new features
    FullX = pd.read_csv(temp_path,chunksize=ch_sz,na_values=na) 
    count = 0   
    for X in FullX:
        # Data Imputing
        ## Fill self country
        table = pd.pivot_table(X, values=cs, index=[cz], aggfunc=mode)
        index = (table.values=='unknown')
        if sum(index)>0:
            table.loc[index,cs]='United-States'
        miss = X[cs].isnull()
        X.loc[miss,cs] = X.loc[miss,cz].apply(lambda x: table.loc[x])
        ## Fill father/mother country
        table = pd.pivot_table(X, values=[cf,cm], index=[cs], aggfunc=mode)
        index = (table.values=='unknown')
        table.loc[index[:,0],cf]=table[index[:,0]].index
        table.loc[index[:,1],cm]=table[index[:,1]].index
        miss = X[cf].isnull()
        X.loc[miss,cf] = X.loc[miss,cs].apply(lambda x: table.loc[x,cf])
        miss = X[cm].isnull()
        X.loc[miss,cm] = X.loc[miss,cs].apply(lambda x: table.loc[x,cm])
        ## Fill hispanic_origin
        miss = X['hispanic_origin'].isnull()
        X.loc[miss,'hispanic_origin'] = mode(X['hispanic_origin'])
        ## Fill migration_reg
        md = mode(X['migration_reg'])
        miss = X['migration_reg'].isnull()
        X.loc[miss,'migration_reg'] = md
        miss = (X['migration_reg']=='Not in universe')
        X.loc[miss,'migration_reg'] = md
        ## Fill wage per hour
        miss = (X[wph]==0)
        for i,row in X.loc[miss].iterrows():
            ind = tuple([row[edu],row[fpes],row['age']])
            if ind in Tot_avg_wage:
                X.loc[i,wph] = Tot_avg_wage.loc[ind]
            else:
                X.loc[i,wph] = 0

        # New Features Added
        ## net_capital_gains
        X['net_investment_gains'] = X['dividend_from_Stocks'] \
                                    + X['capital_gains'] \
                                    - X['capital_losses']
        col2drop = ['dividend_from_Stocks', 'capital_gains', 'capital_losses']
        X.drop(col2drop, axis=1, inplace=True)
        # Saving The Changes
        if count==0:
            X.to_csv(mod_path, index=False)
            count+=1
        else:
            X.to_csv(mod_path, header=None, index=False, mode='a')
            count+=1
            
def EncodeTrainData_OHC(src_path,OHC_path,ch_sz):
    # Implement One Hot Encoder for categorical features
    FullX = pd.read_csv(src_path,chunksize=ch_sz,na_values=na) 
    # get the categorical data columns
    X = FullX.get_chunk(1)
    num_col = ['age','wage_per_hour',
               'num_person_Worked_employer','income_level',
               'weeks_worked_in_year','net_investment_gains']
    cat_col = X.columns.difference(num_col)
    Encoder = OneHotEncoder()
    for X in FullX:
        Encoder.partial_fit(X[cat_col])
    
    new_col = []
    for j,feature in enumerate(Encoder.keymap):
        feat_col = [cat_col[j]+'_'+str(i) for i in range(len(feature))]
        new_col.extend(feat_col)
        
    Encoder.set_columns(cat_col,num_col,new_col)
    
    FullX = pd.read_csv(src_path,chunksize=ch_sz,na_values=na) 
    count = 0   
    for X in FullX:
        X_OHC = pd.DataFrame(Encoder.transform(X[cat_col]),columns=new_col)
        X_OHC[num_col] = X[num_col]
        # Saving The Changes
        if count==0:
            X_OHC.to_csv(OHC_path, index=False)
            count+=1
        else:
            X_OHC.to_csv(OHC_path, header=None, index=False, mode='a')
            count+=1
    return Encoder

def EncodeTrainData_binary(src_path,binary_path,ch_sz):
    # Implement Binary Encoder for categorical features
    FullX = pd.read_csv(src_path,chunksize=ch_sz,na_values=na) 
    # get the categorical data columns
    X = FullX.get_chunk(1)
    num_col = ['age','wage_per_hour',
               'num_person_Worked_employer','income_level',
               'weeks_worked_in_year','net_investment_gains']
    cat_col = X.columns.difference(num_col)
    Encoder = BinaryEncoder()
    for X in FullX:
        Encoder.partial_fit(X[cat_col])
    
    new_col = []
    for j,feature in enumerate(Encoder.keymap):
        bin_sz = len(bin(len(feature)))-2
        feat_col = [cat_col[j]+'_'+str(i) for i in range(bin_sz)]
        new_col.extend(feat_col)
        
    Encoder.set_columns(cat_col,num_col,new_col)
    
    FullX = pd.read_csv(src_path,chunksize=ch_sz,na_values=na) 
    count = 0   
    for X in FullX:
        X_bin = pd.DataFrame(Encoder.transform(X[cat_col]),columns=new_col)
        X_bin[num_col] = X[num_col]
        # Saving The Changes
        if count==0:
            X_bin.to_csv(binary_path, index=False)
            count+=1
        else:
            X_bin.to_csv(binary_path, header=None, index=False, mode='a')
            count+=1
    return Encoder
    
def EncodeTestData(src_path,target_path,ch_sz,Encoder):
    num_col = Encoder.num_col
    cat_col = Encoder.cat_col
    new_col = Encoder.new_col

    FullX = pd.read_csv(src_path,chunksize=ch_sz,na_values=na) 
    count = 0   
    for X in FullX:
        for col in cat_col:
            if X[col].dtype == 'object':
                X[col] = X[col].str.strip()
        X_Enc = pd.DataFrame(Encoder.transform(X[cat_col]),columns=new_col)
        X_Enc[num_col] = X[num_col]
        # Saving The Changes
        if count==0:
            X_Enc.to_csv(target_path, index=False)
            count+=1
        else:
            X_Enc.to_csv(target_path, header=None, index=False, mode='a')
            count+=1