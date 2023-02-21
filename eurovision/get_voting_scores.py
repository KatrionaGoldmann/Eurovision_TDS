"""
Eurovision voting scores from 1998

Using a data frame from Kaggle which goes up to 2019 available at: 
    https://www.kaggle.com/datasets/datagraver/eurovision-song-contest-scores-19752019). 
"""

import pandas as pd
import numpy as np
import pickle
import pycountry
import requests
from bs4 import BeautifulSoup

def import_and_clean(filename):
    """
    Import the local excel data set. Subset to finals and post-1998 only. Clean text columns.  
    """
    # read in xlsx
    df_input = pd.read_excel(filename)
    df = df_input[df_input['Year'] > 1997]
    df = df.loc[df['(semi-) final'] == 'f'] # Finals only
    df = df.drop_duplicates()

    # remove white space from countries
    df['To country'] = df['To country'].str.strip()
    df['From country'] = df['From country'].str.strip()

    # lower case
    df['To country'] = df['To country'].str.lower()
    df['From country'] = df['From country'].str.lower()

    return df

def convert_country_names(df, replace_list=None, country_codes_pickle_file=None):
    """
    Tidy the country names using a manually curated list of replacements.
    Convert the country names to ISO alpha-2 codes.
    Create dictionary of country names and codes.
    """

    if replace_list is None:
        # tidy country names: fix typos, fill whitespace, rename
        replace_list = [['-', ' '],
                        ['&', 'and'], 
                        ['netherands', 'netherlands'],
                        ['f.y.r. macedonia', 'north macedonia'], 
                        ['russia', 'russian federation'], 
                        ['the netherlands', 'netherlands'], 
                        ['czech republic', 'czechia'],
                        ['serbia and montenegro', 'yugoslavia'],
                        ['moldova', 'moldova, republic of']] 

    # replace and tidy country names
    for replacements in replace_list: 
        df['To country'] = df['To country'].str.replace(replacements[0], replacements[1], regex=True)
        df['From country'] = df['From country'].str.replace(replacements[0], replacements[1], regex=True)

    countries = [df['From country'], df['To country']]
    countries = np.sort(np.unique(countries))
    
    # Get country codes
    country_codes = []
    for country in countries:
        try:
            country_codes.append(pycountry.countries.get(name=country).alpha_2)
        except:
            country_codes.append('NaN')

    # convert list to dictionary
    country_codes_dict = dict(zip(countries, country_codes))

    # Add 'serbia and montenegro' or 'yugoslavia' code
    country_codes_dict.update({"yugoslavia": "YU"})

    # print those NaN, who did not match to a country code
    for key, value in country_codes_dict.items():
        if value == 'NaN':
            print('Missing code: ' + key) 

    # We are saving the country_code as a pickle for convenience for other data curation pipelines
    # Add other country names which could realistically be included
    country_codes_dict.update({"serbia and montenegro": "YU"})
    country_codes_dict.update({"czech republic": "CZ"})
    country_codes_dict.update({'f.y.r. macedonia': 'MK'})
    country_codes_dict.update({'russia': 'RU'})
    country_codes_dict.update({'the netherlands': 'NL'})
    country_codes_dict.update({'moldova': 'MD'})

    # get the country codes as a list
    country_codes = list(np.unique(list(country_codes_dict.values())))

    # save country_codes_dict to pickle
    if country_codes_pickle_file is not None:
        with open(country_codes_pickle_file, 'wb') as handle:
            pickle.dump(country_codes_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # replace the to country with the country code
    df['To country'] = df['To country'].map(country_codes_dict)
    df['From country'] = df['From country'].map(country_codes_dict)

    return df, country_codes, country_codes_dict

def import_votes_from_wiki(year, country_codes_dict, table_ids=[16, 17]):

    url=f"https://en.wikipedia.org/wiki/Eurovision_Song_Contest_{year}#Final_2"
    response=requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    tables=soup.find_all('table',{'class':"wikitable"})

    out_tables=[]

    for table in [tables[i] for i in table_ids]:
        df_table=pd.read_html(str(table))
        df_table=pd.DataFrame(df_table[0])

        # remove redundant rows/columns
        df_table = df_table.drop(df_table.columns[[0, 2, 3, 4]], axis=1)
        df_table = df_table.drop(df_table.index[[0, 2]], axis=0) 

        # set the index to the first column
        df_table = df_table.set_index(df_table.columns[0])

        # set the column names as the first row
        df_table.columns = df_table.iloc[0]
        df_table = df_table.drop(df_table.index[0])

        # replace NaN with 0
        df_table = df_table.fillna(0)

        # squash the column index with stack
        df_table = df_table.stack().reset_index()

        df_table.columns = ['To country', 'From country', 'Points']
        df_table['Jury or Televoting'] = 'J'

        df_table['Year'] = year
        df_table['(semi-) final'] = 'f'
        df_table['Edition'] = f"{year}f"
        df_table['Duplicate'] = ""
        
        # if From country = To country, set Duplicate = x
        df_table.loc[df_table['From country'] == df_table['To country'], 'Duplicate'] = 'x'
        df_table.loc[df_table['From country'] == df_table['To country'], 'Points'] = np.nan

        # convert the country names to country codes
        df_table['To country'] = df_table['To country'].str.lower().map(country_codes_dict)
        df_table['From country'] = df_table['From country'].str.lower().map(country_codes_dict)

        # re-order the columns to match the original data   
        df_table = df_table[['Year', '(semi-) final', 'Edition', 
                             'Jury or Televoting','From country', 'To country', 
                             'Points', 'Duplicate']]
        
        df_table['Points'] = df_table['Points'].astype(np.float).astype("Int32")

        out_tables=out_tables+[df_table]

    jury_table = out_tables[0]
    tele_table = out_tables[1]

    tele_table['Jury or Televoting'] = 'T'

    combined_table = pd.concat([jury_table, tele_table])

    return combined_table


def calculate_voting_scores(df, country_codes):
    """
    Return the voting scores for each country, for each year, 
    - 1998 - 2016: return the raw combined score
    - 2016 onwards: combined the tele-voting and jury score then refactor for 1..8, 10, 12 scale.
    """

   # pre-2016 scores
    df2 = df.loc[df['Year'] < 2016]
    df_to_2016 = df2.pivot(index=['Year', 'From country'], columns=['To country'], 
                        values='Points')
    df_to_2016 = df_to_2016

    # Post-2016 Scores : must be rescaled for combined tele-voting and jury scores
    # create new df with column for each country
    df_from_2016 = pd.DataFrame(columns=['Year', 'From country'] + country_codes)
    df_from_2016 = df_from_2016.set_index(['Year', 'From country'])

    # for each year, and country, get the total points
    for i in range(2016, 2023):
        subset = df.loc[(df['Year'] == i)] 
        for country in subset['From country'].unique():  
            
            subset_country = subset.loc[subset['From country'] == country,]
            subset_country = subset_country.pivot(index=['Jury or Televoting'], 
                                                    columns=['To country'], 
                                                    values='Points')

            # if the number of rows > 0, then add the total row
            if len(subset_country.index) > 0:

                # Add row for total points
                # - 50:50 split between jury and tele-vote so can half the points
                # - this results in multiple performers getting the same score but 
                # - scales with years prior
                subset_country.loc['Total'] = subset_country.sum()

                # Add row for the order rank of the countries        
                s = subset_country.loc['Total']
                subset_country.loc['Rank'] = [sorted(s, reverse=True).index(val) + 1 for val in s] 

                # Map 1-10 Rank to 1-8, 10, 12 Points
                subset_country.loc['Points'] = subset_country.loc['Rank'].map({1: 12, 2: 10, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1})
                subset_country.loc['Points'] = subset_country.loc['Points'].fillna(0)

                subset_country = subset_country.loc[['Points'],:]
                subset_country['Year'] = i
                subset_country['From country'] = country

                subset_country = subset_country.set_index(['Year', 'From country'])

                # order columns
                subset_country = subset_country.reindex(columns=df_from_2016.columns)
                df_from_2016 = pd.concat([df_from_2016, subset_country])

    # remove duplicated rows
    df_from_2016 = df_from_2016.drop_duplicates()

    # Find country_codes not in column names and add them
    missing_countries = [c for c in country_codes if c not in df_to_2016.columns]
    for c in missing_countries:
        df_to_2016[c] = np.nan

    missing_countries = [c for c in country_codes if c not in df_from_2016.columns]
    for c in missing_countries:
        df_from_2016[c] = np.nan

    # reorder columns
    df_to_2016 = df_to_2016.reindex(columns=country_codes)
    df_from_2016 = df_from_2016.reindex(columns=country_codes)

    # Merge the two dataframes
    df_all = pd.concat([df_to_2016, df_from_2016])

    return df_all

def get_final_output(df, output_file=None):
    """
    convert df for bayesian model input (long format)
    """
    df_long = df.stack(dropna=False).reset_index()
    df_long.columns = ['Year', 'From country', 'To country', 'Votes']

    if output_file is not None:
        df_long.to_csv(output_file, index=False)
    
    return df_long

def get_voting_scores(filename, replace_list=None, country_codes_pickle_file= None, output_file=None):
    """
    Main wrapper function to curate the voting scores
    """
    df = import_and_clean(filename)
    df, country_codes, country_codes_dict = convert_country_names(df, replace_list, country_codes_pickle_file)
    votes_2021 = import_votes_from_wiki(2021, country_codes_dict)
    votes_2022 = import_votes_from_wiki(2022, country_codes_dict)
    df = pd.concat([df, votes_2021, votes_2022])
    df = calculate_voting_scores(df, country_codes)
    return get_final_output(df, output_file)
