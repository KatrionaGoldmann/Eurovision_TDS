"""
Eurovision voting scores from 1998

Using a data frame from Kaggle which goes up to 2019 available at: 
    https://www.kaggle.com/datasets/datagraver/eurovision-song-contest-scores-19752019). 
"""

import pandas as pd
import numpy as np
import pickle
import pycountry

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

    # get the country codes as a list
    country_codes = list(country_codes_dict.values())

    # save country_codes_dict to pickle
    if country_codes_pickle_file is not None:
        with open(country_codes_pickle_file, 'wb') as handle:
            pickle.dump(country_codes_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # replace the to country with the country code
    df['To country'] = df['To country'].map(country_codes_dict)
    df['From country'] = df['From country'].map(country_codes_dict)

    return df, country_codes

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
    df_to_2016.head()

    # Post-2016 Scores : must be rescaled for combined tele-voting and jury scores
    # create new df with column for each country
    df_from_2016 = pd.DataFrame(columns=['Year', 'From country'] + country_codes)
    df_from_2016 = df_from_2016.set_index(['Year', 'From country'])

    # for each year, and country, get the total points
    for i in range(2016, 2022):
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

    # Combine the two data frames
    df_all = df_from_2016.add(df_to_2016, fill_value=0)

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
    Main function to curate the voting scores
    """
    df = import_and_clean(filename)
    df, country_codes = convert_country_names(df, replace_list, country_codes_pickle_file)
    df = calculate_voting_scores(df, country_codes)
    return get_final_output(df, output_file)
