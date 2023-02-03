

import pandas as pd
from wikipeople import wikipeople as wp # use wp to get the gender data as needed. 
import re
import requests
import multiprocessing as mp
import numpy as np

def search_wikidata(string):
    """
    Query the Wikidata API using the wbsearchentities function.
    Return the concept ID of the search result that has the musician identifier.
    """
    query = 'https://www.wikidata.org/w/api.php?action=wbsearchentities&search='
    query += string
    query += '&language=en&format=json'
    music_markers = [
        'singer', 'artist', 'musician', 'music',
        'band', 'group', 'duo', 'ensemble'
    ]
    res = requests.get(query).json()
    if len(res['search']) == 0:
        raise Exception('Wikidata search failed.')
    target = 0
    for i in range(len(res['search'])):
        if 'description' in res['search'][i]['display']:
            description = res['search'][i]['display']['description']['value']
            if any(markers in description for markers in music_markers):
                concept_id = res['search'][i]['id']
                contestant_in = wp.get_property(concept_id, 'P1344')[-1]
                if "Eurovision" in contestant_in:
                    target = i
    return res['search'][target]['id']


def lookup_gender(name):
    """Find gender of given name. If the name is not related to a wiki entry it will return 'RNF' (record not found).
    Alternatively it will return the gender if the record has one or NA if it does not have this property.

    Args:
        name (str): The name to search

    Returns:
        str: The gender of the person searched
    """
    gender = 'RNF'
    try:
        data = search_wikidata(name)
        gender = wp.get_property(data, 'P21')[-1]
        instance = wp.get_property(data, 'P31')[-1]
        if gender == 'NA':
            group_checks = [
                "group", "duo", "trio", "music", "band", "ensemble"
            ]
            if any(x in instance for x in group_checks):
                gender = "group"
    except:
        Exception('Wikidata search failed.')
    return gender

def get_artist_gender(search):
    s = requests.Session()
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "opensearch",
        "namespace": "0",
        "search": search,
        "limit": "10000",
        "format": "json"
    }
    r = s.get(url=url, params=params)
    names = r.json()[1]
    gender = ""
    if len(names) < 1:
        gender = "RNF"
    else:
        for n in names:
            gender = lookup_gender(n)
            if not any(gender == x for x in ["RNF", "NA"]):
                return gender
    if gender == "RNF" or gender == "NA":
        if "&" in search:
            gender = "group"
    return gender

def final_fixes(df):
    # group into gender fluid
    idx = np.where(
        (df['Gender'] == "trans woman") |
        (df['Gender'] == "genderfluid") |
        (df['Gender'] == "non-binary")
    )
    df.loc[idx[0], "Gender"] = "gender-fluid"

    # clear up male organism item
    idx = np.where(
        df['Gender'] == "male organism"
    )
    df.loc[idx[0], "Gender"] = "male"

    # manually update remaining values
    df.loc[df["Artist"] == "Die Orthopädischen Strümpfe", "Gender"] = "group"
    df.loc[df["Artist"] == "Michael Hajiyanni", "Gender"] = "male"
    df.loc[df["Artist"] == "Mietek Szcześniak", "Gender"] = "male"
    df.loc[df["Artist"] == "Lado Members", "Gender"] = "group"
    df.loc[df["Artist"] == "Copycat", "Gender"] = "group"
    df.loc[df["Artist"] == "Agathon Iakovidis", "Gender"] = "male"
    df.loc[df["Artist"] == "Minus One", "Gender"] = "group"
    df.loc[df["Artist"] == "ZAA Sanja Vučić", "Gender"] = "female"
    df.loc[df["Artist"] == "Julia Samoylova", "Gender"] = "female"
    df.loc[df["Artist"] == "Amanda Georgiadi Tenfjord", "Gender"] = "female"
    
    # sanity check 
    assert len(df.loc[df["Gender"] == "NA"]) == 0, "NA genders > 0"
    assert len(df.loc[df["Gender"] == "RNF"]) == 0, "RNF genders > 0"
    
    return df

def main():
    # Load data
    df = pd.read_json('data/eurovision-lyrics-2022.json', orient = 'index')

    # reduce data to post 1997
    mask = (df['Year'] > 1997)
    df2 = df.loc[mask].reset_index()

    # Clean name up (remove any brackets or extra artist information)
    df2["Artist"] = [re.sub("[\(\[].*?[\)\]]", "", x) for x in df2["Artist"]]
    df2["Artist"] = [x.split("feat.")[-1] for x in df2["Artist"]]
    df2["Artist"] = [x.replace("/", "&") for x in df2["Artist"]]
    df2["Artist"] = [x.strip() for x in df2["Artist"]]
    
    with mp.Pool(mp.cpu_count() - 1) as p:
        genders = p.map(get_artist_gender, df2["Artist"])
    
    df2["Gender"] = genders
    df3 = final_fixes(df2)
    df3.to_json('data/eurovision-lyrics-2022-Gender.json', orient = 'index')

    

if __name__ == "__main__":
    main()
    