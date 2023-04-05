"""
Determine how languages relate to a country

Utiltieis for determining whether a list of languages contains a
country's official language, or a language that isn't the official
language of the country.
"""

import pandas as pd
import requests
import re
from bs4 import BeautifulSoup

class LanguageMatch():
    """Performi matching between a country and its official languages.
    """
    language_map = None

    def __init__(self, country_codes_dict):
        print("Loading country to language mapping")
        # Get the official languages from Wikipedia
        url=f"https://en.wikipedia.org/wiki/List_of_official_languages_by_country_and_territory"
        response=requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        tables=soup.find_all('table',{'class':"wikitable"})

        table = tables[0]

        df_table=pd.read_html(str(table))
        df_table=pd.DataFrame(df_table[0])

        print("Processing language mapping")

        # Simplify the column naming
        df_table.rename(columns={'Official language':'official', 'Country/Region':'country'}, inplace=True)

        # Replace United Kingdom and Crown dependencies etc with United Kingdom
        df_table['country'] = df_table['country'].replace('United Kingdom and Crown dependencies etc.','United Kingdom')

        # Embed the country codes and remove countries without a code
        df_table['country'] = df_table['country'].str.lower()
        df_table['country'] = df_table['country'].apply(lambda x: re.sub("\[.*?\]","",x))
        df_table['code'] = df_table['country'].map(country_codes_dict)
        df_table=df_table[["code", "official"]]
        df_table = df_table.dropna(subset=["code"])

        # Tidy the columns
        df_table = df_table.fillna('')
        df_table['official'] = df_table['official'].apply(lambda x: re.sub("\[.*?\]|\(.*?\)","",x))

        # Add missing languages or country names
        df_table.loc[len(df_table)] = ['YU', 'serbian montegegrin']
        df_table.loc[df_table['code'] == 'LT', 'official'] = 'samogitian ' + df_table.loc[df_table['code'] == 'LT', 'official']
        df_table.loc[df_table['code'] == 'FR', 'official'] = 'breton corsican ' + df_table.loc[df_table['code'] == 'FR', 'official']
        df_table.loc[df_table['code'] == 'SI', 'official'] = 'slovenian ' + df_table.loc[df_table['code'] == 'SI', 'official']
        df_table.loc[df_table['code'] == 'EE', 'official'] = 'võro ' + df_table.loc[df_table['code'] == 'EE', 'official']

        # Convert into lists of languages
        df_table['official'] = df_table['official'].apply(lambda x: x.lower().split())
        self.language_map = df_table

    def contains_own_language(self, country_code, languages):
        """Returns whether a list of languages contains one of the official langauges of a country

        Args:
            country_code (str): the country code for the country
            languages ([str]): a list of languages
        Returns:
            bool: true if the list contains one of the country's official languagges, false otherwise
        """
        official = self.language_map[self.language_map["code"] == country_code]["official"]
        contains = bool(set(languages) & set([x for y in official for x in y]))
        return contains

    def contains_other_language(self, country_code, languages):
        """Returns whether a list of languages contains something other than a country's official language

        Args:
            country_code (str): the country code for the country
            languages ([str]): a list of languages
        Returns:
            bool: true if the list contains a language that isn't one of the country's official languages
        """
        official = self.language_map[self.language_map["code"] == country_code]["official"]
        contains = bool(set(languages) - set([x for y in official for x in y] + ['english']))
        return contains


