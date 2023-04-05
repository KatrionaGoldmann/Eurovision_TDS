"""
Eurovision placeholder data for 2023

Creates a CSV file populated with 2023 data for the participating countries, without scores
since at time of writing they don't yet exist.
"""

import pandas as pd
import numpy as np
import pickle
import requests
import sys
from pathlib import Path
from bs4 import BeautifulSoup

from languagematch import LanguageMatch
from genderguess import GenderGuess

class EurovisionFuture():
    """Collect together information about acts that might appear in the
    Eurovision final, based on their appearance in the semi finals.
    """
    in_file = None
    out_file = None
    countries = None
    result = None
    country_codes_dict = None
    language_map = None
    df = None
    language_match = None
    gender = None

    def __init__(self, in_file, out_file, countries):
        self.in_file = in_file
        self.out_file = out_file
        self.countries = countries

        # Don't run if the files aren't present
        for file in [in_file, countries]:
            if not file.is_file():
                raise FileNotFoundError(f"file {str(file)} was not found")

        self.__load_country_codes()
        self.__import_existing()
        self.language_match = LanguageMatch(self.country_codes_dict)
        self.gender = GenderGuess(future=True)

    def __expand_data(self, collected):
        """Expand the data using a cross join on voting countries.

        For each row in the data frame, generate a dupilcate for each country
        that might vote for it, filling the "From country" column with the
        voting country.

        Args:
            collected (pd.DataFrame): the current data to be expanded
        Returns:
            pd.DataFrame: the existing data expanded with the "From country" column
        """
        from_country = pd.DataFrame({"From country": self.country_codes_dict.values()})
        collected = pd.merge(collected, from_country, how="cross")

        # Remove rows where country and from_country are the same (self-voting)
        collected = collected.drop(collected[collected["code"] == collected["From country"]].index)

        # Reset the index
        collected = collected.reset_index(drop=True)

        return collected

    def __transfer_existing_data(self, collected):
        """Transfer data from the historical data frame into the future frame.

        Transfers border and migration data from the most recent year for which
        it exists in the historical data into the future data frame.

        Args:
            collected (pd.DataFrame): the current data to be augmented
        Returns:
            pd.DataFrame: data updated with border and migration details
        """
        # Fill out the has_border entries from existing data
        print('Transfering border data')
        collected['has_border'] = collected[['code', 'From country']].apply(lambda x: self.__get_has_border(x['code'], x['From country']), axis=1)

        # Fill out migration data
        print('Transfering migration data')
        collected[['migration_band', 'migration_year', 'count', 'prop_emigrants']] = collected[['code', 'From country']].apply(lambda x: self.__get_migration(x['code'], x['From country']), axis=1)

        return collected

    def __get_has_border(self, code, from_country):
        """Check whether there's a land border bertween two countries

        Args:
            code (str): the country code of one country
            from_country (str): the country code of another country
        Returns:
            iot: 1 if the countries share a land border, 0 otherwise
        """
        result = self.df[(self.df['code'] == code) & (self.df['From country'] == from_country)].nlargest(1, 'Year')['has_border'].values
        return result[0] if len(result) > 0 else 0

    def __get_migration(self, code, from_country):
        """Get migration data from one countrhy to another

        Args:
            code (str): the country code for migrating to
            from_country (str): the country code for migraition from
        Returns:
            [band, year, count, prop]: a list containing respectively the migration band (year), the year the data relates to
                                       the number of migratnts and the proportion of population of the country migrating from
        """
        result = self.df[(self.df['code'] == code) & (self.df['From country'] == from_country)].nlargest(1, 'Year')[['migration_band', 'migration_year', 'count', 'prop_emigrants']].values
        result = result[0] if len(result) > 0 else [0, 0, 0, 0]
        return pd.Series(result, index=['migration_band', 'migration_year', 'count', 'prop_emigrants'])

    def __get_population(self, code):
        """Get the most recent population info for a country

        Args:
            code (str): the country code for the country to check
        Returns:
            int: population data for the country for the most recent year found in the historical data frame
        """
        result = self.df[self.df['code'] == code][['Year', 'population']].nlargest(1, 'Year')['population'].values
        return result[0] if len(result) > 0 else 0

    def process(self):
        """Process the data

        Downloads and processes the data for the semi finals in order to generate data for the finals

        Args:
            None
        Returns:
            None

        """
        semi_final_1 = self.__import_participants_from_wiki(2023, 'Semi-final_1')
        semi_final_2 = self.__import_participants_from_wiki(2023, 'Semi-final_2')
        semi_finals = pd.concat([semi_final_1, semi_final_2])

        semi_finals = self.__expand_data(semi_finals)
        semi_finals = self.__transfer_existing_data(semi_finals)

        # Match columns
        semi_finals = semi_finals[['Year', 'From country', 'Votes', 'Country',
                                   'Own', 'English', 'Other', 'has_border',
                                   'migration_band', 'migration_year',
                                   'code', 'population', 'count', 'prop_emigrants',
                                   'Gender', 'comps_since_last_win']]

        # Store the result
        self.result = semi_finals

    def get_result(self):
        """Get the result of all that processing

        As long as processing has been performed, this will return a data frame of data for the
        acts that are likely to appear in the Eurovision final based on their appearance in the
        semi finals.

        Args:
            None
        Returns:
            pd.DataFrame: the processed data
        """
        return self.result

    def __import_existing(self):
        """
        Import the 1998-2022 data set. We need it later do copy some data into the 2023 rows.

        Args:
            None
        Returns:
            None
        """
        # Read in CSV
        self.df = pd.read_csv(self.in_file)
        self.df['has_border'] = self.df['has_border'].fillna(0)
        self.df = self.df.reset_index(drop=True)

    def __load_country_codes(self):
        """Load the country codes pickle

        Args:
            None
        Returns:
            None
        """
        self.country_codes_dict = {}
        # Load the codes pickle so we can convert to proper names
        with open(self.countries, 'rb') as f:
            self.country_codes_dict = pickle.load(f)
            # Reverse the keys and values
            #country_codes_dict = {y: x.title() for x, y in country_codes_dict.items()}
            print('Number of country codes: {}'.format(len(self.country_codes_dict)))

    def __last_win(self, country, year):
        """Returns the number of years since the last win

        Returns the number of years since the last win based on the historical data.

        This does not get updated for the most recent winner.

        Args:
            country (str): the country code for the country to get the info for
            year (int): the current year
        Returns:
            int: the number of years since the country last won
        """
        result = self.df[self.df['code'] == country][['Year', 'comps_since_last_win']].nlargest(1, 'Year').values
        recent_year, duration = result[0] if len(result) > 0 else [0, 0] 
        # We're going to assume year > recent_year to make our lives easier
        assert(year > recent_year)
        return duration + (year - recent_year)

    def __import_participants_from_wiki(self, year, header_id):
        """Download the data from the wiki for a particular semi final

        Downloads artist data from Wikipedia based on the year and the subheading anchor.

        Data is scraped from the first table in the section with the anchor tag or id provided.

        Args:
            year (int): the year to get the data for
            header_id (str): the anchor for the section to scrapte the table data from
        """
        url=f"https://en.wikipedia.org/wiki/Eurovision_Song_Contest_{year}"
        print('Downloading wikipedia page: {}'.format(url))
        response=requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        tables=soup.find_all('table',{'class':"wikitable"})

        table = soup.find(id=header_id).find_all_next('table')[0]

        df_table=pd.read_html(str(table))
        df_table=pd.DataFrame(df_table[0])

        # Remove citations references from the column titles
        df_table.columns = [x.split('[')[0] for x in df_table.columns]

        # These values are the same for every row
        df_table['Year'] = year
        df_table['Votes'] = 0
        df_table['Own'] = 0

        # Country codes
        df_table['code'] = df_table['Country'].apply(lambda x: self.country_codes_dict[x.lower()])

        # Convert the entry to a list of languages and strip any citation references
        df_table['Language(s)'] = df_table['Language(s)'].apply(lambda x: [y.lower().split('[')[0] for y in x.split(', ')])

        # Derive the language entries from the languages list for each country
        print('Deriving language entries')
        df_table['English'] = df_table['Language(s)'].apply(lambda x: 'english' in x)
        df_table['Own'] = df_table.apply(lambda x: self.language_match.contains_own_language(x["code"], x['Language(s)']), axis=1)
        df_table['Other'] = df_table.apply(lambda x: self.language_match.contains_other_language(x["code"], x['Language(s)']), axis=1)

        # Figure out the gender from Wikipedia
        print('Guessing genders')
        df_table['Gender'] = df_table['Artist'].apply(lambda x: self.gender.guess_gender(x))

        # Copy over data from the existing dataset
        print('Transfering population data')
        df_table['population'] = df_table['code'].apply(lambda x: self.__get_population(x))

        print('Calculating last win')
        df_table['comps_since_last_win'] = df_table['code'].apply(lambda x: self.__last_win(x, year))

        return df_table

    def save(self, out_file):
        """Save the result to a CSV file
        """
        print("Writing out results to: {}".format(out_file))
        self.result.to_csv(out_file)

def print_syntax():
	print('Syntax: get_future_performers.py <input-file> <country-pickle> <out-file>')
	print()
	print('\tCollect data about future Eurovision performers')
	print('\t<input-file>     : CSV file containing data for previous years')
	print('\t<country-pickle> : pickle file mapping countries to country codes')
	print('\t<out-file>       : file to save the output CSV to')
	print()
	print('Example usage')
	print('\tget_future_performers.py eurovision_merged_covariates_03Feb.csv country_codes_dict.pickle out.csv')

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("No arguments provided, using defaults")
        in_file = Path(__file__).parent.parent / "data" / "eurovision_merged_covariates_03Feb.csv"
        countries = Path(__file__).parent.parent / "data" / "country_codes_dict.pickle"
        out_file = Path(__file__).parent.parent / "data" / "eurovision_2023.csv"
    else:
        if len(sys.argv) != 4:
            print_syntax()
            exit()
        in_file = sys.argv[1]
        countries = sys.argv[2]
        out_file = sys.argv[3]

    print("Historical data CSV: {}".format(in_file))
    print("Country map: {}".format(countries))
    print("Out file: {}".format(out_file))

    # Initalise
    future = EurovisionFuture(in_file, out_file, countries)

    # Do the work
    future.process()

    # Write out CSV
    future.save(out_file)
    print("Done")

