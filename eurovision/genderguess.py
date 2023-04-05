"""
Determine the gender of an artist

Utiltieis for determining the gender (group, female, male) of an artist
from the name of the artist, using Wikidata.
"""

import requests
from wikipeople import wikipeople as wp # use wp to get the gender data as needed.

class GenderGuess():
    """Guess the gender of a performer based on entries from Wikidata.
    """
    future = False
    exceptions = {
        "Brunette": "female",
    }

    def __init__(self, future=False):
        print("Initialising gender guesser")
        pass

    def __search_wikidata(self, string):
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
            return None

        target = 0
        for i in range(len(res['search'])):
            if 'description' in res['search'][i]['display']:
                description = res['search'][i]['display']['description']['value']
                if any(markers in description.lower() for markers in music_markers):
                    if self.future:
                        target = i
                        break
                    else:
                        concept_id = res['search'][i]['id']
                        contestant_in = wp.get_property(concept_id, 'P1344')[-1]
                        if "Eurovision" in contestant_in:
                            target = i
                            break

        return res['search'][target]['id']

    def __lookup_gender(self, name):
        """Find gender of given name. If the name is not related to a wiki entry it will return 'RNF' (record not found).
        Alternatively it will return the gender if the record has one or NA if it does not have this property.
        Args:
            name (str): The name to search
        Returns:
            str: The gender of the person searched
        """
        gender = 'RNF'
        data = self.__search_wikidata(name)
        if data:
            gender = wp.get_property(data, 'P21')[-1]
            instance = wp.get_property(data, 'P31')[-1]
            if gender == 'NA':
                group_checks = [
                    "group", "duo", "trio", "music", "band", "ensemble"
                ]
                if any(x in instance for x in group_checks):
                    gender = "group"
        return gender

    def __get_artist_gender(self, search):
        if search in self.exceptions:
            return self.exceptions[search]

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
                gender = self.__lookup_gender(n)
                if not any(gender == x for x in ["RNF", "NA"]):
                    return gender
        if gender == "RNF" or gender == "NA":
            if ("&" in search) or (' and ' in search):
                gender = "group"
        return gender

    def guess_gender(self, artist):
        """Guess the gender of an artist.

        Returns a string representing the gender of the artist.

        Args:
            artist (str): The name of the artist
        Returns:
            str: One of "group", "female", "male" or "RNF"
        """
        gender = self.__get_artist_gender(artist)
        print("Artist: {}, gender: {}".format(artist, gender))
        return gender


