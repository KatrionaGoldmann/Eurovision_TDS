import numpy as np
import asyncio
import aiohttp


async def get_property(session, concept_id, property_id):
    """Async reimplementation of wikipeople.get_property
    https://github.com/samvanstroud/wikipeople/blob/master/wikipeople/wikipeople.py
    """
    url = 'https://www.wikidata.org/w/api.php'
    params = {'action': 'wbgetclaims',
              'entity': concept_id,
              'property': property_id,
              'language': 'en',
              'format': 'json'}
    async with session.get(url, params=params) as resp:
        res = await resp.json()

    if property_id not in res['claims']:
        return None
    # This gives yet another 'id', and we then need to perform yet another HTTP
    # request to find the actual *label* that this corresponds to.
    else:
        id = None
        for prop in res['claims'][property_id]:
            try:
                id = prop['mainsnak']['datavalue']['value']['id']
            except:
                continue

        if id is None:
            return None
        else:
            new_params =  {'action': 'wbgetentities',
                           'ids': id,
                           'languages': 'en',
                           'format': 'json',
                           'props': 'labels'}
            async with session.get(url, params=new_params) as resp:
                res = await resp.json()
            try:
                return res['entities'][id]['labels']['en']['value']
            except:
                return None


async def get_concept_id(session, page_name):
    """
    Query the Wikidata API using the wbsearchentities function. Return the
    concept ID of the search result that has the musician identifier.
    """
    url = 'https://www.wikidata.org/w/api.php'
    params = {'action': 'wbsearchentities',
              'search': page_name,
              'language': 'en',
              'format': 'json'}
    music_markers = [
        'singer', 'artist', 'musician', 'music',
        'band', 'group', 'duo', 'ensemble'
    ]

    async with session.get(url, params=params) as resp:
        # Titles of WP pages that match the search query.
        json = await resp.json()

    result = json['search']

    if len(result) == 0:
        # Couldn't find a concept id for the person/group
        return None

    # By default, choose the first result from the list
    target = 0
    # But check the other results to see if any of them are musicians (as
    # indicated by the markers) and Eurovision contestants
    for i, res in enumerate(result):
        if 'description' in res['display']:
            description = res['display']['description']['value']
            if any(markers in description for markers in music_markers):
                concept_id = res['id']
                contestant_in = await get_property(session, concept_id, 'P1344')
                if contestant_in is not None and "Eurovision" in contestant_in:
                    target = i
    # Return the concept ID of the result found
    return result[target]['id']


async def lookup_gender(session, page_name):
    """Find gender of a performing act, using the name associated with their
    Wikipedia page. Returns None if could not be found."""
    concept_id = await get_concept_id(session, page_name)
    if concept_id is None:
        return None

    gender = await get_property(session, concept_id, 'P21')
    instance = await get_property(session, concept_id, 'P31')

    if gender is None and instance is None:
        # Really failed. Last chance: check for '&' in the name
        return 'group' if '&' in page_name else None
    elif gender is None and instance is not None:
        group_checks = ["group", "duo", "trio", "music", "band", "ensemble"]
        if any(x in instance for x in group_checks):
            return "group"
    else:
        return gender


async def get_pages(session, name):
    """Obtain a list of Wikipedia pages obtained by searching for a name."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "opensearch",
        "namespace": "0",
        "search": name,
        "limit": "10000",
        "format": "json"
    }
    async with session.get(url, params=params) as resp:
        # Titles of WP pages that match the search query.
        json = await resp.json()
    return json[1]


async def get_artist_gender(session, name):
    pages = await get_pages(session, name)
    if len(pages) > 0:
        gender = await lookup_gender(session, pages[0])
        return gender
    else:
        return None


def final_fixes(df):
    # group remaining categories accordingly
    idx = np.where(
        (df['Gender'] == "genderfluid") |
        (df['Gender'] == "non-binary")
    )
    df.loc[idx[0], "Gender"] = "non-binary"
    idx = np.where(
        df['Gender'] == "trans woman"
    )
    df.loc[idx[0], "Gender"] = "female"

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
    df.loc[df["Artist"] == "Prime Minister", "Gender"] = "group"
    
    # sanity check 
    assert len(df.loc[df["Gender"] == "NA"]) == 0, "NA genders > 0"
    assert len(df.loc[df["Gender"] == "RNF"]) == 0, "RNF genders > 0"
    return df

async def main():
    performers = [
        'Ronela Hajati', 'Rosa Linn', 'Sheldon Riley',
        'LUM!X feat. Pia Maria', 'Nadir Rustamli', 'Jérémie Makiese',
        'Intelligent Music Project', 'Mia Dimšić', 'Andromache',
        'We Are Domi', 'REDDI', 'Stefan',
        'The Rasmus', 'Alvan & Ahez',
        'Circus Mircus', 'Malik Harris', 'Amanda Georgiadi Tenfjord',
        'Systur', 'Brooke', 'Michael Ben David', 'Mahmood & BLANCO',
        'Citi Zēni', 'Monika Liu', 'Emma Muscat',
        'Zdob şi Zdub & Advahov Brothers', 'Vladana', 'Andrea',
        'Subwoolfer', 'Ochman', 'MARO', 'WRS', 'Achille Lauro',
        'Konstrakta', 'LPS', 'Chanel', 'Cornelia Jakobs', 'Marius Bear',
        'S10', 'Kalush Orchestra', 'Sam Ryder'
    ]
    async with aiohttp.ClientSession() as session:
        tasks = asyncio.gather(*[get_artist_gender(session, p) for p in performers])
        genders = await tasks
    print(dict(zip(performers, genders)))

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
