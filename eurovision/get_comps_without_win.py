""" get_comps_without_win.py

This script counts the time since a given country last won Eurovision. Strictly
speaking, it shows the number of competitions the country has participated in
without winning: this makes it a well-defined quantity even for countries which
have never won Eurovision before. Thus, for example, going into this year's
competition:

- Ukraine won last year, so they have participated 0 times without winning.
- The UK last won in 1997, so going into this year's competition, they have
  participated in 24 competitions without winning. (This includes the 1998
  through 2019 competitions, plus 2021 and 2022; the 2020 competition was
  cancelled.)
- Australia has never won Eurovision before, so they've gone 66
  competitions without winning. (This year is the 67th edition.)

Technically our data only ranges from 1999 to 2019, so the 2020 cancellation
doesn't actually matter. I coded it in anyway, but it shouldn't affect the
output.
"""

import pandas as pd
import pickle

# copy paste from Wikipedia.
winners = [(1956, "Switzerland"), (1957, "Netherlands"), (1958, "France"),
           (1959, "Netherlands"), (1960, "France"), (1961, "Luxembourg"),
           (1962, "France"), (1963, "Denmark"), (1964, "Italy"),
           (1965, "Luxembourg"), (1966, "Austria"), (1967, "United Kingdom"),
           (1968, "Spain"), (1969, "Spain"), (1969, "United Kingdom"),
           (1969, "Netherlands"), (1969, "France"), (1970, "Ireland"),
           (1971, "Monaco"), (1972, "Luxembourg"), (1973, "Luxembourg"),
           (1974, "Sweden"), (1975, "Netherlands"), (1976, "United Kingdom"),
           (1977, "France"), (1978, "Israel"), (1979, "Israel"),
           (1980, "Ireland"), (1981, "United Kingdom"), (1982, "Germany"),
           (1983, "Luxembourg"), (1984, "Sweden"), (1985, "Norway"),
           (1986, "Belgium"), (1987, "Ireland"), (1988, "Switzerland"),
           (1989, "Yugoslavia"), (1990, "Italy"), (1991, "Sweden"),
           (1992, "Ireland"), (1993, "Ireland"), (1994, "Ireland"),
           (1995, "Norway"), (1996, "Ireland"), (1997, "United Kingdom"),
           (1998, "Israel"), (1999, "Sweden"), (2000, "Denmark"),
           (2001, "Estonia"), (2002, "Latvia"), (2003, "Turkey"),
           (2004, "Ukraine"), (2005, "Greece"), (2006, "Finland"),
           (2007, "Serbia"), (2008, "Russia"), (2009, "Norway"),
           (2010, "Germany"), (2011, "Azerbaijan"), (2012, "Sweden"),
           (2013, "Denmark"), (2014, "Austria"), (2015, "Sweden"),
           (2016, "Ukraine"), (2017, "Portugal"), (2018, "Israel"),
           (2019, "Netherlands"), (2021, "Italy"), (2022, "Ukraine")]

def update_data(input_file, output_file, country_codes_pickled_file):
    """
    Reads in CSV format data from `input_file`, adds an extra column
    named `comps_since_last_win`, and stores it in `output_file`.

    Requires a pickled dictionary of country codes.

    Assumes that the CSV data read in contains a column 'code' (representing
    the 2-letter code of the participating country) and a column 'year'
    (representing the year of the competition).
    """
    # Read in country codes. Manually add Lux
    with open(country_codes_pickled_file, "rb") as f:
      country_codes = pickle.load(f)
    country_codes["luxembourg"] = "LU"

    # Construct a dictionary of all wins, by country.
    all_wins = {}
    for y, c in winners:
      code = country_codes[c.lower()]
      if code in all_wins:
        all_wins[code].append(y)
      else:
        all_wins[code] = [y]
    # At this point, we have all_wins = {
    #   'CH': [1956, 1988],
    #   'NL': [1957, 1959, 1969, 1975, 2019],
    #   'FR': [1958, 1960, 1962, 1969, 1977],
    #   ...,
    # }

    def comps_since_last_win(code, year):
      # Find last win. Use 1955 (year before ESC started) if there isn't one.
      if code not in all_wins:
        last_win = 1955
      else:
         last_win = max([y for y in all_wins[code] if y < year],
                        default=1955)
      # Count the number of competitions since the last win. Note that the 2020
      # contest was cancelled.
      comps = year - last_win - 1
      if year > 2020 and last_win < 2020:
        comps = comps - 1
      return comps

    # Some quick tests.
    assert(comps_since_last_win("UA", 2023) == 0)   # won in 2022
    assert(comps_since_last_win("GB", 2023) == 24)  # won in 1997
    assert(comps_since_last_win("AU", 2023) == 66)  # never won
    assert(comps_since_last_win("SE", 1983) == 8)   # won in 1974
    assert(comps_since_last_win("SE", 2019) == 3)   # won in 2015
    assert(comps_since_last_win("NL", 2019) == 43)  # won in 1975

    df = pd.read_csv(input_file)
    df["comps_since_last_win"] = df.apply(lambda r: comps_since_last_win(r.code, r.Year), axis=1)
    df.to_csv(output_file)


if __name__ == "__main__":
    from pathlib import Path
    in_file = Path(__file__).parent / "data" / "eurovision-migration-covariates-gender-merged.csv"
    out_file = Path(__file__).parent / "data" / "eurovision_merged_covariates_03Feb.csv"
    countries = Path(__file__).parent / "data" / "countries.pickle"
    # don't run if the files aren't present
    for file in [in_file, countries]:
        if not file.is_file():
            raise FileNotFoundError(f"file {str(file)} was not found")

    update_data(input_file=str(in_file),
                output_file=str(out_file),
                country_codes_pickled_file=str(countries))
