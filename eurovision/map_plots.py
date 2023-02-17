import plotly
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import pycountry
from pathlib import Path
from copy import deepcopy

# For debugging
pd.set_option('display.max_columns', None)

# Read in data
df = pd.read_csv(Path(__file__).parents[1] / "data" / "all_covariates.csv")
df = df.drop_duplicates(subset=["code", "Year"])
df = df[df["Year"] == 2017]
df = df.reset_index(drop=True)

# convert iso3166 alpha-3 codes to alpha-2
def alpha3to2(a3):
    cty = pycountry.countries.get(alpha_3=a3)
    if cty is None:
        return None
    else:
        return cty.alpha_2

# All countries JSON
with open(Path(__file__).parents[1] / "data" / "countries.geojson") as file:
    countries = json.load(file)
    # Add in the alpha-2 codes as the ids, which are used when plotting.
    for f in countries["features"]:
        f["id"] = alpha3to2(f["properties"]["ISO_A3"])
    # Remove stuff that doesn't correspond to a valid country.
    countries["features"] = [f for f in countries["features"]
                             if f["id"] is not None]

# Europe and Israel-only JSON
europe = deepcopy(countries)
europe["features"] = [f for f in europe["features"]
                      if f["id"] != "AU"]

# Australia-only JSON
australia = deepcopy(countries)
australia["features"] = [f for f in australia["features"]
                         if f["id"] == "AU"]

# Europe
fig1 = px.choropleth(df, geojson=europe,
                     locations="code",
                     color="population")
fig1_geo_kwargs = {
    "resolution": 50,
    "scope": "europe",
    "showcountries": True,
    "countrycolor": "Black",
}

# Australia
fig2 = px.choropleth(df, geojson=australia,
                     locations="code",
                     color="population")
fig2_geo_kwargs = {
    "scope": "world",
    "fitbounds": "locations"
}

layout_kwargs = {
    "title": "Population in 2017",
    "margin": {"l": 0, "t": 40, "b": 0, "r": 0},
    "dragmode": False,
}

# 1 for fig1 only, 2 for fig2 only, 3 for both
plot_type = 2

if plot_type == 1:
    fig1.update_geos(**fig1_geo_kwargs)
    fig1.update_layout(**layout_kwargs)
    fig1.show()
elif plot_type == 2:
    fig2.update_geos(**fig2_geo_kwargs)
    fig2.update_layout(**layout_kwargs)
    fig2.show()
elif plot_type == 3:
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{'type': 'choropleth'},
                                {'type': 'choropleth'}]],
                        column_widths=[0.6, 0.4],
                        horizontal_spacing=0,
                        )
    # Plot Europe
    fig.add_trace(fig1['data'][0], row=1, col=1)
    fig.update_geos(**fig1_geo_kwargs, row=1, col=1)
    # Plot Australia
    fig.add_trace(fig2['data'][0], row=1, col=2)
    fig.update_geos(**fig2_geo_kwargs, row=1, col=2)
    fig.update_layout(**layout_kwargs)
    fig.show()
