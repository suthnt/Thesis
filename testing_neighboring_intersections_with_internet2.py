#!/usr/bin/env python
# coding: utf-8
# This code was written with the assistance of Claude (Anthropic).


# In[1]:


import osmnx as ox, geopandas as gpd, pandas as pd, numpy as np
from shapely.geometry import Point

from osmnx._errors import InsufficientResponseError


# In[2]:


from osmnx._errors import InsufficientResponseError

ox.settings.use_cache = True
ox.settings.timeout = 300           # more forgiving
ox.settings.overpass_rate_limit = True


# In[3]:


def find_intersections(lon, lat, radius_ft=800, network_type="drive"):
    radius_m = float(radius_ft) * 0.3048

    # --- fetch with simple fallback if area is sparse ---
    def grab(lat, lon, dist_m, net):
        try:
            G = ox.graph_from_point((lat, lon), dist=dist_m, network_type=net, simplify=True)
            if G.number_of_edges() == 0:
                raise InsufficientResponseError("empty graph")
            return G
        except InsufficientResponseError:
            # try bigger + 'all'
            G = ox.graph_from_point((lat, lon), dist=dist_m*2.5, network_type="all", simplify=True)
            if G.number_of_edges() == 0:
                raise InsufficientResponseError("still empty after fallback")
            return G

    G = grab(float(lat), float(lon), radius_m, network_type)

    # --- project before consolidating (meters!) ---
    Gp = ox.projection.project_graph(G)
    Gc = ox.consolidate_intersections(Gp, tolerance=10, rebuild_graph=True, dead_ends=False)

    if Gc.number_of_edges() == 0:
        # if consolidation nuked everything, skip consolidation
        Gc = Gp

    # --- nodes & street counts on the same graph ---
    nodes_proj, _ = ox.graph_to_gdfs(Gc)
    sc = ox.stats.count_streets_per_node(Gc)
    nodes_proj["street_count"] = nodes_proj.index.map(sc)

    # --- build projected point, filter by radius (meters) ---
    pt_proj = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(nodes_proj.crs).iloc[0]
    hits = nodes_proj[(nodes_proj["street_count"] >= 3) &
                      (nodes_proj.geometry.distance(pt_proj) <= radius_m)]

    # return lon/lat
    return hits.to_crs(4326)


# In[4]:


csv_path = '/scratch/gpfs/ALAINK/Suthi/NYC_MVC_2024_BikePed.csv'
pts = pd.read_csv(csv_path)   # expects columns: lon, lat
pts = pts.iloc[2290:]


# In[6]:


results = []
for i, r in pts.iterrows():
    try:
        gdf = find_intersections(r["LONGITUDE"], r["LATITUDE"], radius_ft=1200, network_type="drive")
        gdf["point_id"] = i
        gdf["src_lon"] = r["LONGITUDE"]
        gdf["src_lat"] = r["LATITUDE"]
        results.append(gdf)
        print(f"OK row {i}: {len(gdf)} intersections")
    except InsufficientResponseError as e:
        print(f"Overpass returned no data for row {i} (lon={r['LONGITUDE']}, lat={r['LATITUDE']}). Skipping.")
    except Exception as e:
        print(f"Row {i} failed: {e}")

out = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs="EPSG:4326")

# --- 6. Save or view ---
out = out.drop(columns=["src_lon", "src_lat"], errors="ignore")

out.to_file("/scratch/gpfs/ALAINK/Suthi/SafeIntersections2.csv", driver="GeoJSON")
out.head()


# In[ ]:




