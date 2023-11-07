import pandas as pd
import geopandas as gpd
from shapely import wkt, contains_properly, box
import rasterio
from tqdm import tqdm


def extract_relevant_raw_data(wtgs: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Extracts relevant data from a DataFrame of wind turbine information and returns
    a filtered GeoDataFrame.

    Parameters:
    - wtgs (gpd.GeoDataFrame): Input DataFrame containing wind turbine data.

    Returns:
    - geopandas.DataFrame: A GeoDataFrame containing a subset of the input data,
      filtered based on confidence levels and with geometry information for turbine
      locations.

    The function extracts the following columns from the input DataFrame:
    - 'case_id': Unique stable identification number (int).
    - 't_rd': Turbine rotor diameter in meters (float).
    - 'xlong': Longitude of the turbine, in decimal degrees (float).
    - 'ylat': Latitude of the turbine, in decimal degrees (float).
    - 't_hh': Turbine hub height in meters (float).
    - 't_ttlh': Turbine total height from ground to tip in meters (float).
    - 't_cap': Turbine rated capacity (int) - output power at rated wind speed.
    - 't_manu': Turbine manufacturer (string) - name of the turbine's OEM.
    - 't_model': Turbine model (string) - manufacturer's model name.
    - 't_conf_atr': Level of confidence in the turbine attributes (int).
    - 't_conf_loc': Level of confidence in turbine location (int).

    The function filters the data to include only rows where 't_conf_atr' and
    't_conf_loc' are both equal to 3. It also removes any rows with missing values.
    The resulting data is converted into a GeoDataFrame with point geometries based
    on 'xlong' and 'ylat' columns, and a coordinate reference system (CRS) of EPSG:4326.
    The filtered GeoDataFrame is returned.

    Example:
    extracted_data = extract_relevant_raw_data(wind_turbine_data)
    """
    turbine_subset = wtgs[
        [
            'case_id',  # int - Unique stable identification number.
            't_rd',  # float - Turbine rotor diameter in meters (m).
            'xlong',  # float - Logitude of the turbine, in decimal degrees.
            'ylat',  # float - Latitude of the turbine, in decimal degrees.
            't_hh',  # float - Turbine hub height in meters (m).
            't_ttlh',  # float - Turbine total height from ground to tip (m).
            't_cap',   # int - Turbine rated capacity - output power at rated wind speed
            't_manu',  # string - Turbine manufacturer - name of the turbine's OEM
            't_model',  # string - Turbine model - manufacturer's model name.
            't_conf_atr',  # int - Level of confidence in the turbine attributes.
            't_conf_loc'  # int - Level of confidence in turbine location.
        ]
    ]
    turbine_subset = turbine_subset[
        (turbine_subset['t_conf_atr'] == 3) & (turbine_subset['t_conf_loc'] == 3)
    ]
    turbine_subset = turbine_subset.dropna()
    turbinesdf = gpd.GeoDataFrame(
        turbine_subset,
        geometry=gpd.points_from_xy(turbine_subset['xlong'], turbine_subset['ylat']),
        crs='EPSG:4326'
    )
    return turbinesdf


def square_poly(lat: float, lon: float, distance: float) -> gpd.GeoSeries:
    """
    Generates a square-shaped bounding box around a specified point, given its
    geographical coordinates (latitude and longitude) and a buffer radius in meters.

    This function first converts the input coordinates to the EPSG:3857 projection to
    create the buffer, and then reverts the result back to EPSG:4326 for the output.
    The resulting polygon represents a square with the specified distance as the side
    length and the center at the provided latitude and longitude.

    Parameters:
        lat (float): Center latitude in geographical coordinates (EPSG:4326).
        lon (float): Center longitude in geographical coordinates (EPSG:4326).
        distance (float): Buffer radius in meters to determine the side length of
                          the square.

    Returns:
        geopandas.geoseries.GeoSeries: A GeoSeries containing the square-shaped polygon
        in geographical coordinates (EPSG:4326).
    """
    gs = gpd.GeoSeries(wkt.loads(f'POINT ({lon} {lat})'))
    gdf = gpd.GeoDataFrame(geometry=gs)
    gdf.crs = 'EPSG:4326'
    gdf = gdf.to_crs('EPSG:3857')
    # Buffer the points using a square cap style
    # Note cap_style: round = 1, flat = 2, square = 3
    res = gdf.buffer(
        distance=distance,
        cap_style=3,
    )
    return res.to_crs('EPSG:4326').iloc[0]


def label_naip_images(files: list,
                      turbinedf: gpd.GeoDataFrame,
                      distance_ratio: float) -> gpd.GeoDataFrame:
    """
    Generate labels for geotiff images based on the provided enhanced US wind turbine
    database.

    This function associates geotiff images with wind turbines from the given database,
    calculates label boxes for the turbines, and enriches the database with
    label-related information.

    Parameters:
        files (list): List of geotiff image file paths to be labeled.
        turbinedf (geopandas.GeoDataFrame): Enhanced US wind turbine database as a
                                            GeoDataFrame.
        distance_ratio (float): A multiplier for calculating the buffer distance for
                                label boxes (buffer = distance_ratio * t_ttlh).

    Returns:
        geopandas.GeoDataFrame: The enhanced turbine database with added label columns.

    The function performs the following steps:
    1. Copy the input database to avoid modifying the original data.
    2. For each geotiff image file in the list:
       a. Open the geotiff raster and retrieve its metadata.
       b. Query the turbines contained within the image's bounding box.
       c. Associate the image file, raster width, and raster height with the turbines
          in the image.
       d. For each turbine in the image:
          i. Calculate the pixel coordinates of the turbine's center.
          ii. Create a label box around the turbine based on the buffer distance.
          iii. Ensure the label box falls within the raster bounds.
          iv. Store the label box and its pixel coordinates in the database.
    3. Remove turbines not contained within any images.
    4. Return the enriched turbine database with added label columns.

    Example usage is provided in the "Data preparation and labeling" notebook.
    """
    gdf = turbinedf.copy()
    for file in tqdm(files):
        raster = rasterio.open(file)
        meta = raster.meta
        turbines_in_image = gdf.sindex.query(box(*raster.bounds), predicate="contains")
        gdf.loc[gdf.iloc[turbines_in_image].index, 'image'] = file
        gdf.loc[gdf.iloc[turbines_in_image].index, 'width'] = raster.width
        gdf.loc[gdf.iloc[turbines_in_image].index, 'height'] = raster.height
        for t in turbines_in_image:
            turbine_row = gdf.iloc[t].name
            turbine = gdf.loc[turbine_row]
            turbine_pixel_coordinates = rasterio.transform.rowcol(meta['transform'],
                                                                  xs=turbine['xlong'],
                                                                  ys=turbine['ylat'])
            gdf.loc[turbine_row, 'cx'] = turbine_pixel_coordinates[1]
            gdf.loc[turbine_row, 'cy'] = turbine_pixel_coordinates[0]
            turbine_box = square_poly(turbine['ylat'],
                                      turbine['xlong'],
                                      distance=distance_ratio*turbine['t_ttlh'])
            if contains_properly(box(*raster.bounds), turbine_box):
                gdf.loc[turbine_row, 'box'] = turbine_box
                minx_miny_coordinates = \
                    rasterio.transform.rowcol(meta['transform'],
                                              xs=turbine_box.bounds[0],
                                              ys=turbine_box.bounds[1])

                maxx_maxy_coordinates = \
                    rasterio.transform.rowcol(meta['transform'],
                                              xs=turbine_box.bounds[2],
                                              ys=turbine_box.bounds[3])
                xs = [minx_miny_coordinates[1], maxx_maxy_coordinates[1]]
                ys = [minx_miny_coordinates[0], maxx_maxy_coordinates[0]]
                gdf.loc[turbine_row, 'minx'] = min(xs)
                gdf.loc[turbine_row, 'miny'] = min(ys)
                gdf.loc[turbine_row, 'maxx'] = max(xs)
                gdf.loc[turbine_row, 'maxy'] = max(ys)
    gdf = gdf.dropna()
    gdf['image_file'] = gdf['image'].str.split('/').apply(lambda x: x[-1])
    return gdf


def label_overlapping_boxes(turbine_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    """
    Identify and label overlapping bounding boxes within a GeoDataFrame containing
    turbine data.

    This function takes a GeoDataFrame with turbine data, where each row represents a
    turbine with a bounding box. It identifies overlapping bounding boxes and assigns
    a label to indicate the presence of overlap.

    The following steps are performed by the code:

    1. Create a copy of the input GeoDataFrame to prevent modifying the original data.
    2. Convert the 'box' column into a GeoSeries to treat the boxes as geometries.
    3. Dissolve overlapping geometries into one large multipolygon.
    4. Split the multipolygon into individual non-overlapping parts.
    5. Set the coordinate reference system (CRS) for the resulting single parts to
       EPSG:4326.
    6. Calculate a 'cluster' column to assign a unique cluster/row number to each
       single part.
    7. Perform a spatial join to associate cluster numbers with the original
       GeoDataFrame.
    8. Merge the original GeoDataFrame with the counts of geometries per cluster.
    9. Remove the 'index_right' column created during the spatial join.
    10. Create an 'overlaps' column indicating whether a turbine's bounding box overlaps
        with others in the same cluster.

    Parameters:
        turbine_df (geopandas.GeoDataFrame): A GeoDataFrame containing turbine data with
                                             bounding boxes.

    Returns:
        geopandas.GeoDataFrame: The input GeoDataFrame enriched with an 'overlaps'
                                column indicating box overlap.
    """
    gdf = turbine_df.copy()
    gdf['geometry'] = gpd.GeoSeries(gdf['box'])
    diss = gdf.dissolve()
    singleparts = gpd.GeoDataFrame(
        geometry=diss.apply(lambda x: [part for part in x.geometry.geoms],
                            axis=1).explode(), crs=gdf.crs)
    singleparts.crs = "EPSG:4326"
    singleparts["cluster"] = range(singleparts.shape[0])
    gdf = gpd.sjoin(left_df=gdf, right_df=singleparts)
    gdf = gdf.merge(gdf.groupby("cluster")["cluster"].count().rename("polycount"),
                    left_on="cluster", right_index=True)
    gdf = gdf.drop(['index_right'], axis=1)
    gdf['overlaps'] = gdf['polycount'] > 1
    return gdf


def remove_overlaped_labels(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Removes overlapping labels in a GeoDataFrame and returns a filtered GeoDataFrame.

    Parameters:
    - gdf (geopandas.GeoDataFrame): Input GeoDataFrame containing labels.

    Returns:
    - geopandas.GeoDataFrame: A filtered GeoDataFrame with overlapping labels removed.

    This function takes a GeoDataFrame as input and removes labels that overlap with
    other labels in the same GeoDataFrame. The 'overlaps' column in the GeoDataFrame
    should indicate whether a label overlaps with others.

    Example:
    filtered_labels = remove_overlapped_labels(label_data)
    """
    return gdf[~gdf['overlaps']]
