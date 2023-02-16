import threading
import datetime
import math

from osgeo import gdal, ogr, osr
import numpy as np
import pandas as pd

gdal.UseExceptions()

def get_non_forest_area_geometry(cursor, srid3067_geotransform, resolution):
    x_size, y_size = resolution

    tl_coord = (
        srid3067_geotransform[0],
        srid3067_geotransform[3],
    )
    tl_point = ogr.Geometry(ogr.wkbPoint)
    tl_point.AddPoint(*tl_coord)

    tr_coord = (
        srid3067_geotransform[0] + x_size * srid3067_geotransform[1] + 0 * srid3067_geotransform[2],
        srid3067_geotransform[3] + x_size * srid3067_geotransform[4] + 0 * srid3067_geotransform[5],
    )
    tr_point = ogr.Geometry(ogr.wkbPoint)
    tr_point.AddPoint(*tr_coord)

    ll_coord = (
        srid3067_geotransform[0] + 0 * srid3067_geotransform[1] + y_size * srid3067_geotransform[2],
        srid3067_geotransform[3] + 0 * srid3067_geotransform[4] + y_size * srid3067_geotransform[5],
    )
    ll_point = ogr.Geometry(ogr.wkbPoint)
    ll_point.AddPoint(*ll_coord)

    lr_coord = (
        srid3067_geotransform[0] + x_size * srid3067_geotransform[1] + y_size * srid3067_geotransform[2],
        srid3067_geotransform[3] + x_size * srid3067_geotransform[4] + y_size * srid3067_geotransform[5],
    )
    lr_point = ogr.Geometry(ogr.wkbPoint)
    lr_point.AddPoint(*lr_coord)

    fin_srs = osr.SpatialReference()
    fin_srs.ImportFromEPSG(3067)

    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(*tl_coord)
    ring.AddPoint(*tr_coord)
    ring.AddPoint(*lr_coord)
    ring.AddPoint(*ll_coord)
    ring.AddPoint(*tl_coord)

    bbox = ogr.Geometry(ogr.wkbPolygon)
    bbox.AddGeometry(ring)

    bbox_wkt = bbox.ExportToWkt()

    cursor.execute("""
        with enveloped_geometry as (
          select st_setsrid(st_geometryfromtext(%s), 3067) as envelope
        ),
        metsamaski as (
            select
                st_intersection(st_union(geometry), envelope) as geometry, envelope from metsamaski.forestmask3067
            inner join enveloped_geometry on st_intersects(geometry, envelope)
            group by envelope
        ),
        swamp_areas as (
            select
                st_transform(
                    st_difference(st_transform(envelope, 4326), st_intersection(st_union(geom), st_transform(envelope, 4326))),
                    3067
                ) as geometry, envelope from maastotietokanta.suo
            inner join enveloped_geometry on st_intersects(geom, st_transform(envelope, 4326))
            group by envelope
        ),
        non_forest_union as (
            select
                metsamaski.geometry as metsamaski,
                swamp_areas.geometry as swamps
            from metsamaski
            full outer join swamp_areas on metsamaski.envelope = swamp_areas.envelope
        )
        select
            st_astext(st_force2d(st_intersection(metsamaski, swamps))) as geom
        from non_forest_union
    """, (bbox_wkt,))

    ogr_mem_driver = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('GTiff')

    forestmask_geom_wkt = cursor.fetchone()[0]
    forestmask_geom = ogr.CreateGeometryFromWkt(forestmask_geom_wkt, fin_srs)

    thread_id = threading.get_ident()
    output_filename = f'/vsimem/forestmask_{thread_id}.tif'
    # The final parameter 6 means float32 pixel type I think...
    output_ds = driver.Create(output_filename, x_size, y_size, 1, 6)
    output_ds.SetProjection(fin_srs.ExportToWkt())
    output_ds.SetGeoTransform(srid3067_geotransform)

    ogr_vsimem_name = f'/vsimem/{thread_id}_ogr'
    vector_ds = ogr_mem_driver.CreateDataSource(ogr_vsimem_name)
    vector_ds_layer = vector_ds.CreateLayer('vector_layer', srs=fin_srs)
    ogr_feature = ogr.Feature(vector_ds_layer.GetLayerDefn())
    ogr_feature.SetGeometryDirectly(forestmask_geom)
    vector_ds_layer.CreateFeature(ogr_feature)

    gdal.RasterizeLayer(output_ds, [1], vector_ds_layer, burn_values=[1.0])
    output_ds.FlushCache()

    return output_ds

def get_non_forest_area_geometry_arr(cursor, srid3067_geotransform, resolution):
    ds = get_non_forest_area_geometry(cursor, srid3067_geotransform, resolution)
    arr = np.array(ds.GetRasterBand(1).ReadAsArray())
    return arr


# Returns the raster arrays of `num_closest` muuavoinalue areas that are the
# closest to the point from a radarimage specified by `data_take_id` and
# `data_take_start_time`
def get_n_closest_maastotietokanta_muuavoinalue_from_raster(cursor, point, data_take_id, data_take_start_time, num_closest):
    thread_id = threading.get_ident()
    cursor.execute("""
        with closest_features as (
            select
                fid,
                geom,
                st_envelope(geom) as envelope
            from maastotietokanta.muuavoinalue
            order by st_distance(st_setsrid(st_makepoint(%s, %s), 4326), geom) asc limit %s
        ), metadata as (
            select
                -- note that there might be multiple tiff files with the same xml metadata file
                -- That is why this distinct is here (function can return more than 2 bands for the raster in other case)
                distinct on(xml_filename)
                tiff_filename,
                xml_filename as metadata_filename,
                stop_time,
                polarization,
                pass,
                platform_heading,
                data_take_id
            from radarimage_infos
            inner join radarimage_metadata_filemaps on filename = xml_filename
            where data_take_id = %s and start_time = %s
        ), radar_img as (
            select
                st_union(rast) as rast,
                polarization,
                fid,
                geom
            from metadata
            inner join radarimages
                on tiff_filename = radarimages.filename
            inner join closest_features
                on st_intersects(rast::geometry, closest_features.envelope)
            group by fid, polarization, geom
        ), clipped as (
            select
                st_clip(rast, geom) as rast,
                polarization,
                fid
            from radar_img
        )
        select
            fid,
            st_asgdalraster(
                st_addband(null::raster, array_agg(rast order by polarization desc)),
                'gtiff'
            ) as rast
        from clipped
        group by fid
    """, (point[0], point[1], num_closest, data_take_id, data_take_start_time))
    image_rows = pd.DataFrame([i.copy() for i in cursor])

    if len(image_rows) == 0:
        return None

    images = []

    for i, r in image_rows.iterrows():
        vsimem_name = f"/vsimem/{thread_id}_closest_feature_image{i}.tif"
        gdal.FileFromMemBuffer(vsimem_name, r['rast'].tobytes())
        gdal_file = gdal.Open(vsimem_name)

        arr = np.array(
            [
                gdal_file.GetRasterBand(1).ReadAsArray(),
                gdal_file.GetRasterBand(2).ReadAsArray(),
            ]
        )
        arr[arr == -9999.0] = None
        images.append(arr)

    return images


def get_maastotietokanta_non_forest_mask_arr(cursor, srid3067_geotransform, resolution):
    x_size, y_size = resolution

    tl_coord = (
        srid3067_geotransform[0],
        srid3067_geotransform[3],
    )
    tl_point = ogr.Geometry(ogr.wkbPoint)
    tl_point.AddPoint(*tl_coord)

    tr_coord = (
        srid3067_geotransform[0] + x_size * srid3067_geotransform[1] + 0 * srid3067_geotransform[2],
        srid3067_geotransform[3] + x_size * srid3067_geotransform[4] + 0 * srid3067_geotransform[5],
    )
    tr_point = ogr.Geometry(ogr.wkbPoint)
    tr_point.AddPoint(*tr_coord)

    ll_coord = (
        srid3067_geotransform[0] + 0 * srid3067_geotransform[1] + y_size * srid3067_geotransform[2],
        srid3067_geotransform[3] + 0 * srid3067_geotransform[4] + y_size * srid3067_geotransform[5],
    )
    ll_point = ogr.Geometry(ogr.wkbPoint)
    ll_point.AddPoint(*ll_coord)

    lr_coord = (
        srid3067_geotransform[0] + x_size * srid3067_geotransform[1] + y_size * srid3067_geotransform[2],
        srid3067_geotransform[3] + x_size * srid3067_geotransform[4] + y_size * srid3067_geotransform[5],
    )
    lr_point = ogr.Geometry(ogr.wkbPoint)
    lr_point.AddPoint(*lr_coord)

    fin_srs = osr.SpatialReference()
    fin_srs.ImportFromEPSG(3067)

    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(*tl_coord)
    ring.AddPoint(*tr_coord)
    ring.AddPoint(*lr_coord)
    ring.AddPoint(*ll_coord)
    ring.AddPoint(*tl_coord)

    bbox = ogr.Geometry(ogr.wkbPolygon)
    bbox.AddGeometry(ring)

    bbox_wkt = bbox.ExportToWkt()

    thread_id = threading.get_ident()
    cursor.execute("""
        with enveloped_geometry as (
          select st_transform(st_setsrid(st_geometryfromtext(%s), 3067), 4326) as envelope
        ),
        suo as (
            select st_intersection(st_union(geom), envelope) as geom from maastotietokanta.suo
            inner join enveloped_geometry on st_intersects(geom, envelope)
            group by envelope
        ),
        maatuvavesialue as (
            select st_intersection(st_union(geom), envelope) as geom from maastotietokanta.maatuvavesialue
            inner join enveloped_geometry on st_intersects(geom, envelope)
            group by envelope
        ),
        maatalousmaa as (
            select st_intersection(st_union(geom), envelope) as geom from maastotietokanta.maatalousmaa
            inner join enveloped_geometry on st_intersects(geom, envelope)
            group by envelope
        ),
        niitty as (
            select st_intersection(st_union(geom), envelope) as geom from maastotietokanta.niitty
            inner join enveloped_geometry on st_intersects(geom, envelope)
            group by envelope
        ),
        jarvi as (
            select st_intersection(st_union(geom), envelope) as geom from maastotietokanta.jarvi
            inner join enveloped_geometry on st_intersects(geom, envelope)
            group by envelope
        ),
        soistuma as (
            select st_intersection(st_union(geom), envelope) as geom from maastotietokanta.soistuma
            inner join enveloped_geometry on st_intersects(geom, envelope)
            group by envelope
        ),
        muuavoinalue as (
            select st_intersection(st_union(geom), envelope) as geom from maastotietokanta.muuavoinalue
            inner join enveloped_geometry on st_intersects(geom, envelope)
            group by envelope
        ),
        virtavesialue as (
            select st_intersection(st_union(geom), envelope) as geom from maastotietokanta.virtavesialue
            inner join enveloped_geometry on st_intersects(geom, envelope)
            group by envelope
        ),
        union_query as (
            select geom from suo
            union
            select geom from maatuvavesialue
            union
            select geom from maatalousmaa
            union
            select geom from niitty
            union
            select geom from jarvi
            union
            select geom from soistuma
            union
            select geom from muuavoinalue
            union
            select geom from virtavesialue
        )
        select st_astext(st_transform(st_union(geom), 3067)) from union_query
    """, (bbox_wkt,))

    ogr_mem_driver = ogr.GetDriverByName('Memory')
    driver = gdal.GetDriverByName('GTiff')

    forestmask_geom_wkt = cursor.fetchone()[0]
    forestmask_geom = ogr.CreateGeometryFromWkt(forestmask_geom_wkt, fin_srs)

    output_filename = f'/vsimem/forestmask_{thread_id}.tif'
    # The final parameter 6 means float32 pixel type I think...
    output_ds = driver.Create(output_filename, x_size, y_size, 1, 6)
    output_ds.SetProjection(fin_srs.ExportToWkt())
    output_ds.SetGeoTransform(srid3067_geotransform)

    ogr_vsimem_name = f'/vsimem/{thread_id}_ogr'
    vector_ds = ogr_mem_driver.CreateDataSource(ogr_vsimem_name)
    vector_ds_layer = vector_ds.CreateLayer('vector_layer', srs=fin_srs)
    ogr_feature = ogr.Feature(vector_ds_layer.GetLayerDefn())
    ogr_feature.SetGeometryDirectly(forestmask_geom)
    vector_ds_layer.CreateFeature(ogr_feature)

    gdal.RasterizeLayer(output_ds, [1], vector_ds_layer, burn_values=[1.0])
    output_ds.FlushCache()

    return np.array(output_ds.GetRasterBand(1).ReadAsArray())
