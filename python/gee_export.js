// Google Earth Engine: per-grid Sentinel-2 NDVI/NDRE/NDMI masked to Global Mangrove Watch 2020 extent.
// Grid matches index.html (LAT0, LON0, DG). Asset path is from the GEE community catalog; check you can access it.
var LAT0=24.523, LON0=54.420, DG=0.002, cells=[];
'ABCD'.split('').forEach(function(L,r){for(var c=0;c<6;c++)cells.push(ee.Feature(ee.Geometry.Rectangle([LON0+c*DG,LAT0-(r+1)*DG,LON0+(c+1)*DG,LAT0-r*DG]),{grid_id:L+(c+1)}))});
var grid=ee.FeatureCollection(cells);
var gmw=ee.ImageCollection('projects/sat-io/open-datasets/GMW/annual-extent/GMW_MNG_2020').mosaic().gt(0);
function idx(a,b){var s=ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(grid).filterDate(a,b).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20)).median();
  return s.normalizedDifference(['B8','B4']).rename('ndvi').addBands(s.normalizedDifference(['B8','B5']).rename('ndre')).addBands(s.normalizedDifference(['B8','B11']).rename('ndmi')).updateMask(gmw)}
var prev=idx('2025-10-01','2025-11-30'), cur=idx('2026-10-01','2026-11-15');
var out=cur.addBands(cur.select('ndvi').subtract(prev.select('ndvi')).rename('ndvi_change')).reduceRegions({collection:grid,reducer:ee.Reducer.mean(),scale:10});
Export.table.toDrive({collection:out,description:'mangrove_grid_indices',fileFormat:'CSV'});
