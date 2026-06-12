# BayFlood

BayFlood is an interactive Bay Area flood-exposure screening tool. Users can
search for an address or click the map to receive a score based on FEMA flood
zones, elevation, nearby floodplain distance, and extreme rainfall.

The result also reports data confidence, the nearest mapped water source, an
explanation of the main inputs, and basic preparedness guidance.

> BayFlood is a screening and educational tool. It is not a parcel-level flood
> determination, engineering study, insurance quote, or prediction that a
> property will or will not flood.

## Features

- Search for Bay Area addresses.
- Select locations directly from an interactive map.
- Display FEMA flood-hazard zones.
- Sample local USGS elevation rasters.
- Sample NOAA Atlas 14 100-year, 24-hour rainfall.
- Measure distance to a FEMA Special Flood Hazard Area.
- Find the nearest mapped water source within 5 kilometers.
- Calculate a flood-exposure score from 0 to 100.
- Classify results as Low, Moderate, High, or Very high.
- Report High, Medium, or Low data confidence.
- Provide risk-level preparedness feedback.

## Technology

### Application

- [Next.js](https://nextjs.org/) 16 with the App Router
- [React](https://react.dev/) 19
- [TypeScript](https://www.typescriptlang.org/)
- [MapLibre GL JS](https://maplibre.org/maplibre-gl-js/docs/) for the map
- [PMTiles](https://docs.protomaps.com/pmtiles/) for browser-friendly FEMA
  vector tiles
- [GeoTIFF.js](https://geotiffjs.github.io/) for elevation and rainfall raster
  sampling

### Data and services

- FEMA National Flood Hazard Layer data for mapped flood zones
- USGS elevation GeoTIFF tiles
- NOAA Atlas 14 precipitation-frequency data
- OpenStreetMap raster map tiles
- Photon for address geocoding
- OpenStreetMap Overpass API for nearby water features

## How It Works

```text
Address search or map click
             |
             v
 Latitude and longitude
             |
             v
 /api/flood-risk
             |
             +--> USGS elevation raster
             +--> FEMA flood-zone GeoJSON
             +--> NOAA rainfall raster
             +--> OpenStreetMap nearby water
             |
             v
 Score + confidence + explanation + feedback
```

The flood-risk API loads the independent data sources in parallel. Elevation,
FEMA, and rainfall are sent to the scoring function. Nearby water is currently
displayed as additional context and does not affect the numerical score.

## Risk Score

The score is a hand-tuned screening index:

```text
score =
  elevation contribution
  + FEMA contribution
  + nearby floodplain contribution
  + rainfall contribution
```

The final value is capped at 100.

### Risk levels

| Score | Level |
| ---: | --- |
| 0-27 | Low |
| 28-49 | Moderate |
| 50-74 | High |
| 75-100 | Very high |

### Elevation contribution

| Elevation | Points |
| --- | ---: |
| 0 m or lower | 42 |
| Greater than 0 m through 3 m | 38 |
| Greater than 3 m through 7 m | 32 |
| Greater than 7 m through 15 m | 23 |
| Greater than 15 m through 30 m | 14 |
| Greater than 30 m through 60 m | 7 |
| Greater than 60 m through 120 m | 3 |
| Greater than 120 m | 0 |

The elevation sampler reads a 5-by-5 pixel window and uses its median valid
value. This reduces the effect of a single unusual raster pixel.

### FEMA contribution

| FEMA classification | Points |
| --- | ---: |
| V or VE coastal high-hazard zone | 58 |
| A-family zone or Special Flood Hazard Area | 52 |
| 0.2% annual-chance or moderate hazard | 31 |
| X | 10 |
| D | 8 |
| Other or unavailable classification | 15 |

If overlapping polygons contain the selected point, the lookup prefers the
feature with the higher hazard priority.

### Nearby floodplain contribution

For locations outside an SFHA, the application measures the distance to the
nearest SFHA boundary:

| Distance | Base points |
| --- | ---: |
| 75 m or closer | 30 |
| 76-200 m | 23 |
| 201-500 m | 14 |
| 501-1,000 m | 6 |
| More than 1,000 m | 0 |

These points are multiplied by an elevation factor:

- Elevation at or below 15 m: `1.0`
- Elevation at or below 30 m: `0.75`
- Higher elevation: `0.4`

### Rainfall contribution

The rainfall input is NOAA Atlas 14's estimated precipitation for a 100-year,
24-hour storm. Rainfall receives 2 to 12 base points, then an elevation-based
terrain multiplier from `1.0` on low terrain to `0.25` above 120 meters.

If rainfall data is unavailable, its contribution is zero.

## Data Confidence

Confidence describes data completeness, not certainty about whether flooding
will happen.

- **High:** elevation, FEMA, and rainfall data are available
- **Medium:** elevation and one additional source are available
- **Low:** only elevation is available

Elevation is required for a result. A request outside the installed elevation
coverage returns an error instead of a score.

## Nearby Water

BayFlood queries OpenStreetMap's Overpass API for mapped water features within
5 kilometers, including:

- Rivers
- Streams
- Canals
- Drainage channels
- Lakes and other mapped water bodies
- Coastline

The result displays the nearest source's name when one is available, its type,
and its approximate distance from the selected point.

Nearby-water results depend on OpenStreetMap coverage and Overpass
availability. They are informational and currently do not change the score.

## Project Structure

```text
src/
  app/
    api/
      flood-risk/route.ts  Combines data and returns an assessment
      geocode/route.ts     Searches addresses with Photon
    globals.css            Application styling
    layout.tsx             Metadata and root layout
    page.tsx               Map, search, and result interface
  lib/
    elevation.ts           Samples USGS elevation GeoTIFFs
    fema.ts                Looks up FEMA zones and SFHA distance
    flood-risk.ts          Calculates score, level, confidence, and feedback
    rainfall.ts            Samples the NOAA rainfall GeoTIFF
    water.ts               Queries and measures nearby mapped water

FEMAdata/                  Local FEMA county GeoJSON files
elevation/                 Local USGS elevation GeoTIFF files
rainfall/                  NOAA rainfall GeoTIFF
public/data/               Browser-served FEMA PMTiles archive
```

## Local Setup

### Requirements

- Node.js 20 or newer
- npm
- Internet access for map tiles, address search, and nearby-water queries
- Required local geospatial files

### Install dependencies

```bash
npm install
```

### Add the data files

The application expects these paths:

```text
FEMAdata/
  alameda_county_flood_zones.geojson
  contra_costa_county_flood_zones.geojson
  marin_county.geojson
  napa_county.geojson
  san_francisco_county_flood_zones.geojson
  san_mateo_county_flood_zones.geojson
  santa_clara_flood_zones.geojson
  solano_county.geojson
  sonoma_county_flood_zones.geojson

elevation/
  *.tif or *.tiff

rainfall/
  rainfall_data.tif

public/data/
  bay-area-flood-zones.pmtiles
```

`FEMAdata/` and `elevation/` are ignored by Git because these source datasets
are large. They must be installed separately after cloning the repository.

### Optional geocoder configuration

Photon is used by default:

```text
https://photon.komoot.io/api
```

To use another compatible Photon endpoint, create `.env.local`:

```bash
PHOTON_GEOCODER_URL=https://your-photon-server.example/api
```

### Start development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Using BayFlood

1. Enter at least three characters of a Bay Area address.
2. Select an address from the search results.
3. Alternatively, click any covered point on the map.
4. Wait while BayFlood reads elevation, FEMA, rainfall, and nearby-water data.
5. Review the score, level, confidence, mapped zone, rainfall, water source,
   explanation, and preparedness feedback.

Map colors:

- Dark red: V or VE coastal high-hazard zones
- Red: A-family high-risk zones
- Orange: shaded X or moderate-hazard areas
- Blue: other mapped flood zones

## API

### Flood-risk assessment

```http
GET /api/flood-risk?lat=37.7749&lng=-122.4194
```

The response includes:

- Coordinates and elevation
- FEMA zone and distance to an SFHA
- Rainfall estimate
- Nearby-water name, type, and distance
- Score and risk level
- Individual score contributions
- Data confidence
- Explanation and preparedness feedback

### Address search

```http
GET /api/geocode?q=San%20Francisco
```

The search is limited to the configured Bay Area bounding box and returns up to
five Photon results.

## Commands

```bash
npm run dev      # Start the development server
npm run lint     # Run ESLint
npm run build    # Create and type-check a production build
npm run start    # Start the production server after building
```

Before committing a change, run:

```bash
npm run lint
npm run build
```

## Limitations

BayFlood currently does not model:

- Parcel or building-footprint elevation
- Local slopes, depressions, or downhill flow paths
- Storm-drain capacity and maintenance
- Impervious surface coverage
- Soil infiltration
- Levees, pumps, culverts, or detention basins
- Building foundations, basements, or raised utilities
- Live weather or current flood warnings
- Tides, storm surge, wave action, or sea-level rise
- Flood depth, velocity, damage, or annual loss
- Historical claims or observed-flood validation

The score uses hand-selected thresholds and is not a calibrated probability.
Two properties with similar scores may have different flood mechanisms and
real-world outcomes.

## Production Notes

- The Node.js runtime must have access to the local FEMA, elevation, and
  rainfall files.
- Serverless deployments may have file-size and bundle-size limits that make
  the local GeoJSON and GeoTIFF approach unsuitable.
- The public Overpass API can be slow or rate-limited. A production deployment
  should consider cached or locally hosted water data.
- OpenStreetMap's public tile service has usage requirements. Review its tile
  policy before serving substantial production traffic.
- FEMA lookups currently scan county GeoJSON features in application memory.
  A spatial database or indexed vector format would scale better.

## Responsible Use

Do not use BayFlood as the only basis for purchasing property, setting
insurance coverage, planning construction, or making emergency decisions.
Confirm important decisions with FEMA products, local flood-control agencies,
licensed engineers, insurers, and official emergency information.
