# BayFlood

BayFlood is a Bay Area flood-exposure screening and education tool. Users can
search for an address or click the map to see FEMA's mapped 100-year event
benchmark when available or an experimental physical-only estimate when the
address has no numeric FEMA classification.

The result also reports historical NFIP claim activity for the surrounding
Census block group, a separate claim-location similarity score, elevation,
rainfall, nearby water and floodplain context, data confidence, and basic
preparedness guidance.

> BayFlood is a screening and educational tool. It is not a parcel-level flood
> determination, engineering study, insurance quote, or prediction that a
> property will or will not flood.

## Property-level hazard dataset

Run `npm run hazard:prepare` to validate parcel observations, sample geospatial
features, build annual exposure rows, and generate a calibration-readiness
report. See [the property hazard model guide](docs/property-hazard-model.md) for
the evidence requirements and probability interpretation.

## Terrain feature preprocessing

After creating `terrain/dem-10m.tif` and installing the Whitebox Python package
inside `.venv-whitebox`, generate the additional physical-model rasters with:

```sh
npm run terrain:features
```

The command is resumable and creates 250 m and 500 m local-elevation rasters, a
hydrologically corrected DEM, D8 flow accumulation, a derived stream network,
and height above drainage. The Bay Area mosaic contains roughly 407 million
cells, so this step can run for a long time and create several gigabytes of
temporary and final raster data.

After the 10 m source rasters finish, build the deployment-oriented terrain
package:

```sh
npm run terrain:runtime
```

This creates a 30 m, five-band, ZSTD-compressed Cloud-Optimized GeoTIFF at
`terrain/runtime/terrain-features-30m.tif`. The current file is approximately
432 MiB instead of roughly 7.3 GiB for the five 10 m feature rasters. Local
development uses this COG automatically. A deployed server can read the same
artifact from public object storage by setting:

```text
TERRAIN_COG_URL=https://your-object-host/terrain-features-30m.tif
```

The object host must support HTTP byte-range requests. The original 10 m
rasters remain the reproducible source data and are not deleted.

## Features

- Search for Bay Area addresses.
- Select locations directly from an interactive map.
- Display FEMA flood-hazard zones.
- Sample local USGS elevation rasters.
- Sample NOAA Atlas 14 100-year, 24-hour rainfall.
- Measure distance to a FEMA Special Flood Hazard Area.
- Find the nearest mapped water source within 5 kilometers.
- Display a cumulative 100-year FEMA event benchmark when the mapped zone
  defines a numeric annual chance.
- Estimate a 100-year percentage from physical conditions when FEMA supplies no
  numeric annual-chance classification for the address.
- Display historical NFIP claim activity for the surrounding Census block
  group.
- Report the trained claim-location similarity model separately as a score out
  of 100.
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
- [Proj4js](https://proj4js.org/) for longitude/latitude to California Albers
  terrain-grid conversion

### Data and services

- FEMA National Flood Hazard Layer data for mapped flood zones
- OpenFEMA NFIP claims and policies for the neighborhood historical signal
- U.S. Census coordinate geography service for block-group matching
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
             +--> 10 m terrain-feature rasters
             +--> OpenStreetMap nearby water
             +--> Census block-group lookup
                        |
                        v
                  OpenFEMA area signal
             |
             v
 FEMA benchmark + neighborhood history + physical context
```

The flood-risk API loads the independent data sources in parallel. A
physical-only calibration model uses elevation, rainfall, nearby-water, and
terrain context to estimate unclassified locations without using FEMA data for
the target address. A separate claim model uses FEMA context and reports
claim-location similarity as a secondary score.

## Displayed Results

### Primary 100-year estimate

The primary result is `hundredYearFloodEstimate`. For FEMA classifications that
define a numeric annual-chance event, BayFlood calculates the chance of at
least one such event over 100 years:

```text
100-year chance = 1 - (1 - annual chance)^100
```

That produces:

- `63.4%` for a mapped 1%-annual-chance FEMA event
- `18.1%` for a mapped 0.2%-annual-chance FEMA event

When the address has no numeric FEMA classification, BayFlood switches to an
experimental model trained to reconstruct mapped FEMA hazard classes from
physical conditions alone. The target address contributes only:

- Elevation
- NOAA Atlas 14 100-year, 24-hour rainfall
- Nearest mapped-water type and distance
- Terrain slope
- Local elevation relative to the surrounding 250 m and 500 m
- D8 upslope flow accumulation
- Height above modeled drainage

The calibration table contains 10,000 spatially sampled Bay Area locations.
Deterministic 0.05-degree geographic blocks target 70% training, 15%
validation, and 15% untouched test partitions. Because whole blocks stay
together, the realized split contains 7,364 training rows, 1,336 validation
rows, and 1,300 test rows. The production gradient-boosted model reaches
`84.9%` three-class accuracy on the untouched test set. ROC-AUC is `0.988` for
identifying 1%-annual-chance terrain and `0.902` for identifying terrain at or
above the 0.2% class. Its Brier scores are `0.0336` and `0.1050`, respectively;
lower Brier scores indicate better probability predictions.

The model probability-weights three annual-chance anchors: `1%`, `0.2%`, and a
central `0.1%` assumption for minimal-hazard terrain. Because the minimal class
has no exact FEMA annual chance, the interface also reports a range obtained by
varying that anchor from `0%` through `0.2%`.

This is the cumulative chance of the mapped event under constant-chance and
independent-year assumptions. The physical-only result is an experimental
screening estimate, not a calibrated probability that a particular building
will flood.

### FEMA neighborhood claim history

BayFlood matches the selected coordinates to a Census block group and looks up
the corresponding historical OpenFEMA NFIP claim signal. When available, the
card reports:

- Historical claim count
- NFIP policy-years used as the exposure denominator
- Bay Area percentile
- Very low, Low, Moderate, High, or Very high historical signal
- Census block group and data year

This is an area-level insurance-history signal, not an exact-address flood
history or property probability. A block group without a usable NFIP rate is
shown as `No area rate`; that does not mean the property is flood-free.

### Secondary claim-location similarity

The trained model's 0-100 output appears separately as `Claim-location
similarity: N/100`. It estimates how closely the physical conditions resemble
known NFIP claim locations in the training data. It is not displayed with a
percent sign and is not used as the 100-year probability.

The model can screen a covered Bay Area address even when that address has no
known flood or NFIP claim history. If FEMA membership or a mapped distance is
unavailable, the model treats that input as missing and substitutes the
training-set median rather than treating the address as safely outside the
floodplain. Elevation, rainfall, and nearby-water inputs can still be used.

No recorded flood is not proof that flooding has never occurred. Results in
areas with limited mapping or history should be interpreted with lower
confidence and remain screening estimates.

### Secondary model levels

| Model probability | Level |
| ---: | --- |
| Below 60% of the trained threshold | Low |
| 60% of the trained threshold to the trained threshold | Moderate |
| Trained threshold to 0.75 | High |
| 0.75 and above | Very high |

The numeric features have monotonic physical constraints: greater elevation or
distance from water/floodplains cannot increase the model signal, while greater
rainfall or SFHA membership cannot reduce it.

The elevation sampler reads a 5-by-5 pixel window and uses its median valid
value. This reduces the effect of a single unusual raster pixel.

If overlapping polygons contain the selected point, the lookup prefers the
feature with the higher hazard priority.

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
availability. Distance and feature type are inputs to the trained model.

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
    fema-area-signal.ts     Matches Census block groups to OpenFEMA history
    fema.ts                Looks up FEMA zones and SFHA distance
    flood-guidance.ts      Builds condition explanations and preparedness tips
    flood-model.ts         Calculates claim-location similarity
    physical-flood-estimate.ts  Estimates unclassified physical hazard
    flood-probability.ts   Calculates mapped FEMA event benchmarks
    rainfall.ts            Samples the NOAA rainfall GeoTIFF
    water.ts               Queries and measures nearby mapped water

FEMAdata/                  Local FEMA county GeoJSON files
elevation/                 Local USGS elevation GeoTIFF files
rainfall/                  NOAA rainfall GeoTIFF
public/data/               FEMA PMTiles and OpenFEMA area-signal files
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
  san_joaquin_boundary_flood_zones.geojson
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
  fema-area-claim-signal.json
```

`FEMAdata/` and `elevation/` are ignored by Git because these source datasets
are large. They must be installed separately after cloning the repository.

To download fresh copies of the nine Bay Area DFIRMs and the adjacent San
Joaquin boundary data directly from FEMA's official National Flood Hazard Layer,
then rebuild the browser map archive, run:

```bash
npm run fema:sync
```

The command records the source URL, retrieval time, DFIRM identifiers, feature
counts, geographic bounds, and SHA-256 checksums in
`data/fema-nfhl-snapshot.json`. Check the local files against FEMA's live layer
with:

```bash
npm run fema:verify
```

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
4. Wait while BayFlood reads elevation, FEMA, rainfall, nearby-water, Census,
   and OpenFEMA area data.
5. Review the 100-year benchmark or experimental estimate, neighborhood claim
   history, secondary similarity score, physical inputs, confidence, and
   preparedness feedback.

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
- FEMA 100-year event benchmark or experimental physical-only estimate
- Experimental estimate range and confidence when FEMA is unclassified
- FEMA neighborhood claim-history signal and Census block group
- Claim-location similarity score and interpretation
- Whether exact-address flood history was used
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
- Building-specific flood history or observed inundation reports

The FEMA benchmark describes mapped event frequency, not the chance that a
specific structure will experience inundation or damage. Where that benchmark
is unavailable, the physical-only percentage is explicitly experimental and
depends on a documented minimal-hazard annual-chance assumption.

The secondary claim-location similarity score is not a calibrated flood
probability. Two properties with similar scores may have different flood
mechanisms and real-world outcomes. The current comparison rows also lack
verified observation windows, so the claim classifier cannot be converted
responsibly into a probability of flooding within a future number of years.

## Production Notes

- The Node.js runtime must have access to the FEMA, elevation, rainfall, and
  hydrography files. Terrain can be local or supplied through
  `TERRAIN_COG_URL`.
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
