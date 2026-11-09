# Research starter: Bay Area flood prediction from storm and location data

## Goal

Predict whether a Bay Area location will be inundated under a specified storm or coastal-water-level event. The existing boosted model supplies a **static location baseline**; event rainfall, water levels, and observed flood outcomes are the next inputs needed to turn that baseline into an actual event-prediction model. After validation on historical events, sourced future rainfall and sea-level scenarios can be evaluated separately.

## Research question

**Can a model combining the project's terrain, imperviousness, and proximity-to-water inputs with event rainfall and coastal/river water levels predict observed Bay Area flood inundation more accurately than the current static screening model?**

The prediction unit is a location during a storm event. The first outcome is whether that location is inside a documented inundation footprint during that event. The eventual output could be an inundation map for a specified storm or forecast window, with uncertainty and coverage limits shown.

## What is already available

The app's current boosted model uses five static inputs and learns a FEMA mapped 1%-annual-chance versus minimal-hazard class. It reports 0.944 balanced accuracy on a geographic test split ([evaluation](../reports/boosted-dnn-comparison.md)). That is map-class agreement, **not a measured flood-event prediction**. The repository also has elevation, NLCD imperviousness, NOAA design-storm rainfall, hydrography, FEMA map labels, and an offline three-class model. The calibration table includes slope and height above drainage, which can be candidate location features, but testing those features is a subordinate modeling choice rather than the research goal.

## First buildable study

1. **Choose a bounded study area and events.** Start with one Bay Area subregion with usable historical flood observations and nearby water-level stations. Record several distinct storms, their dates, and their observed inundation footprints or verified flooded-location reports. Mark unobserved locations as unknown; do not automatically label them dry.
2. **Build a location-by-event table.** Reuse the current location features. Add rainfall accumulated before and during each event, relevant river stage or coastal tide/water level, and event time. Keep sources, timestamps, units, coverage, and missingness flags. Join each event to an observed flooded/not-flooded outcome only where that outcome is supported by evidence.
3. **Establish baselines.** Evaluate the existing static model score alone, then a simple event model using dynamic rainfall and water-level inputs, then a combined model. This measures whether the new event data actually adds predictive skill.
4. **Validate on unseen storms and places.** Hold out whole storm events for temporal evaluation and separate geographic areas for transfer evaluation. Tune thresholds on training/validation events only. Report precision, recall, false alarms, missed floods, calibration, and map overlap where full footprints exist. Stratify coastal and riverine sites because the mechanisms differ.
5. **Add future scenarios only after event validation.** Apply sourced rainfall and sea-level conditions to the validated event model as scenarios. BCDC's [RSAP coastal layers](https://rsap-open-data-bcdc.hub.arcgis.com/maps/BCDC%3A%3Acoastal-flood-hazards/about) can supply coastal planning context or an independent scenario comparison, but its 2050/2100 maps are not observed future outcomes or current event labels.

**Immediate next deliverable:** an event inventory and data dictionary for one subregion, followed by a small joined location-by-event pilot dataset. This makes the missing observations explicit and gives the current model a direct role in the first comparison.

## How the supplied papers inform the approach

- [Nevo et al. (2022)](https://hess.copernicus.org/articles/26/4013/2022/) separate observed-data quality, water-stage forecasting, inundation mapping, and communication. Their work motivates distinct prediction and mapping stages when gauges and event observations are available.
- [Tang et al. (2023)](https://www.sciencedirect.com/science/article/pii/S2214581823000939) group historical floods by rainfall behavior and adapt hydrologic-model parameters. This supports testing whether Bay Area storm types require different treatment after an event dataset exists.
- [Ighile et al. (2022)](https://www.mdpi.com/2071-1050/14/9/5039) combine GIS factors with historical flood records for susceptibility mapping. Their use of observed flood records is more relevant to the new outcome than using FEMA classes alone.
- [Mosavi et al. (2018)](https://www.mdpi.com/2073-4441/10/11/1536) and [Maspo et al. (2020)](https://eprints.utm.my/89973/1/NurAdibMaspo2020_EvaluationofMachineLearningApproachinFlood.pdf) review ML methods and flood-prediction inputs; they motivate comparing simple and hybrid baselines rather than assuming a larger network is automatically better.

## Interpretation

The current FEMA classifier is useful as a prior indication of where flooding is physically plausible. Actual event prediction requires event-varying inputs and independent observations. A future-scenario map should be labeled as conditional on the chosen rainfall and water-level assumptions, not as a verified probability that a particular building will flood.
