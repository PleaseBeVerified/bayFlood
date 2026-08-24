export function coordinateKey(latitude, longitude) {
  return `${Number(latitude).toFixed(6)},${Number(longitude).toFixed(6)}`;
}

export function extractCensusGeography(payload) {
  const geographies = payload?.result?.geographies ?? {};
  const block = geographies["2020 Census Blocks"]?.[0];
  if (!block?.GEOID || String(block.GEOID).length < 12) return null;
  return {
    census_block_geoid: String(block.GEOID),
    census_geoid: String(block.GEOID).slice(0, 12),
    census_tract_geoid: String(block.GEOID).slice(0, 11),
    county_fips: String(block.GEOID).slice(0, 5),
  };
}
