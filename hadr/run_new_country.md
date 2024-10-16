## How to run a new country

1. To add all files for a new country, run `onboard_country.py {country name}`
2. To run scraping for the country and insert into the general DB, run `clean_cycles.py {country name} {year}`
3. To calculate the metrics for the country, run `calculate_crps.py {country name} {year}`

## Different features to consider
Some country-feature pairs are already in the db (in parenthesis). Bold is the keyword (used in Chroma metadata).
- "militia activity" (drc, myanmar, afghanistan)
- "general" (drc)
  - this is a baseline comparison for the RAG model
- "refugee movements" (syria)