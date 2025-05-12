import fs from "fs";
import {
  CATEGORY,
  SORT_BY,
  SORT_ORDER,
  getFeaturesFromCategory,
  search,
  searchMultiples,
} from "leboncoin-api-search";

const typeVente = "Appartement";
const featureVente = getFeaturesFromCategory(CATEGORY.LOCATIONS).find(
  (feature) => feature.label === "Type de bien"
);

const paramTypeVente = featureVente?.param;
const valueTypeVente = featureVente?.values.find(
  (value) => value.label === typeVente
)?.value;

const results = await searchMultiples(
  {
    category: CATEGORY.LOCATIONS,
    sort_by: SORT_BY.PRICE,
    sort_order: SORT_ORDER.ASC,
    enums: {
      [paramTypeVente]: [valueTypeVente],
    },
    locations: ["Paris"],
    price_min: 500,
    price_max: 1500,
    limit: 10000,
  },
  60
);

// const categories = getCategories();
// console.log(categories);

fs.writeFileSync("resultats.json", JSON.stringify(results, null, 2));
console.log;
