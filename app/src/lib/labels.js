// Turns raw class names such as "Tomato___Early_blight" into friendly text.
const CONDITION_OVERRIDES = {
  'Tomato___Spider_mites Two-spotted_spider_mite': 'Spider mites',
  'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Gray leaf spot',
  'Orange___Haunglongbing_(Citrus_greening)': 'Citrus greening',
  'Grape___Esca_(Black_Measles)': 'Esca (black measles)',
  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Leaf blight',
  'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Yellow leaf curl virus',
  'Tomato___Tomato_mosaic_virus': 'Mosaic virus'
};

const CROP_OVERRIDES = {
  'Pepper,_bell': 'Bell pepper',
  'Corn_(maize)': 'Corn',
  'Cherry_(including_sour)': 'Cherry'
};

const tidy = (s) => s.replace(/_/g, ' ').replace(/\s+/g, ' ').trim();
const sentence = (s) => s.charAt(0).toUpperCase() + s.slice(1);

export function describeClass(raw) {
  const [cropRaw, condRaw = ''] = raw.split('___');
  const crop = CROP_OVERRIDES[cropRaw] ?? tidy(cropRaw);
  const condition = CONDITION_OVERRIDES[raw] ?? sentence(tidy(condRaw));
  return { raw, crop, condition, healthy: condRaw.trim().toLowerCase() === 'healthy' };
}
