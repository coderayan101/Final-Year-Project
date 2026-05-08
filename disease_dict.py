"""
Disease dictionary - symptoms, cause, treatment, prevention advice for each
of the 38 PlantVillage classes.

Information compiled from general agricultural extension references.
Use this as a starting point; for a production system you would cite
region-specific extension services.
"""

from typing import Dict

DiseaseInfo = Dict[str, str]

# Shared healthy template
_HEALTHY = {
    "symptoms": "No visible disease symptoms. Leaves are of normal colour, shape, and size.",
    "cause": "Plant appears healthy.",
    "treatment": "No treatment required.",
    "prevention": "Maintain regular watering, balanced fertilization, and routine monitoring for pests and diseases.",
}

DISEASE_DICT: Dict[str, DiseaseInfo] = {
    # ---- Apple ----
    "Apple___Apple_scab": {
        "symptoms": "Olive-green to brown velvety spots on leaves and fruit; leaves may yellow and drop.",
        "cause": "Fungus Venturia inaequalis; favoured by cool, wet spring weather.",
        "treatment": "Apply fungicides such as captan or myclobutanil at green-tip and continue at 7-10 day intervals during wet weather.",
        "prevention": "Plant resistant varieties, prune for air circulation, remove fallen leaves in autumn to reduce overwintering inoculum.",
    },
    "Apple___Black_rot": {
        "symptoms": "Purple flecks on leaves that enlarge into brown 'frog-eye' spots; black rotted areas on fruit.",
        "cause": "Fungus Botryosphaeria obtusa; enters through wounds and dead tissue.",
        "treatment": "Prune out cankered wood and mummified fruit. Apply captan or thiophanate-methyl during the growing season.",
        "prevention": "Maintain tree vigour, prune dead wood, remove mummified fruit, and avoid mechanical injury to bark.",
    },
    "Apple___Cedar_apple_rust": {
        "symptoms": "Bright yellow-orange spots on upper leaf surface; orange tube-like structures on the underside.",
        "cause": "Fungus Gymnosporangium juniperi-virginianae; requires both apple and juniper/cedar hosts.",
        "treatment": "Apply myclobutanil or mancozeb from pink-bud stage through early fruit development.",
        "prevention": "Remove nearby juniper/cedar trees where possible, or plant resistant apple cultivars.",
    },
    "Apple___healthy": _HEALTHY,

    # ---- Blueberry ----
    "Blueberry___healthy": _HEALTHY,

    # ---- Cherry ----
    "Cherry_(including_sour)___Powdery_mildew": {
        "symptoms": "White, powdery fungal growth on leaves, shoots, and fruit; leaves may curl.",
        "cause": "Fungus Podosphaera clandestina; thrives in warm, dry weather with high humidity at night.",
        "treatment": "Apply sulphur-based fungicides or horticultural oil at first sign of disease; repeat every 7-14 days.",
        "prevention": "Prune to improve airflow, avoid excessive nitrogen, and plant resistant cultivars where available.",
    },
    "Cherry_(including_sour)___healthy": _HEALTHY,

    # ---- Corn ----
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "symptoms": "Narrow rectangular gray to tan lesions running parallel to leaf veins.",
        "cause": "Fungus Cercospora zeae-maydis; favoured by high humidity and warm temperatures.",
        "treatment": "Apply foliar fungicides such as azoxystrobin or pyraclostrobin at tasseling if disease pressure is high.",
        "prevention": "Rotate crops (avoid continuous corn), till residues, and plant tolerant hybrids.",
    },
    "Corn_(maize)___Common_rust_": {
        "symptoms": "Small, cinnamon-brown pustules on both leaf surfaces; pustules may turn black later.",
        "cause": "Fungus Puccinia sorghi; spores are wind-borne from southern regions.",
        "treatment": "Apply triazole or strobilurin fungicides if infection appears before tasseling.",
        "prevention": "Plant resistant hybrids; early planting can help avoid peak spore loads.",
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "symptoms": "Long, cigar-shaped gray-green to tan lesions on leaves, often starting on lower leaves.",
        "cause": "Fungus Exserohilum turcicum; favoured by moderate temperatures and prolonged leaf wetness.",
        "treatment": "Apply foliar fungicides (azoxystrobin, propiconazole) at early onset.",
        "prevention": "Rotate crops, use resistant hybrids, and manage crop debris by tilling.",
    },
    "Corn_(maize)___healthy": _HEALTHY,

    # ---- Grape ----
    "Grape___Black_rot": {
        "symptoms": "Circular tan leaf spots with dark borders; shriveled black 'mummy' berries.",
        "cause": "Fungus Guignardia bidwellii; overwinters in mummified fruit and canes.",
        "treatment": "Apply mancozeb or myclobutanil beginning at 2-4 inch shoot growth and repeat per label.",
        "prevention": "Remove mummified fruit, prune out infected canes, and ensure good vineyard sanitation.",
    },
    "Grape___Esca_(Black_Measles)": {
        "symptoms": "Interveinal 'tiger-stripe' leaf chlorosis and necrosis; dark spots on berries; sudden vine collapse possible.",
        "cause": "Complex of wood-rotting fungi (Phaeomoniella, Phaeoacremonium, Fomitiporia).",
        "treatment": "No effective chemical cure. Remove and destroy infected wood; surgical removal of cankers may delay progression.",
        "prevention": "Protect pruning wounds, avoid large cuts on old wood, and use clean nursery stock.",
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "symptoms": "Irregular dark brown to black angular leaf spots, often with yellow halos.",
        "cause": "Fungus Pseudocercospora vitis; thrives in warm, humid conditions.",
        "treatment": "Apply copper-based fungicides or mancozeb at first sign.",
        "prevention": "Improve canopy airflow through pruning, avoid overhead irrigation, and remove fallen leaves.",
    },
    "Grape___healthy": _HEALTHY,

    # ---- Orange ----
    "Orange___Haunglongbing_(Citrus_greening)": {
        "symptoms": "Asymmetric blotchy mottling of leaves, small lopsided bitter fruit, yellow shoots, eventual tree decline.",
        "cause": "Bacterium Candidatus Liberibacter asiaticus, spread by the Asian citrus psyllid.",
        "treatment": "No cure. Remove infected trees to reduce inoculum; nutritional programs can prolong productivity.",
        "prevention": "Control psyllid vectors with insecticides, use certified disease-free nursery stock, and monitor regularly.",
    },

    # ---- Peach ----
    "Peach___Bacterial_spot": {
        "symptoms": "Small dark angular leaf spots that drop out giving a 'shot-hole' appearance; dark sunken spots on fruit.",
        "cause": "Bacterium Xanthomonas arboricola pv. pruni; spreads in wet, windy weather.",
        "treatment": "Apply copper sprays during dormancy and early growth. Oxytetracycline can reduce severity in some cases.",
        "prevention": "Plant resistant varieties, avoid overhead irrigation, prune for airflow, and remove infected debris.",
    },
    "Peach___healthy": _HEALTHY,

    # ---- Pepper ----
    "Pepper,_bell___Bacterial_spot": {
        "symptoms": "Small water-soaked leaf spots that become brown and angular; raised scabby spots on fruit.",
        "cause": "Bacteria Xanthomonas species; spreads via rain splash and contaminated seed.",
        "treatment": "Apply copper + mancozeb sprays at first symptoms. Remove and destroy infected plants.",
        "prevention": "Use certified disease-free seed, rotate crops, avoid working wet plants, and mulch to reduce splash.",
    },
    "Pepper,_bell___healthy": _HEALTHY,

    # ---- Potato ----
    "Potato___Early_blight": {
        "symptoms": "Dark brown concentric-ringed 'target' spots on older leaves; lesions may coalesce and cause defoliation.",
        "cause": "Fungus Alternaria solani; favoured by warm temperatures and alternating wet/dry periods.",
        "treatment": "Apply chlorothalonil, mancozeb, or azoxystrobin at first sign and repeat every 7-10 days.",
        "prevention": "Rotate crops, maintain plant vigour with balanced fertilization, and remove infected debris.",
    },
    "Potato___Late_blight": {
        "symptoms": "Water-soaked pale green leaf spots turning brown-black; white fuzzy growth on leaf undersides in humid weather.",
        "cause": "Oomycete Phytophthora infestans; favoured by cool (15-20 C) and very humid conditions.",
        "treatment": "Apply chlorothalonil or mancozeb preventively; systemic fungicides (mefenoxam) if detected early. Remove and destroy infected plants.",
        "prevention": "Plant certified seed, avoid overhead irrigation, destroy cull piles and volunteer plants, and monitor forecasts.",
    },
    "Potato___healthy": _HEALTHY,

    # ---- Raspberry ----
    "Raspberry___healthy": _HEALTHY,

    # ---- Soybean ----
    "Soybean___healthy": _HEALTHY,

    # ---- Squash ----
    "Squash___Powdery_mildew": {
        "symptoms": "White powdery patches on leaves and stems; leaves eventually yellow and die.",
        "cause": "Fungi Podosphaera xanthii / Erysiphe cichoracearum; favoured by warm days with high humidity at night.",
        "treatment": "Apply sulphur, potassium bicarbonate, or neem oil at first sign. Systemic fungicides if severe.",
        "prevention": "Plant resistant varieties, space plants for airflow, and avoid excessive nitrogen.",
    },

    # ---- Strawberry ----
    "Strawberry___Leaf_scorch": {
        "symptoms": "Small dark purple spots on leaves that enlarge; leaves appear scorched and dry out.",
        "cause": "Fungus Diplocarpon earlianum; spreads in warm wet conditions.",
        "treatment": "Apply captan or myclobutanil at early bloom and repeat as per label.",
        "prevention": "Renovate beds, remove old infected leaves, and use drip irrigation to keep foliage dry.",
    },
    "Strawberry___healthy": _HEALTHY,

    # ---- Tomato ----
    "Tomato___Bacterial_spot": {
        "symptoms": "Small water-soaked leaf spots turning brown; scabby spots on fruit.",
        "cause": "Bacteria Xanthomonas species; seed-borne and splash-spread.",
        "treatment": "Apply copper sprays, sometimes combined with mancozeb. Remove severely infected plants.",
        "prevention": "Use disease-free seed, rotate crops, avoid overhead irrigation, and stake plants for airflow.",
    },
    "Tomato___Early_blight": {
        "symptoms": "Brown target-like concentric-ring spots on older leaves; may cause stem cankers and fruit rot.",
        "cause": "Fungus Alternaria solani; favoured by warm temperatures and leaf wetness.",
        "treatment": "Apply chlorothalonil or mancozeb at first sign, repeating every 7-10 days.",
        "prevention": "Mulch, stake plants, rotate crops, remove lower leaves that touch soil, and avoid overhead watering.",
    },
    "Tomato___Late_blight": {
        "symptoms": "Large irregular pale-green to brown water-soaked lesions on leaves and stems; white mold on leaf undersides; firm dark fruit rot.",
        "cause": "Oomycete Phytophthora infestans; favoured by cool (15-20 C) very humid conditions.",
        "treatment": "Apply chlorothalonil, mancozeb, or systemic fungicides (mefenoxam) immediately. Destroy infected plants to halt spread.",
        "prevention": "Plant resistant varieties, avoid overhead irrigation, space for airflow, and monitor weather forecasts.",
    },
    "Tomato___Leaf_Mold": {
        "symptoms": "Pale green to yellow spots on upper leaf surface; olive-green to gray fuzzy mold on underside.",
        "cause": "Fungus Passalora fulva; favoured by humidity above 85 percent, especially in greenhouses.",
        "treatment": "Apply chlorothalonil or mancozeb; improve ventilation if growing under cover.",
        "prevention": "Reduce humidity, improve airflow, avoid wetting leaves, and plant resistant cultivars.",
    },
    "Tomato___Septoria_leaf_spot": {
        "symptoms": "Many small circular spots with dark borders and gray centers on lower leaves; tiny black specks in lesions.",
        "cause": "Fungus Septoria lycopersici; favoured by warm, wet weather.",
        "treatment": "Apply chlorothalonil, mancozeb, or copper sprays at first sign, repeating weekly.",
        "prevention": "Mulch, stake, prune lower leaves, rotate crops, and avoid overhead irrigation.",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "symptoms": "Fine stippling/yellowing on leaves; fine webbing on leaf undersides; heavy infestations cause bronzing and leaf drop.",
        "cause": "Two-spotted spider mite (Tetranychus urticae); thrives in hot, dry conditions.",
        "treatment": "Spray with insecticidal soap, horticultural oil, or miticides such as abamectin. Encourage predatory mites.",
        "prevention": "Maintain adequate humidity, avoid water stress, and monitor regularly with a hand lens.",
    },
    "Tomato___Target_Spot": {
        "symptoms": "Small brown spots on leaves that enlarge into concentric rings resembling a target.",
        "cause": "Fungus Corynespora cassiicola; favoured by warm humid weather.",
        "treatment": "Apply chlorothalonil or mancozeb; azoxystrobin for rotation.",
        "prevention": "Prune for airflow, rotate crops, and remove infected plant debris promptly.",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "symptoms": "Leaves curl upward, become small and yellow, plants become stunted with few flowers or fruit.",
        "cause": "Virus spread by the silverleaf whitefly (Bemisia tabaci).",
        "treatment": "No cure. Remove and destroy infected plants; control whiteflies with insecticides or insecticidal soap.",
        "prevention": "Use resistant varieties, exclude whiteflies with fine mesh, apply reflective mulches, and monitor regularly.",
    },
    "Tomato___Tomato_mosaic_virus": {
        "symptoms": "Mottled light and dark green leaves; leaf distortion; stunted growth; reduced yield.",
        "cause": "Tomato mosaic virus (ToMV); mechanically transmitted on hands, tools, and seed.",
        "treatment": "No cure. Remove infected plants immediately. Disinfect tools with 10 percent bleach or milk solution.",
        "prevention": "Use certified virus-free seed, wash hands and tools, control weeds, and plant resistant varieties.",
    },
    "Tomato___healthy": _HEALTHY,
}


def get_disease_info(class_name: str) -> DiseaseInfo:
    """Return disease info for a class; fall back to a generic message if unknown."""
    return DISEASE_DICT.get(
        class_name,
        {
            "symptoms": "Information not available for this class.",
            "cause": "Unknown.",
            "treatment": "Consult your local agricultural extension service.",
            "prevention": "Practice good crop hygiene, rotation, and regular monitoring.",
        },
    )
