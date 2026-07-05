"""
============================================================
  Flight Delay Predictor — Boarding Pass Edition
  A Streamlit application that predicts whether a flight will
  arrive 15+ minutes late, using a pre-trained XGBoost model.
  The result renders as an animated boarding-pass stamp.
============================================================
"""

# ---------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------
import pickle
import warnings
from datetime import date, time as dtime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).parent
MODELS_DIR = APP_DIR / "Models"
UTILS_DIR = APP_DIR / "Utils"
DATA_DIR = APP_DIR / "Data"
ENCODERS_PATH = UTILS_DIR / "label_encoders.pkl"

# Project identity
PROJECT_NAME = "SkyWise AI"
PROJECT_TAGLINE = "Predict delays before takeoff."

# Both trained models are shipped side by side so the person can compare them
# or pick a favorite before predicting.
MODEL_REGISTRY = {
    "XGBoost": dict(
        path=MODELS_DIR / "flight_delay_xgboost_model.pkl",
        blurb="Gradient-boosted trees, tuned for raw accuracy on tabular flight records.",
        accent="var(--gold)",
    ),
    "LightGBM": dict(
        path=MODELS_DIR / "flight_delay_lgbm_model.pkl",
        blurb="Leaf-wise gradient boosting, tuned for speed on the same feature set.",
        accent="var(--teal)",
    ),
}

DATA_REGISTRY = {
    "Flight delay data": DATA_DIR / "cleaned_flight_delay_data.csv",
    "Sample": DATA_DIR / "sample.csv",
}

st.set_page_config(
    page_title=f"{PROJECT_NAME} · Flight Delay Predictor",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Quick-start routes so people can try the tool in one click instead of
# hand-typing every field.
PRESETS = {
    "ATL → FLL": dict(origin="ATL", dest="FLL", distance=581, elapsed=110, dep=(16, 31), arr=(18, 21)),
    "JFK → LAX": dict(origin="JFK", dest="LAX", distance=2475, elapsed=390, dep=(8, 0), arr=(11, 30)),
    "ORD → DEN": dict(origin="ORD", dest="DEN", distance=888, elapsed=150, dep=(9, 15), arr=(11, 5)),
    "DFW → HOU": dict(origin="DFW", dest="HOU", distance=224, elapsed=75, dep=(7, 0), arr=(8, 15)),
}

# A complete reference table mapping every IATA airport code seen in
# training (369 airports, sourced from the cleaned training dataset) to its
# city / state, so the form can auto-fill those model features instead of
# forcing the user to type them by hand.
AIRPORT_DB = {
    "ABE": ("Allentown/Bethlehem/Easton, PA", "PA", "Pennsylvania"),
    "ABI": ("Abilene, TX", "TX", "Texas"),
    "ABQ": ("Albuquerque, NM", "NM", "New Mexico"),
    "ABR": ("Aberdeen, SD", "SD", "South Dakota"),
    "ABY": ("Albany, GA", "GA", "Georgia"),
    "ACT": ("Waco, TX", "TX", "Texas"),
    "ACV": ("Arcata/Eureka, CA", "CA", "California"),
    "ACY": ("Atlantic City, NJ", "NJ", "New Jersey"),
    "ADK": ("Adak Island, AK", "AK", "Alaska"),
    "ADQ": ("Kodiak, AK", "AK", "Alaska"),
    "AEX": ("Alexandria, LA", "LA", "Louisiana"),
    "AGS": ("Augusta, GA", "GA", "Georgia"),
    "AKN": ("King Salmon, AK", "AK", "Alaska"),
    "ALB": ("Albany, NY", "NY", "New York"),
    "ALO": ("Waterloo, IA", "IA", "Iowa"),
    "ALS": ("Alamosa, CO", "CO", "Colorado"),
    "ALW": ("Walla Walla, WA", "WA", "Washington"),
    "AMA": ("Amarillo, TX", "TX", "Texas"),
    "ANC": ("Anchorage, AK", "AK", "Alaska"),
    "APN": ("Alpena, MI", "MI", "Michigan"),
    "ART": ("Watertown, NY", "NY", "New York"),
    "ASE": ("Aspen, CO", "CO", "Colorado"),
    "ATL": ("Atlanta, GA", "GA", "Georgia"),
    "ATW": ("Appleton, WI", "WI", "Wisconsin"),
    "ATY": ("Watertown, SD", "SD", "South Dakota"),
    "AUS": ("Austin, TX", "TX", "Texas"),
    "AVL": ("Asheville, NC", "NC", "North Carolina"),
    "AVP": ("Scranton/Wilkes-Barre, PA", "PA", "Pennsylvania"),
    "AZA": ("Phoenix, AZ", "AZ", "Arizona"),
    "AZO": ("Kalamazoo, MI", "MI", "Michigan"),
    "BDL": ("Hartford, CT", "CT", "Connecticut"),
    "BET": ("Bethel, AK", "AK", "Alaska"),
    "BFF": ("Scottsbluff, NE", "NE", "Nebraska"),
    "BFL": ("Bakersfield, CA", "CA", "California"),
    "BGM": ("Binghamton, NY", "NY", "New York"),
    "BGR": ("Bangor, ME", "ME", "Maine"),
    "BHM": ("Birmingham, AL", "AL", "Alabama"),
    "BIH": ("Bishop, CA", "CA", "California"),
    "BIL": ("Billings, MT", "MT", "Montana"),
    "BIS": ("Bismarck/Mandan, ND", "ND", "North Dakota"),
    "BJI": ("Bemidji, MN", "MN", "Minnesota"),
    "BLI": ("Bellingham, WA", "WA", "Washington"),
    "BLV": ("Belleville, IL", "IL", "Illinois"),
    "BMI": ("Bloomington/Normal, IL", "IL", "Illinois"),
    "BNA": ("Nashville, TN", "TN", "Tennessee"),
    "BOI": ("Boise, ID", "ID", "Idaho"),
    "BOS": ("Boston, MA", "MA", "Massachusetts"),
    "BPT": ("Beaumont/Port Arthur, TX", "TX", "Texas"),
    "BQK": ("Brunswick, GA", "GA", "Georgia"),
    "BQN": ("Aguadilla, PR", "PR", "Puerto Rico"),
    "BRD": ("Brainerd, MN", "MN", "Minnesota"),
    "BRO": ("Brownsville, TX", "TX", "Texas"),
    "BRW": ("Barrow, AK", "AK", "Alaska"),
    "BTM": ("Butte, MT", "MT", "Montana"),
    "BTR": ("Baton Rouge, LA", "LA", "Louisiana"),
    "BTV": ("Burlington, VT", "VT", "Vermont"),
    "BUF": ("Buffalo, NY", "NY", "New York"),
    "BUR": ("Burbank, CA", "CA", "California"),
    "BWI": ("Baltimore, MD", "MD", "Maryland"),
    "BZN": ("Bozeman, MT", "MT", "Montana"),
    "CAE": ("Columbia, SC", "SC", "South Carolina"),
    "CAK": ("Akron, OH", "OH", "Ohio"),
    "CDC": ("Cedar City, UT", "UT", "Utah"),
    "CDV": ("Cordova, AK", "AK", "Alaska"),
    "CGI": ("Cape Girardeau, MO", "MO", "Missouri"),
    "CHA": ("Chattanooga, TN", "TN", "Tennessee"),
    "CHO": ("Charlottesville, VA", "VA", "Virginia"),
    "CHS": ("Charleston, SC", "SC", "South Carolina"),
    "CID": ("Cedar Rapids/Iowa City, IA", "IA", "Iowa"),
    "CIU": ("Sault Ste. Marie, MI", "MI", "Michigan"),
    "CKB": ("Clarksburg/Fairmont, WV", "WV", "West Virginia"),
    "CLE": ("Cleveland, OH", "OH", "Ohio"),
    "CLL": ("College Station/Bryan, TX", "TX", "Texas"),
    "CLT": ("Charlotte, NC", "NC", "North Carolina"),
    "CMH": ("Columbus, OH", "OH", "Ohio"),
    "CMI": ("Champaign/Urbana, IL", "IL", "Illinois"),
    "CMX": ("Hancock/Houghton, MI", "MI", "Michigan"),
    "CNY": ("Moab, UT", "UT", "Utah"),
    "COD": ("Cody, WY", "WY", "Wyoming"),
    "COS": ("Colorado Springs, CO", "CO", "Colorado"),
    "COU": ("Columbia, MO", "MO", "Missouri"),
    "CPR": ("Casper, WY", "WY", "Wyoming"),
    "CRP": ("Corpus Christi, TX", "TX", "Texas"),
    "CRW": ("Charleston/Dunbar, WV", "WV", "West Virginia"),
    "CSG": ("Columbus, GA", "GA", "Georgia"),
    "CVG": ("Cincinnati, OH", "KY", "Kentucky"),
    "CWA": ("Mosinee, WI", "WI", "Wisconsin"),
    "CYS": ("Cheyenne, WY", "WY", "Wyoming"),
    "DAB": ("Daytona Beach, FL", "FL", "Florida"),
    "DAL": ("Dallas, TX", "TX", "Texas"),
    "DAY": ("Dayton, OH", "OH", "Ohio"),
    "DBQ": ("Dubuque, IA", "IA", "Iowa"),
    "DCA": ("Washington, DC", "VA", "Virginia"),
    "DDC": ("Dodge City, KS", "KS", "Kansas"),
    "DEC": ("Decatur, IL", "IL", "Illinois"),
    "DEN": ("Denver, CO", "CO", "Colorado"),
    "DFW": ("Dallas/Fort Worth, TX", "TX", "Texas"),
    "DHN": ("Dothan, AL", "AL", "Alabama"),
    "DIK": ("Dickinson, ND", "ND", "North Dakota"),
    "DLG": ("Dillingham, AK", "AK", "Alaska"),
    "DLH": ("Duluth, MN", "MN", "Minnesota"),
    "DRO": ("Durango, CO", "CO", "Colorado"),
    "DRT": ("Del Rio, TX", "TX", "Texas"),
    "DSM": ("Des Moines, IA", "IA", "Iowa"),
    "DTW": ("Detroit, MI", "MI", "Michigan"),
    "DVL": ("Devils Lake, ND", "ND", "North Dakota"),
    "EAR": ("Kearney, NE", "NE", "Nebraska"),
    "EAT": ("Wenatchee, WA", "WA", "Washington"),
    "EAU": ("Eau Claire, WI", "WI", "Wisconsin"),
    "ECP": ("Panama City, FL", "FL", "Florida"),
    "EGE": ("Eagle, CO", "CO", "Colorado"),
    "EKO": ("Elko, NV", "NV", "Nevada"),
    "ELM": ("Elmira/Corning, NY", "NY", "New York"),
    "ELP": ("El Paso, TX", "TX", "Texas"),
    "ERI": ("Erie, PA", "PA", "Pennsylvania"),
    "ESC": ("Escanaba, MI", "MI", "Michigan"),
    "EUG": ("Eugene, OR", "OR", "Oregon"),
    "EVV": ("Evansville, IN", "IN", "Indiana"),
    "EWN": ("New Bern/Morehead/Beaufort, NC", "NC", "North Carolina"),
    "EWR": ("Newark, NJ", "NJ", "New Jersey"),
    "EYW": ("Key West, FL", "FL", "Florida"),
    "FAI": ("Fairbanks, AK", "AK", "Alaska"),
    "FAR": ("Fargo, ND", "ND", "North Dakota"),
    "FAT": ("Fresno, CA", "CA", "California"),
    "FAY": ("Fayetteville, NC", "NC", "North Carolina"),
    "FCA": ("Kalispell, MT", "MT", "Montana"),
    "FLG": ("Flagstaff, AZ", "AZ", "Arizona"),
    "FLL": ("Fort Lauderdale, FL", "FL", "Florida"),
    "FLO": ("Florence, SC", "SC", "South Carolina"),
    "FNT": ("Flint, MI", "MI", "Michigan"),
    "FOD": ("Fort Dodge, IA", "IA", "Iowa"),
    "FSD": ("Sioux Falls, SD", "SD", "South Dakota"),
    "FSM": ("Fort Smith, AR", "AR", "Arkansas"),
    "FWA": ("Fort Wayne, IN", "IN", "Indiana"),
    "GCC": ("Gillette, WY", "WY", "Wyoming"),
    "GCK": ("Garden City, KS", "KS", "Kansas"),
    "GEG": ("Spokane, WA", "WA", "Washington"),
    "GFK": ("Grand Forks, ND", "ND", "North Dakota"),
    "GGG": ("Longview, TX", "TX", "Texas"),
    "GJT": ("Grand Junction, CO", "CO", "Colorado"),
    "GNV": ("Gainesville, FL", "FL", "Florida"),
    "GPT": ("Gulfport/Biloxi, MS", "MS", "Mississippi"),
    "GRB": ("Green Bay, WI", "WI", "Wisconsin"),
    "GRI": ("Grand Island, NE", "NE", "Nebraska"),
    "GRK": ("Killeen, TX", "TX", "Texas"),
    "GRR": ("Grand Rapids, MI", "MI", "Michigan"),
    "GSO": ("Greensboro/High Point, NC", "NC", "North Carolina"),
    "GSP": ("Greer, SC", "SC", "South Carolina"),
    "GTF": ("Great Falls, MT", "MT", "Montana"),
    "GTR": ("Columbus, MS", "MS", "Mississippi"),
    "GUC": ("Gunnison, CO", "CO", "Colorado"),
    "GUM": ("Guam, TT", "TT", "U.S. Pacific Trust Territories and Possessions"),
    "HDN": ("Hayden, CO", "CO", "Colorado"),
    "HGR": ("Hagerstown, MD", "MD", "Maryland"),
    "HHH": ("Hilton Head, SC", "SC", "South Carolina"),
    "HIB": ("Hibbing, MN", "MN", "Minnesota"),
    "HLN": ("Helena, MT", "MT", "Montana"),
    "HNL": ("Honolulu, HI", "HI", "Hawaii"),
    "HOB": ("Hobbs, NM", "NM", "New Mexico"),
    "HOU": ("Houston, TX", "TX", "Texas"),
    "HPN": ("White Plains, NY", "NY", "New York"),
    "HRL": ("Harlingen/San Benito, TX", "TX", "Texas"),
    "HSV": ("Huntsville, AL", "AL", "Alabama"),
    "HTS": ("Ashland, WV", "WV", "West Virginia"),
    "HYS": ("Hays, KS", "KS", "Kansas"),
    "IAD": ("Washington, DC", "VA", "Virginia"),
    "IAG": ("Niagara Falls, NY", "NY", "New York"),
    "IAH": ("Houston, TX", "TX", "Texas"),
    "ICT": ("Wichita, KS", "KS", "Kansas"),
    "IDA": ("Idaho Falls, ID", "ID", "Idaho"),
    "ILG": ("Wilmington, DE", "DE", "Delaware"),
    "ILM": ("Wilmington, NC", "NC", "North Carolina"),
    "IMT": ("Iron Mountain/Kingsfd, MI", "MI", "Michigan"),
    "IND": ("Indianapolis, IN", "IN", "Indiana"),
    "INL": ("International Falls, MN", "MN", "Minnesota"),
    "ISP": ("Islip, NY", "NY", "New York"),
    "ITH": ("Ithaca/Cortland, NY", "NY", "New York"),
    "ITO": ("Hilo, HI", "HI", "Hawaii"),
    "JAC": ("Jackson, WY", "WY", "Wyoming"),
    "JAN": ("Jackson/Vicksburg, MS", "MS", "Mississippi"),
    "JAX": ("Jacksonville, FL", "FL", "Florida"),
    "JFK": ("New York, NY", "NY", "New York"),
    "JLN": ("Joplin, MO", "MO", "Missouri"),
    "JMS": ("Jamestown, ND", "ND", "North Dakota"),
    "JNU": ("Juneau, AK", "AK", "Alaska"),
    "JST": ("Johnstown, PA", "PA", "Pennsylvania"),
    "KOA": ("Kona, HI", "HI", "Hawaii"),
    "KTN": ("Ketchikan, AK", "AK", "Alaska"),
    "LAN": ("Lansing, MI", "MI", "Michigan"),
    "LAR": ("Laramie, WY", "WY", "Wyoming"),
    "LAS": ("Las Vegas, NV", "NV", "Nevada"),
    "LAW": ("Lawton/Fort Sill, OK", "OK", "Oklahoma"),
    "LAX": ("Los Angeles, CA", "CA", "California"),
    "LBB": ("Lubbock, TX", "TX", "Texas"),
    "LBE": ("Latrobe, PA", "PA", "Pennsylvania"),
    "LBF": ("North Platte, NE", "NE", "Nebraska"),
    "LBL": ("Liberal, KS", "KS", "Kansas"),
    "LCH": ("Lake Charles, LA", "LA", "Louisiana"),
    "LCK": ("Columbus, OH", "OH", "Ohio"),
    "LEX": ("Lexington, KY", "KY", "Kentucky"),
    "LFT": ("Lafayette, LA", "LA", "Louisiana"),
    "LGA": ("New York, NY", "NY", "New York"),
    "LGB": ("Long Beach, CA", "CA", "California"),
    "LIH": ("Lihue, HI", "HI", "Hawaii"),
    "LIT": ("Little Rock, AR", "AR", "Arkansas"),
    "LNK": ("Lincoln, NE", "NE", "Nebraska"),
    "LRD": ("Laredo, TX", "TX", "Texas"),
    "LSE": ("La Crosse, WI", "WI", "Wisconsin"),
    "LWB": ("Lewisburg, WV", "WV", "West Virginia"),
    "LWS": ("Lewiston, ID", "ID", "Idaho"),
    "LYH": ("Lynchburg, VA", "VA", "Virginia"),
    "MAF": ("Midland/Odessa, TX", "TX", "Texas"),
    "MBS": ("Saginaw/Bay City/Midland, MI", "MI", "Michigan"),
    "MCI": ("Kansas City, MO", "MO", "Missouri"),
    "MCO": ("Orlando, FL", "FL", "Florida"),
    "MCW": ("Mason City, IA", "IA", "Iowa"),
    "MDT": ("Harrisburg, PA", "PA", "Pennsylvania"),
    "MDW": ("Chicago, IL", "IL", "Illinois"),
    "MEI": ("Meridian, MS", "MS", "Mississippi"),
    "MEM": ("Memphis, TN", "TN", "Tennessee"),
    "MFE": ("Mission/McAllen/Edinburg, TX", "TX", "Texas"),
    "MFR": ("Medford, OR", "OR", "Oregon"),
    "MGM": ("Montgomery, AL", "AL", "Alabama"),
    "MHK": ("Manhattan/Ft. Riley, KS", "KS", "Kansas"),
    "MHT": ("Manchester, NH", "NH", "New Hampshire"),
    "MIA": ("Miami, FL", "FL", "Florida"),
    "MKE": ("Milwaukee, WI", "WI", "Wisconsin"),
    "MKG": ("Muskegon, MI", "MI", "Michigan"),
    "MLB": ("Melbourne, FL", "FL", "Florida"),
    "MLI": ("Moline, IL", "IL", "Illinois"),
    "MLU": ("Monroe, LA", "LA", "Louisiana"),
    "MOB": ("Mobile, AL", "AL", "Alabama"),
    "MOT": ("Minot, ND", "ND", "North Dakota"),
    "MQT": ("Marquette, MI", "MI", "Michigan"),
    "MRY": ("Monterey, CA", "CA", "California"),
    "MSN": ("Madison, WI", "WI", "Wisconsin"),
    "MSO": ("Missoula, MT", "MT", "Montana"),
    "MSP": ("Minneapolis, MN", "MN", "Minnesota"),
    "MSY": ("New Orleans, LA", "LA", "Louisiana"),
    "MTJ": ("Montrose/Delta, CO", "CO", "Colorado"),
    "MYR": ("Myrtle Beach, SC", "SC", "South Carolina"),
    "OAJ": ("Jacksonville/Camp Lejeune, NC", "NC", "North Carolina"),
    "OAK": ("Oakland, CA", "CA", "California"),
    "OGD": ("Ogden, UT", "UT", "Utah"),
    "OGG": ("Kahului, HI", "HI", "Hawaii"),
    "OGS": ("Ogdensburg, NY", "NY", "New York"),
    "OKC": ("Oklahoma City, OK", "OK", "Oklahoma"),
    "OMA": ("Omaha, NE", "NE", "Nebraska"),
    "OME": ("Nome, AK", "AK", "Alaska"),
    "ONT": ("Ontario, CA", "CA", "California"),
    "ORD": ("Chicago, IL", "IL", "Illinois"),
    "ORF": ("Norfolk, VA", "VA", "Virginia"),
    "ORH": ("Worcester, MA", "MA", "Massachusetts"),
    "OTH": ("North Bend/Coos Bay, OR", "OR", "Oregon"),
    "OTZ": ("Kotzebue, AK", "AK", "Alaska"),
    "OWB": ("Owensboro, KY", "KY", "Kentucky"),
    "PAE": ("Everett, WA", "WA", "Washington"),
    "PAH": ("Paducah, KY", "KY", "Kentucky"),
    "PBG": ("Plattsburgh, NY", "NY", "New York"),
    "PBI": ("West Palm Beach/Palm Beach, FL", "FL", "Florida"),
    "PDX": ("Portland, OR", "OR", "Oregon"),
    "PGD": ("Punta Gorda, FL", "FL", "Florida"),
    "PGV": ("Greenville, NC", "NC", "North Carolina"),
    "PHF": ("Newport News/Williamsburg, VA", "VA", "Virginia"),
    "PHL": ("Philadelphia, PA", "PA", "Pennsylvania"),
    "PHX": ("Phoenix, AZ", "AZ", "Arizona"),
    "PIA": ("Peoria, IL", "IL", "Illinois"),
    "PIB": ("Hattiesburg/Laurel, MS", "MS", "Mississippi"),
    "PIE": ("St. Petersburg, FL", "FL", "Florida"),
    "PIH": ("Pocatello, ID", "ID", "Idaho"),
    "PIR": ("Pierre, SD", "SD", "South Dakota"),
    "PIT": ("Pittsburgh, PA", "PA", "Pennsylvania"),
    "PLN": ("Pellston, MI", "MI", "Michigan"),
    "PNS": ("Pensacola, FL", "FL", "Florida"),
    "PPG": ("Pago Pago, TT", "TT", "U.S. Pacific Trust Territories and Possessions"),
    "PQI": ("Presque Isle/Houlton, ME", "ME", "Maine"),
    "PRC": ("Prescott, AZ", "AZ", "Arizona"),
    "PSC": ("Pasco/Kennewick/Richland, WA", "WA", "Washington"),
    "PSE": ("Ponce, PR", "PR", "Puerto Rico"),
    "PSG": ("Petersburg, AK", "AK", "Alaska"),
    "PSM": ("Portsmouth, NH", "NH", "New Hampshire"),
    "PSP": ("Palm Springs, CA", "CA", "California"),
    "PUB": ("Pueblo, CO", "CO", "Colorado"),
    "PUW": ("Pullman, WA", "WA", "Washington"),
    "PVD": ("Providence, RI", "RI", "Rhode Island"),
    "PVU": ("Provo, UT", "UT", "Utah"),
    "PWM": ("Portland, ME", "ME", "Maine"),
    "RAP": ("Rapid City, SD", "SD", "South Dakota"),
    "RDD": ("Redding, CA", "CA", "California"),
    "RDM": ("Bend/Redmond, OR", "OR", "Oregon"),
    "RDU": ("Raleigh/Durham, NC", "NC", "North Carolina"),
    "RFD": ("Rockford, IL", "IL", "Illinois"),
    "RHI": ("Rhinelander, WI", "WI", "Wisconsin"),
    "RIC": ("Richmond, VA", "VA", "Virginia"),
    "RIW": ("Riverton/Lander, WY", "WY", "Wyoming"),
    "RKS": ("Rock Springs, WY", "WY", "Wyoming"),
    "RNO": ("Reno, NV", "NV", "Nevada"),
    "ROA": ("Roanoke, VA", "VA", "Virginia"),
    "ROC": ("Rochester, NY", "NY", "New York"),
    "ROW": ("Roswell, NM", "NM", "New Mexico"),
    "RST": ("Rochester, MN", "MN", "Minnesota"),
    "RSW": ("Fort Myers, FL", "FL", "Florida"),
    "SAF": ("Santa Fe, NM", "NM", "New Mexico"),
    "SAN": ("San Diego, CA", "CA", "California"),
    "SAT": ("San Antonio, TX", "TX", "Texas"),
    "SAV": ("Savannah, GA", "GA", "Georgia"),
    "SBA": ("Santa Barbara, CA", "CA", "California"),
    "SBN": ("South Bend, IN", "IN", "Indiana"),
    "SBP": ("San Luis Obispo, CA", "CA", "California"),
    "SBY": ("Salisbury, MD", "MD", "Maryland"),
    "SCC": ("Deadhorse, AK", "AK", "Alaska"),
    "SCE": ("State College, PA", "PA", "Pennsylvania"),
    "SCK": ("Stockton, CA", "CA", "California"),
    "SDF": ("Louisville, KY", "KY", "Kentucky"),
    "SEA": ("Seattle, WA", "WA", "Washington"),
    "SFB": ("Sanford, FL", "FL", "Florida"),
    "SFO": ("San Francisco, CA", "CA", "California"),
    "SGF": ("Springfield, MO", "MO", "Missouri"),
    "SGU": ("St. George, UT", "UT", "Utah"),
    "SHD": ("Staunton, VA", "VA", "Virginia"),
    "SHR": ("Sheridan, WY", "WY", "Wyoming"),
    "SHV": ("Shreveport, LA", "LA", "Louisiana"),
    "SIT": ("Sitka, AK", "AK", "Alaska"),
    "SJC": ("San Jose, CA", "CA", "California"),
    "SJT": ("San Angelo, TX", "TX", "Texas"),
    "SJU": ("San Juan, PR", "PR", "Puerto Rico"),
    "SLC": ("Salt Lake City, UT", "UT", "Utah"),
    "SLN": ("Salina, KS", "KS", "Kansas"),
    "SMF": ("Sacramento, CA", "CA", "California"),
    "SMX": ("Santa Maria, CA", "CA", "California"),
    "SNA": ("Santa Ana, CA", "CA", "California"),
    "SPI": ("Springfield, IL", "IL", "Illinois"),
    "SPN": ("Saipan, TT", "TT", "U.S. Pacific Trust Territories and Possessions"),
    "SPS": ("Wichita Falls, TX", "TX", "Texas"),
    "SRQ": ("Sarasota/Bradenton, FL", "FL", "Florida"),
    "STC": ("St. Cloud, MN", "MN", "Minnesota"),
    "STL": ("St. Louis, MO", "MO", "Missouri"),
    "STS": ("Santa Rosa, CA", "CA", "California"),
    "STT": ("Charlotte Amalie, VI", "VI", "U.S. Virgin Islands"),
    "STX": ("Christiansted, VI", "VI", "U.S. Virgin Islands"),
    "SUN": ("Sun Valley/Hailey/Ketchum, ID", "ID", "Idaho"),
    "SUX": ("Sioux City, IA", "IA", "Iowa"),
    "SWF": ("Newburgh/Poughkeepsie, NY", "NY", "New York"),
    "SWO": ("Stillwater, OK", "OK", "Oklahoma"),
    "SYR": ("Syracuse, NY", "NY", "New York"),
    "TBN": ("Fort Leonard Wood, MO", "MO", "Missouri"),
    "TLH": ("Tallahassee, FL", "FL", "Florida"),
    "TOL": ("Toledo, OH", "OH", "Ohio"),
    "TPA": ("Tampa, FL", "FL", "Florida"),
    "TRI": ("Bristol/Johnson City/Kingsport, TN", "TN", "Tennessee"),
    "TTN": ("Trenton, NJ", "NJ", "New Jersey"),
    "TUL": ("Tulsa, OK", "OK", "Oklahoma"),
    "TUS": ("Tucson, AZ", "AZ", "Arizona"),
    "TVC": ("Traverse City, MI", "MI", "Michigan"),
    "TWF": ("Twin Falls, ID", "ID", "Idaho"),
    "TXK": ("Texarkana, AR", "AR", "Arkansas"),
    "TYR": ("Tyler, TX", "TX", "Texas"),
    "TYS": ("Knoxville, TN", "TN", "Tennessee"),
    "USA": ("Concord, NC", "NC", "North Carolina"),
    "VCT": ("Victoria, TX", "TX", "Texas"),
    "VEL": ("Vernal, UT", "UT", "Utah"),
    "VLD": ("Valdosta, GA", "GA", "Georgia"),
    "VPS": ("Valparaiso, FL", "FL", "Florida"),
    "WRG": ("Wrangell, AK", "AK", "Alaska"),
    "XNA": ("Fayetteville, AR", "AR", "Arkansas"),
    "XWA": ("Williston, ND", "ND", "North Dakota"),
    "YAK": ("Yakutat, AK", "AK", "Alaska"),
    "YKM": ("Yakima, WA", "WA", "Washington"),
    "YUM": ("Yuma, AZ", "AZ", "Arizona"),
}

# ---------------------------------------------------------------------------
# STYLING — palette: deep-space navy, gold ticket-stamp accent, teal/coral
# status colors. Type: Space Grotesk (display) + Inter (body) + JetBrains
# Mono (flight codes, times, probabilities) for an authentic ticket feel.
# ---------------------------------------------------------------------------
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

        :root {
            --bg-deep: #0b1226;
            --bg-deep-2: #111b34;
            --panel: rgba(255, 255, 255, 0.055);
            --panel-border: rgba(255, 255, 255, 0.12);
            --gold: #f2b134;
            --teal: #2dd4bf;
            --coral: #ef476f;
            --text-hi: #eaf0f7;
            --text-mid: #a9b4c6;
            --text-lo: #6b7690;
        }

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .stApp {
            background:
                radial-gradient(circle at 15% 0%, rgba(45, 212, 191, 0.07), transparent 45%),
                radial-gradient(circle at 85% 15%, rgba(242, 177, 52, 0.06), transparent 40%),
                var(--bg-deep);
        }

        /* Respect reduced-motion preferences */
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { animation-duration: 0.001ms !important; transition-duration: 0.001ms !important; }
        }

        .block-container { padding-top: 1.6rem; max-width: 1180px; }

        /* ---------- Header ---------- */
        .eyebrow {
            text-align: center;
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 0.22em;
            font-size: 0.72rem;
            color: var(--teal);
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }
        .app-title {
            text-align: center;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--text-hi);
            margin-bottom: 0.3rem;
            letter-spacing: -0.01em;
        }
        .app-subtitle {
            text-align: center;
            color: var(--text-mid);
            font-size: 1rem;
            font-weight: 400;
            margin-bottom: 1.6rem;
        }

        /* Animated dashed flight-path divider under the header */
        .flight-path {
            position: relative;
            height: 22px;
            margin: 0 auto 1.8rem auto;
            max-width: 480px;
            border-top: 2px dashed rgba(255,255,255,0.16);
        }
        .flight-path::after {
            content: '✈';
            position: absolute;
            top: -13.55px;
            left: 0%;
            color: var(--gold);
            font-size: 1.1rem;
            animation: flyAcross 6s linear infinite;
        }
        @keyframes flyAcross {
            0%   { left: 2%; opacity: 0; transform: rotate(0deg); }
            8%   { opacity: 1; }
            92%  { opacity: 1; }
            100% { left: 100%; opacity: 0; transform: rotate(0deg); }
        }

        /* ---------- Preset chips ---------- */
        div[data-testid="stButton"] button[kind="secondary"] {
            border-radius: 999px !important;
        }

        /* ---------- Glass panel (form sections) ---------- */
        .glass-card {
            background: var(--panel);
            backdrop-filter: blur(18px) saturate(160%);
            -webkit-backdrop-filter: blur(18px) saturate(160%);
            border-radius: 16px;
            border: 1px solid var(--panel-border);
            padding: 1.4rem 1.6rem 1.1rem 1.6rem;
            margin-bottom: 1rem;
        }

        .section-label {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--gold);
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 0.7rem;
        }

        label, .stMarkdown p, span { color: var(--text-hi) !important; }
        .stCaption, [data-testid="stCaptionContainer"] p { color: var(--text-lo) !important; }

        /* Inputs */
        div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input,
        .stDateInput input, .stTimeInput input {
            background: rgba(255,255,255,0.05) !important;
            border-radius: 10px !important;
            border: 1px solid var(--panel-border) !important;
            color: var(--text-hi) !important;
            font-family: 'JetBrains Mono', monospace !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            border-bottom: 1px solid var(--panel-border);
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.04);
            border-radius: 10px 10px 0 0;
            color: var(--text-mid);
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 500;
            padding: 0.6rem 1.1rem;
        }
        .stTabs [aria-selected="true"] { color: var(--gold) !important; }
        /* The baseweb active-tab underline defaults to a bulky, mis-colored
           bar; slim it down and recolor to match the palette. */
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: var(--gold) !important;
            height: 2px !important;
        }
        .stTabs [data-baseweb="tab-border"] {
            background-color: transparent !important;
        }
        .stTabs [data-baseweb="tab-panel"] {
            padding-top: 1.1rem;
        }

        /* ---------- Bordered containers (replaces the old raw-HTML
           "glass card" divs, which never actually wrapped their
           contents and rendered as an empty floating box). ---------- */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--panel);
            backdrop-filter: blur(18px) saturate(160%);
            -webkit-backdrop-filter: blur(18px) saturate(160%);
            border-radius: 16px !important;
            border: 1px solid var(--panel-border) !important;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] > div {
            gap: 0.6rem;
        }

        /* ---------- Sidebar ---------- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--bg-deep-2), var(--bg-deep));
            border-right: 1px solid var(--panel-border);
        }
        .brand-wrap {
            padding: 0.4rem 0.2rem 1.2rem 0.2rem;
            border-bottom: 1px solid var(--panel-border);
            margin-bottom: 1rem;
        }
        .brand-mark {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 1.5rem;
            color: var(--text-hi);
        }
        .brand-mark span.dot { color: var(--gold); }
        .brand-tagline {
            font-size: 0.82rem;
            color: var(--text-mid);
            margin-top: 0.35rem;
            line-height: 1.4;
        }
        /* Sidebar nav — plain buttons styled as a nav list, not toggles */
        section[data-testid="stSidebar"] div[data-testid="stButton"] {
            margin-bottom: 2px;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] button {
            width: 100%;
            display: flex;
            justify-content: flex-start;
            gap: 0.6rem;
            padding: 0.6rem 0.8rem;
            border-radius: 10px;
            border: 1px solid transparent;
            border-left: 3px solid transparent;
            background: transparent;
            color: var(--text-mid);
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 500;
            font-size: 0.95rem;
            text-align: left;
            box-shadow: none;
            transition: background 0.15s ease, color 0.15s ease;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
            background: rgba(255,255,255,0.06);
            color: var(--text-hi);
            border-color: transparent;
            border-left-color: rgba(242,177,52,0.4);
            transform: none;
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] button:focus:not(:active) {
            color: var(--text-hi);
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
            background: rgba(242,177,52,0.14);
            border-left: 3px solid var(--gold);
            color: var(--gold);
        }
        section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"]:hover {
            background: rgba(242,177,52,0.2);
            color: var(--gold);
        }
        .sidebar-footer {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--panel-border);
            font-size: 0.72rem;
            color: var(--text-lo);
            line-height: 1.5;
        }
        .model-chip {
            display: inline-block;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--bg-deep);
            background: var(--gold);
            border-radius: 999px;
            padding: 0.2rem 0.65rem;
            margin-bottom: 0.9rem;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 10px;
            border: 1px solid var(--panel-border);
            background: rgba(255,255,255,0.05);
            color: var(--text-hi);
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            border-color: var(--gold);
            color: var(--gold);
            transform: translateY(-1px);
        }

        .predict-btn button {
            width: 100%;
            background: linear-gradient(135deg, rgba(242,177,52,0.28), rgba(242,177,52,0.12));
            border: 1px solid rgba(242,177,52,0.55) !important;
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.05rem;
            font-weight: 600;
            padding: 0.7rem 1rem;
        }
        .predict-btn button:hover {
            box-shadow: 0 8px 26px rgba(242,177,52,0.28);
            transform: translateY(-2px);
        }

        /* ---------- Boarding pass ---------- */
        .boarding-pass {
            position: relative;
            background: linear-gradient(155deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
            border: 1px solid var(--panel-border);
            border-radius: 18px;
            padding: 1.5rem 1.6rem;
            backdrop-filter: blur(20px);
            animation: cardRise 0.5s ease-out;
        }
        @keyframes cardRise {
            from { opacity: 0; transform: translateY(14px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .bp-eyebrow {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.18em;
            color: var(--text-lo);
            text-transform: uppercase;
        }
        .bp-route {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 0.5rem 0 0.9rem 0;
        }
        .bp-code {
            font-family: 'JetBrains Mono', monospace;
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-hi);
            line-height: 1;
        }
        .bp-city {
            font-size: 0.72rem;
            color: var(--text-mid);
            margin-top: 2px;
        }
        .bp-plane {
            color: var(--gold);
            font-size: 1.3rem;
            transform: rotate(90deg);
        }
        .bp-divider {
            border-top: 2px dashed var(--panel-border);
            position: relative;
            margin: 0.9rem 0;
        }
        .bp-divider::before, .bp-divider::after {
            content: '';
            position: absolute;
            top: -10px;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: var(--bg-deep);
        }
        .bp-divider::before { left: -1.6rem; }
        .bp-divider::after { right: -1.6rem; }
        .bp-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.8rem;
            margin-bottom: 0.2rem;
        }
        .bp-field-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.62rem;
            letter-spacing: 0.12em;
            color: var(--text-lo);
            text-transform: uppercase;
            margin-bottom: 2px;
        }
        .bp-field-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.95rem;
            color: var(--text-hi);
            font-weight: 500;
        }

        /* Stamp */
        .stamp-wrap {
            display: flex;
            justify-content: center;
            margin: 1.1rem 0 0.4rem 0;
        }
        .stamp {
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 1.3rem;
            letter-spacing: 0.06em;
            padding: 0.55rem 1.3rem;
            border-radius: 10px;
            border: 3px solid currentColor;
            text-transform: uppercase;
            transform: rotate(-4deg) scale(1);
            animation: stampLand 0.45s cubic-bezier(.36,1.4,.4,1);
        }
        @keyframes stampLand {
            0%   { opacity: 0; transform: rotate(-4deg) scale(2.2); }
            60%  { opacity: 1; }
            100% { opacity: 1; transform: rotate(-4deg) scale(1); }
        }
        .stamp-ontime { color: var(--teal); }
        .stamp-delay  { color: var(--coral); }

        .prob-row { display: flex; gap: 1rem; margin-top: 0.8rem; }
        .prob-col { flex: 1; }
        .prob-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--text-lo);
            margin-bottom: 4px;
        }
        .prob-bar-track {
            height: 8px;
            border-radius: 6px;
            background: rgba(255,255,255,0.08);
            overflow: hidden;
        }
        .prob-bar-fill {
            height: 100%;
            border-radius: 6px;
            animation: fillBar 0.7s ease-out;
        }
        @keyframes fillBar {
            from { width: 0%; }
        }
        .prob-pct {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: var(--text-hi);
            margin-top: 3px;
        }

        .preview-hint {
            text-align: center;
            color: var(--text-lo);
            font-size: 0.82rem;
            font-style: italic;
            margin-top: 0.6rem;
        }

        .disclaimer {
            text-align: center;
            margin-top: 1.6rem;
            color: var(--text-lo);
            font-size: 0.78rem;
        }

        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        header[data-testid="stHeader"] { display: none; }
        div[data-testid="stDecoration"] { display: none; }
        div[data-testid="stToolbar"] { display: none; }
        .stApp { margin-top: 0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS — data & model
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Loads every model in the registry plus the shared label encoders."""
    errors = []
    models = {}
    encoders = None

    for name, meta in MODEL_REGISTRY.items():
        try:
            with open(meta["path"], "rb") as f:
                models[name] = pickle.load(f)
        except Exception as e:
            errors.append(f"Could not load {name} model ({meta['path'].name}): {e}")

    try:
        with open(ENCODERS_PATH, "rb") as f:
            encoders = pickle.load(f)
    except Exception as e:
        errors.append(f"Could not load encoders file ({ENCODERS_PATH.name}): {e}")

    return models, encoders, errors


@st.cache_data(show_spinner=False)
def load_dataset(path: Path):
    """Loads a CSV from the Data folder. Cached so switching pages doesn't
    re-read the file from disk every time."""
    return pd.read_csv(path)


FALLBACK_FEATURE_ORDER = [
    "Year", "Quarter", "Month", "DayofMonth", "DayOfWeek",
    "Marketing_Airline_Network", "Operated_or_Branded_Code_Share_Partners",
    "Flight_Number_Marketing_Airline", "Operating_Airline ",
    "Flight_Number_Operating_Airline", "Origin", "OriginCityName",
    "OriginState", "OriginStateName", "Dest", "DestCityName",
    "DestState", "DestStateName", "CRSDepTime", "DepTimeBlk",
    "CRSArrTime", "ArrTimeBlk", "CRSElapsedTime", "Distance",
    "DistanceGroup",
]


def get_feature_names(model) -> list:
    """Pulls the exact feature order the model was trained on, whichever
    of the two boosting libraries it came from."""
    try:
        return list(model.get_booster().feature_names)  # XGBoost
    except Exception:
        pass
    try:
        return list(model.feature_name_)  # LightGBM sklearn wrapper
    except Exception:
        pass
    try:
        return list(model.booster_.feature_name())  # LightGBM booster
    except Exception:
        pass
    return FALLBACK_FEATURE_ORDER


def safe_label_encode(encoders: dict, column: str, value):
    """Encodes a categorical value using its fitted LabelEncoder, falling back
    to the encoder's first known class for unseen categories."""
    encoder = encoders.get(column)
    if encoder is None:
        return value
    try:
        return int(encoder.transform([value])[0])
    except (ValueError, KeyError):
        return int(encoder.transform([encoder.classes_[0]])[0])


def time_to_block(hhmm: int) -> str:
    """Converts an HHMM integer into its scheduled time-block string
    (e.g. 1631 -> '1600-1659'), matching the training convention."""
    hour = int(hhmm) // 100
    if hour == 0:
        return "0001-0559"
    return f"{hour:02d}00-{hour:02d}59"


def distance_to_group(distance: int) -> int:
    """Buckets raw distance (miles) into the standard DOT distance group (1-11)."""
    if distance >= 2500:
        return 11
    return min(11, int(distance) // 250 + 1)


def airport_lookup(code: str):
    """Returns (city, state, state_name) for a known airport code, else placeholders."""
    return AIRPORT_DB.get(code, (f"{code} (city unknown)", "NA", "Unknown"))


ENCODED_COLUMNS = {
    "Marketing_Airline_Network", "Operated_or_Branded_Code_Share_Partners",
    "Operating_Airline ", "Origin", "OriginCityName", "OriginState",
    "OriginStateName", "Dest", "DestCityName", "DestState",
    "DestStateName", "DepTimeBlk", "ArrTimeBlk",
}

# LightGBM silently sanitizes column names it doesn't like (e.g. a trailing
# space becomes a trailing underscore), so a model's own reported feature
# name won't always match the canonical key used in `inputs`/`encoders`.
# This maps a "canonicalized" name (whitespace/underscore-insensitive) back
# to the real key.
_CANON_LOOKUP = {c.strip().rstrip("_").strip(): c for c in FALLBACK_FEATURE_ORDER}


def _canonical_key(col: str) -> str:
    return _CANON_LOOKUP.get(col.strip().rstrip("_").strip(), col)


def build_feature_row(inputs: dict, encoders: dict, feature_order: list) -> pd.DataFrame:
    """Assembles a single-row DataFrame in the exact column order/naming the
    model expects, resolving each column back to its canonical key so this
    works regardless of which library (and its own naming quirks) trained
    the model."""
    row = {}
    for col in feature_order:
        canonical_col = _canonical_key(col)
        value = inputs[canonical_col]
        if canonical_col in ENCODED_COLUMNS:
            row[col] = safe_label_encode(encoders, canonical_col, value)
        else:
            row[col] = value
    return pd.DataFrame([row], columns=feature_order)


def time_obj_to_hhmm(t: dtime) -> int:
    return t.hour * 100 + t.minute


def date_to_fields(d: date):
    """Derives Year/Quarter/Month/DayofMonth/DayOfWeek from a calendar date.
    DayOfWeek follows the training convention: Monday=1 ... Sunday=7."""
    quarter = (d.month - 1) // 3 + 1
    return d.year, quarter, d.month, d.day, d.isoweekday()


# ---------------------------------------------------------------------------
# STATE HELPERS — presets & swap
# ---------------------------------------------------------------------------
def apply_preset(name: str):
    p = PRESETS[name]
    st.session_state["sel_origin"] = p["origin"]
    st.session_state["sel_dest"] = p["dest"]
    st.session_state["num_distance"] = p["distance"]
    st.session_state["num_elapsed"] = p["elapsed"]
    st.session_state["time_dep"] = dtime(*p["dep"])
    st.session_state["time_arr"] = dtime(*p["arr"])


def swap_route():
    o = st.session_state.get("sel_origin")
    d = st.session_state.get("sel_dest")
    st.session_state["sel_origin"] = d
    st.session_state["sel_dest"] = o


# ---------------------------------------------------------------------------
# UI — HEADER
# ---------------------------------------------------------------------------
def render_header(active_model: str = "XGBoost"):
    st.markdown(f'<div class="eyebrow">{active_model} model · live forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-title">Will your flight be on time?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Enter your itinerary and get an instant delay forecast, '
        'styled like a boarding pass.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="flight-path"></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI — INPUT FORM (left column, tabbed)
# ---------------------------------------------------------------------------
def render_input_form(encoders: dict) -> dict:
    origin_codes = sorted(encoders["Origin"].classes_.tolist()) if encoders else sorted(AIRPORT_DB.keys())
    dest_codes = sorted(encoders["Dest"].classes_.tolist()) if encoders else sorted(AIRPORT_DB.keys())
    airline_codes = sorted(encoders["Marketing_Airline_Network"].classes_.tolist()) if encoders else ["AA", "DL", "UA", "WN"]

    # --- Quick-start route chips ---
    st.markdown('<div class="section-label">⚡ Quick start</div>', unsafe_allow_html=True)
    chip_cols = st.columns(len(PRESETS))
    for col, name in zip(chip_cols, PRESETS.keys()):
        col.button(name, key=f"preset_{name}", on_click=apply_preset, args=(name,), use_container_width=True)
    st.write("")

    tab_date, tab_route, tab_schedule, tab_advanced = st.tabs(
        ["📅 Date & Airline", "📍 Route", "🕐 Schedule", "⚙️ Advanced"]
    )

    # ---- Tab: Date & Airline ----
    with tab_date:
        with st.container(border=True):
            c1, c2 = st.columns(2)
            flight_date = c1.date_input("Flight date", value=st.session_state.get("date_flight", date(2024, 1, 6)), key="date_flight")
            year, quarter, month, day_of_month, day_of_week = date_to_fields(flight_date)
            dow_label = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day_of_week - 1]
            c2.markdown(
                f'<div class="section-label" style="margin-top:1.7rem;">Q{quarter} · {dow_label} · '
                f'day {day_of_month} of month {month}</div>',
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            airline_default = st.session_state.get("sel_airline", "DL")
            airline_index = airline_codes.index(airline_default) if airline_default in airline_codes else 0
            airline = c1.selectbox("Marketing airline", airline_codes, index=airline_index, key="sel_airline")
            flight_number = c2.number_input(
                "Flight number", min_value=1, max_value=9999,
                value=st.session_state.get("num_flightnum", 1582), step=1, key="num_flightnum",
            )

    # ---- Tab: Route ----
    with tab_route:
        with st.container(border=True):
            c1, c_swap, c2 = st.columns([5, 1, 5])
            with c1:
                origin_default = st.session_state.get("sel_origin", "ATL")
                origin_index = origin_codes.index(origin_default) if origin_default in origin_codes else 0
                origin = st.selectbox("Origin airport", origin_codes, index=origin_index, key="sel_origin")
                o_city, o_state, o_state_name = airport_lookup(origin)
                st.caption(f"📌 {o_city}")
            with c_swap:
                st.write("")
                st.write("")
                st.button("⇄", key="swap_btn", on_click=swap_route, help="Swap origin and destination")
            with c2:
                dest_default = st.session_state.get("sel_dest", "FLL")
                dest_index = dest_codes.index(dest_default) if dest_default in dest_codes else 0
                dest = st.selectbox("Destination airport", dest_codes, index=dest_index, key="sel_dest")
                d_city, d_state, d_state_name = airport_lookup(dest)
                st.caption(f"📌 {d_city}")

    # ---- Tab: Schedule ----
    with tab_schedule:
        with st.container(border=True):
            c1, c2 = st.columns(2)
            dep_time = c1.time_input("Scheduled departure", value=st.session_state.get("time_dep", dtime(16, 31)), key="time_dep")
            arr_time = c2.time_input("Scheduled arrival", value=st.session_state.get("time_arr", dtime(18, 21)), key="time_arr")

            dep_minutes = dep_time.hour * 60 + dep_time.minute
            arr_minutes = arr_time.hour * 60 + arr_time.minute
            auto_elapsed = (arr_minutes - dep_minutes) % (24 * 60)

            c1, c2 = st.columns(2)
            elapsed = c1.number_input(
                "Scheduled duration (min)", min_value=15, max_value=900,
                value=st.session_state.get("num_elapsed", auto_elapsed or 110), step=1, key="num_elapsed",
            )
            c1.caption(f"⏱ From your times: {auto_elapsed} min")
            distance = c2.number_input(
                "Distance (miles)", min_value=30, max_value=6000,
                value=st.session_state.get("num_distance", 581), step=1, key="num_distance",
            )

    # ---- Tab: Advanced ----
    with tab_advanced:
        with st.container(border=True):
            st.caption("Optional overrides for the operating carrier and codeshare partner. Defaults follow your marketing airline.")
            c1, c2 = st.columns(2)
            operating_options = sorted(encoders["Operating_Airline "].classes_.tolist()) if encoders else [airline]
            operating_default = airline if airline in operating_options else operating_options[0]
            operating_airline = c1.selectbox(
                "Operating airline", operating_options,
                index=operating_options.index(operating_default), key="sel_operating",
            )
            codeshare_options = sorted(encoders["Operated_or_Branded_Code_Share_Partners"].classes_.tolist()) if encoders else [airline]
            codeshare_default = airline if airline in codeshare_options else codeshare_options[0]
            codeshare = c2.selectbox(
                "Codeshare partner", codeshare_options,
                index=codeshare_options.index(codeshare_default), key="sel_codeshare",
            )

    raw_inputs = {
        "Year": int(year),
        "Quarter": int(quarter),
        "Month": int(month),
        "DayofMonth": int(day_of_month),
        "DayOfWeek": int(day_of_week),
        "Marketing_Airline_Network": airline,
        "Operated_or_Branded_Code_Share_Partners": codeshare,
        "Flight_Number_Marketing_Airline": int(flight_number),
        "Operating_Airline ": operating_airline,
        "Flight_Number_Operating_Airline": int(flight_number),
        "Origin": origin,
        "OriginCityName": o_city,
        "OriginState": o_state,
        "OriginStateName": o_state_name,
        "Dest": dest,
        "DestCityName": d_city,
        "DestState": d_state,
        "DestStateName": d_state_name,
        "CRSDepTime": time_obj_to_hhmm(dep_time),
        "DepTimeBlk": time_to_block(time_obj_to_hhmm(dep_time)),
        "CRSArrTime": time_obj_to_hhmm(arr_time),
        "ArrTimeBlk": time_to_block(time_obj_to_hhmm(arr_time)),
        "CRSElapsedTime": int(elapsed),
        "Distance": int(distance),
        "DistanceGroup": distance_to_group(distance),
    }
    return raw_inputs


def current_trip_inputs(encoders: dict) -> dict:
    """Reads the itinerary currently held in session_state (set by the
    Predict tab's widgets, or sensible defaults if it hasn't been touched
    yet) without rendering any new widgets. Used by the comparison page so
    it can stay in sync with whatever trip the person last configured."""
    ss = st.session_state
    flight_date = ss.get("date_flight", date(2024, 1, 6))
    year, quarter, month, day_of_month, day_of_week = date_to_fields(flight_date)

    airline = ss.get("sel_airline", "DL")
    flight_number = ss.get("num_flightnum", 1582)

    origin = ss.get("sel_origin", "ATL")
    dest = ss.get("sel_dest", "FLL")
    o_city, o_state, o_state_name = airport_lookup(origin)
    d_city, d_state, d_state_name = airport_lookup(dest)

    dep_time = ss.get("time_dep", dtime(16, 31))
    arr_time = ss.get("time_arr", dtime(18, 21))
    elapsed = ss.get("num_elapsed", 110)
    distance = ss.get("num_distance", 581)

    operating_airline = ss.get("sel_operating", airline)
    codeshare = ss.get("sel_codeshare", airline)

    return {
        "Year": int(year), "Quarter": int(quarter), "Month": int(month),
        "DayofMonth": int(day_of_month), "DayOfWeek": int(day_of_week),
        "Marketing_Airline_Network": airline,
        "Operated_or_Branded_Code_Share_Partners": codeshare,
        "Flight_Number_Marketing_Airline": int(flight_number),
        "Operating_Airline ": operating_airline,
        "Flight_Number_Operating_Airline": int(flight_number),
        "Origin": origin, "OriginCityName": o_city, "OriginState": o_state,
        "OriginStateName": o_state_name,
        "Dest": dest, "DestCityName": d_city, "DestState": d_state,
        "DestStateName": d_state_name,
        "CRSDepTime": time_obj_to_hhmm(dep_time),
        "DepTimeBlk": time_to_block(time_obj_to_hhmm(dep_time)),
        "CRSArrTime": time_obj_to_hhmm(arr_time),
        "ArrTimeBlk": time_to_block(time_obj_to_hhmm(arr_time)),
        "CRSElapsedTime": int(elapsed), "Distance": int(distance),
        "DistanceGroup": distance_to_group(distance),
    }


# ---------------------------------------------------------------------------
# UI — BOARDING PASS (right column: live preview + result)
# ---------------------------------------------------------------------------
def render_boarding_pass(inputs: dict, prediction=None):
    """Renders the sticky boarding-pass panel. Shows a live preview of the
    trip, and if a prediction has been made, morphs into the stamped result."""
    dep_hh = inputs["CRSDepTime"] // 100
    dep_mm = inputs["CRSDepTime"] % 100
    arr_hh = inputs["CRSArrTime"] // 100
    arr_mm = inputs["CRSArrTime"] % 100

    html = ['<div class="boarding-pass">']
    html.append('<div class="bp-eyebrow">Boarding Pass · Delay Forecast</div>')
    html.append('<div class="bp-route">')
    html.append(f'<div><div class="bp-code">{inputs["Origin"]}</div><div class="bp-city">{inputs["OriginCityName"]}</div></div>')
    html.append('<div class="bp-plane">✈</div>')
    html.append(f'<div style="text-align:right;"><div class="bp-code">{inputs["Dest"]}</div><div class="bp-city">{inputs["DestCityName"]}</div></div>')
    html.append('</div>')

    html.append('<div class="bp-divider"></div>')
    html.append('<div class="bp-grid">')
    for label, value in [
        ("Date", f'{inputs["Year"]}-{inputs["Month"]:02d}-{inputs["DayofMonth"]:02d}'),
        ("Airline", f'{inputs["Marketing_Airline_Network"]} {inputs["Flight_Number_Marketing_Airline"]}'),
        ("Distance", f'{inputs["Distance"]} mi'),
        ("Departs", f'{dep_hh:02d}:{dep_mm:02d}'),
        ("Arrives", f'{arr_hh:02d}:{arr_mm:02d}'),
        ("Duration", f'{inputs["CRSElapsedTime"]} min'),
    ]:
        html.append(f'<div><div class="bp-field-label">{label}</div><div class="bp-field-value">{value}</div></div>')
    html.append('</div>')

    if prediction is not None:
        pred_class, proba = prediction
        ontime_pct = float(proba[0]) * 100
        delay_pct = float(proba[1]) * 100

        html.append('<div class="bp-divider"></div>')
        if pred_class == 1:
            html.append('<div class="stamp-wrap"><div class="stamp stamp-delay">⚠ Delayed</div></div>')
        else:
            html.append('<div class="stamp-wrap"><div class="stamp stamp-ontime">✓ On Time</div></div>')

        html.append('<div class="prob-row">')
        html.append(
            f'<div class="prob-col"><div class="prob-label">On-time</div>'
            f'<div class="prob-bar-track"><div class="prob-bar-fill" style="width:{ontime_pct:.1f}%; background: var(--teal);"></div></div>'
            f'<div class="prob-pct">{ontime_pct:.1f}%</div></div>'
        )
        html.append(
            f'<div class="prob-col"><div class="prob-label">Delay</div>'
            f'<div class="prob-bar-track"><div class="prob-bar-fill" style="width:{delay_pct:.1f}%; background: var(--coral);"></div></div>'
            f'<div class="prob-pct">{delay_pct:.1f}%</div></div>'
        )
        html.append('</div>')
    else:
        html.append('<div class="preview-hint">Fill in your itinerary, then tap Predict.</div>')

    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI — SIDEBAR
# ---------------------------------------------------------------------------
NAV_ITEMS = [
    ("predict", "✈️", "Predict"),
    ("select_model", "🧠", "Select model"),
    ("comparison", "📊", "Model comparison"),
    ("explore_data", "🗂️", "Explore data"),
    ("about", "ℹ️", "About"),
]


def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(
            f'''<div class="brand-wrap">
                <div class="brand-mark">🎫 {PROJECT_NAME}<span class="dot">.</span></div>
                <div class="brand-tagline">{PROJECT_TAGLINE}</div>
            </div>''',
            unsafe_allow_html=True,
        )

        if "nav_page" not in st.session_state:
            st.session_state["nav_page"] = "predict"

        for key, icon, label in NAV_ITEMS:
            is_active = st.session_state["nav_page"] == key
            if st.button(
                f"{icon}  {label}", key=f"nav_{key}", use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["nav_page"] = key
                st.rerun()

        active_model = st.session_state.get("active_model", "XGBoost")
        st.markdown(
            f'<div class="sidebar-footer">Currently forecasting with<br>'
            f'<span class="model-chip">{active_model}</span><br>'
            f'Trained on historical U.S. domestic flight records. '
            f'Forecasts are estimates, not guarantees.</div>',
            unsafe_allow_html=True,
        )
    return st.session_state["nav_page"]


# ---------------------------------------------------------------------------
# UI — PAGE: PREDICT
# ---------------------------------------------------------------------------
def render_predict_page(models: dict, encoders: dict):
    active_model_name = st.session_state.get("active_model", "XGBoost")
    model = models[active_model_name]
    feature_order = get_feature_names(model)

    render_header(active_model_name)

    col_form, col_pass = st.columns([1.55, 1], gap="large")

    with col_form:
        raw_inputs = render_input_form(encoders)
        st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
        predict_clicked = st.button("🔮  Predict Delay Risk", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_pass:
        prediction = None
        if predict_clicked:
            try:
                with st.spinner("Checking the odds..."):
                    feature_row = build_feature_row(raw_inputs, encoders, feature_order)
                    proba = model.predict_proba(feature_row)[0]
                    pred_class = int(np.argmax(proba))
                prediction = (pred_class, proba)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        render_boarding_pass(raw_inputs, prediction)

    st.markdown(
        '<div class="disclaimer">Predictions are estimates based on historical patterns '
        'and are not a guarantee of actual flight performance.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# UI — PAGE: SELECT MODEL
# ---------------------------------------------------------------------------
def render_model_select_page(models: dict):
    st.markdown('<div class="eyebrow">Model registry</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-title" style="font-size:2.1rem;">Choose your forecasting model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Both models trained on the same historical flight data and feature '
        'set. Pick whichever one should power the Predict tab.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="flight-path"></div>', unsafe_allow_html=True)

    cols = st.columns(len(MODEL_REGISTRY))
    for col, (name, meta) in zip(cols, MODEL_REGISTRY.items()):
        with col:
            with st.container(border=True):
                st.markdown(
                    f'<div class="section-label" style="color:{meta["accent"]};">{name}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(meta["blurb"])

                model_obj = models.get(name)
                if model_obj is not None:
                    n_features = len(get_feature_names(model_obj))
                    st.markdown(
                        f'<div class="bp-field-label">Input features</div>'
                        f'<div class="bp-field-value" style="margin-bottom:0.8rem;">{n_features}</div>',
                        unsafe_allow_html=True,
                    )

                is_active = st.session_state.get("active_model", "XGBoost") == name
                if is_active:
                    st.success("Active on the Predict tab", icon="✅")
                else:
                    if st.button(f"Use {name}", key=f"use_{name}", use_container_width=True):
                        st.session_state["active_model"] = name
                        st.rerun()


# ---------------------------------------------------------------------------
# UI — PAGE: MODEL COMPARISON
# ---------------------------------------------------------------------------
def render_comparison_page(models: dict, encoders: dict):
    st.markdown('<div class="eyebrow">Side-by-side forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-title" style="font-size:2.1rem;">Model comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">See how XGBoost and LightGBM each score the same itinerary. '
        'Set the trip on the Predict tab, then check back here.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="flight-path"></div>', unsafe_allow_html=True)

    inputs = current_trip_inputs(encoders)

    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        with st.container(border=True):
            st.markdown(
                f'<div class="bp-eyebrow">Comparing</div>'
                f'<div class="bp-route">'
                f'<div><div class="bp-code">{inputs["Origin"]}</div>'
                f'<div class="bp-city">{inputs["OriginCityName"]}</div></div>'
                f'<div class="bp-plane">✈</div>'
                f'<div style="text-align:right;"><div class="bp-code">{inputs["Dest"]}</div>'
                f'<div class="bp-city">{inputs["DestCityName"]}</div></div>'
                f'</div>'
                f'<div class="bp-field-label">Flight</div>'
                f'<div class="bp-field-value">'
                f'{inputs["Marketing_Airline_Network"]} {inputs["Flight_Number_Marketing_Airline"]} · '
                f'{inputs["Year"]}-{inputs["Month"]:02d}-{inputs["DayofMonth"]:02d}</div>',
                unsafe_allow_html=True,
            )

    feature_order_cache = {}
    results = {}
    errors = []
    for name, model_obj in models.items():
        try:
            feature_order = feature_order_cache.setdefault(name, get_feature_names(model_obj))
            feature_row = build_feature_row(inputs, encoders, feature_order)
            proba = model_obj.predict_proba(feature_row)[0]
            results[name] = proba
        except Exception as e:
            errors.append(f"{name}: {e}")

    for err in errors:
        st.error(err)

    if results:
        chart_df = pd.DataFrame(
            {name: [proba[0] * 100, proba[1] * 100] for name, proba in results.items()},
            index=["On-time %", "Delay %"],
        )
        st.bar_chart(chart_df, height=340, stack=False, y_label="Probability (%)")

        cols = st.columns(len(results))
        for col, (name, proba) in zip(cols, results.items()):
            verdict = "✓ On Time" if proba[0] >= proba[1] else "⚠ Delayed"
            verdict_class = "stamp-ontime" if proba[0] >= proba[1] else "stamp-delay"
            with col:
                with st.container(border=True):
                    st.markdown(
                        f'<div class="section-label" style="color:{MODEL_REGISTRY[name]["accent"]};">{name}</div>'
                        f'<div class="stamp-wrap" style="margin:0.4rem 0;">'
                        f'<div class="stamp {verdict_class}" style="font-size:1rem; transform:none;">{verdict}</div></div>'
                        f'<div class="prob-row">'
                        f'<div class="prob-col"><div class="prob-label">On-time</div>'
                        f'<div class="prob-pct">{proba[0]*100:.1f}%</div></div>'
                        f'<div class="prob-col"><div class="prob-label">Delay</div>'
                        f'<div class="prob-pct">{proba[1]*100:.1f}%</div></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

    st.markdown(
        '<div class="disclaimer">Comparison uses the itinerary currently set on the Predict tab. '
        'Both models see identical inputs — any difference in the chart comes from the model itself.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# UI — PAGE: EXPLORE DATA
# ---------------------------------------------------------------------------
def render_explore_data_page():
    st.markdown('<div class="eyebrow">Under the hood</div>', unsafe_allow_html=True)
    st.markdown('<div class="app-title" style="font-size:2.1rem;">Explore the data</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Browse the historical flight records the models were '
        'trained on.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="flight-path"></div>', unsafe_allow_html=True)

    dataset_name = st.radio(
        "Dataset", list(DATA_REGISTRY.keys()), horizontal=True, label_visibility="collapsed",
    )
    csv_path = DATA_REGISTRY[dataset_name]

    try:
        df = load_dataset(csv_path)
    except Exception as e:
        st.error(f"Could not load {csv_path.name}: {e}")
        return

    with st.container(border=True):
        st.markdown('<div class="section-label">Overview</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Rows", f"{len(df):,}")
        m2.metric("Columns", f"{df.shape[1]:,}")
        m3.metric("File", csv_path.name)

    with st.container(border=True):
        st.markdown('<div class="section-label">Preview</div>', unsafe_allow_html=True)
        n_rows = st.slider("Rows to show", min_value=5, max_value=min(500, max(5, len(df))), value=min(50, len(df)))
        st.dataframe(df.head(n_rows), use_container_width=True)

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        with st.container(border=True):
            st.markdown('<div class="section-label">Column summary</div>', unsafe_allow_html=True)
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

    with st.container(border=True):
        st.markdown('<div class="section-label">Chart a column</div>', unsafe_allow_html=True)
        if numeric_cols:
            chart_col = st.selectbox("Numeric column", numeric_cols)
            st.bar_chart(df[chart_col].value_counts().sort_index(), height=320)
        else:
            st.caption("No numeric columns found to chart.")

    st.markdown(
        '<div class="disclaimer">Data shown here is the same historical dataset used to train '
        'both models — it does not reflect live or future flight conditions.</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# UI — PAGE: ABOUT
# ---------------------------------------------------------------------------
def render_about_page():
    st.markdown('<div class="eyebrow">Project overview</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-title" style="font-size:2.1rem;">About {PROJECT_NAME}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="app-subtitle">{PROJECT_TAGLINE}</div>', unsafe_allow_html=True)
    st.markdown('<div class="flight-path"></div>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="section-label">What it does</div>', unsafe_allow_html=True)
        st.write(
            f"{PROJECT_NAME} estimates the odds that a scheduled U.S. domestic flight will land "
            "15 or more minutes behind schedule, using only the details found on a boarding "
            "pass: the date, airline, flight number, route, and scheduled departure/arrival times."
        )

    with st.container(border=True):
        st.markdown('<div class="section-label">How it works</div>', unsafe_allow_html=True)
        st.write(
            "Two gradient-boosted classifiers — XGBoost and LightGBM — were trained on the same "
            "historical flight-record dataset and feature set. Categorical fields (airports, "
            "airlines, codeshare partners, time blocks) are label-encoded before being passed "
            "to whichever model is currently active."
        )
        st.write(
            "Use the **Select model** tab to switch which one powers your forecast, and the "
            "**Model comparison** tab to see them score the same itinerary side by side."
        )

    with st.container(border=True):
        st.markdown('<div class="section-label">Limitations</div>', unsafe_allow_html=True)
        st.write(
            "Predictions reflect patterns in historical data only. They do not account for "
            "live conditions such as current weather, air-traffic control delays, mechanical "
            "issues, or crew scheduling on the day of travel — treat them as a planning signal, "
            "not a guarantee."
        )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    inject_custom_css()

    if "active_model" not in st.session_state:
        st.session_state["active_model"] = "XGBoost"

    models, encoders, load_errors = load_artifacts()
    if load_errors:
        for err in load_errors:
            st.error(err)
        st.stop()

    page = render_sidebar()

    if page == "predict":
        render_predict_page(models, encoders)
    elif page == "select_model":
        render_model_select_page(models)
    elif page == "comparison":
        render_comparison_page(models, encoders)
    elif page == "explore_data":
        render_explore_data_page()
    else:
        render_about_page()


if __name__ == "__main__":
    main()