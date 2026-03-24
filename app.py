from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

import pandas as pd
import joblib
import os

from datetime import datetime


app = FastAPI()

# 🔥 chemins robustes (IMPORTANT Render)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

# 🔥 FIX CRITIQUE JINJA (TON ERREUR)
templates.env.cache = None
templates.env.auto_reload = True


# charger modèle ML
model = joblib.load(os.path.join(BASE_DIR, "flight_price_model.pkl"))

# charger encodeurs
label_encoders = joblib.load(os.path.join(BASE_DIR, "label_encoders.pkl"))


# colonnes catégorielles
categorical_cols = [
    "route",
    "airline_marketing",
    "cabin_class",
    "departure_time_bucket"
]


# features du modèle
features = [
    "distance_km",
    "flight_duration_min",
    "number_of_segments",
    "number_of_stops",
    "competition_level",
    "route_popularity",
    "business_route_flag",
    "days_to_departure",
    "month",
    "week_of_year",
    "day_of_week",
    "is_weekend",
    "season",
    "seats_available",
    "route",
    "airline_marketing",
    "cabin_class",
    "departure_time_bucket",
    "price_change_1d",
    "rolling_mean_price",
    "price_volatility",
    "price_momentum"
]


# routes disponibles
routes = {

("Montreal","Toronto"): {
"route":"YUL-YYZ",
"distance_km":504,
"duration":90,
"base_price":180
},

("Montreal","Vancouver"): {
"route":"YUL-YVR",
"distance_km":3680,
"duration":300,
"base_price":420
},

("Montreal","Paris"): {
"route":"YUL-CDG",
"distance_km":5525,
"duration":420,
"base_price":750
},

("Montreal","London"): {
"route":"YUL-LHR",
"distance_km":5200,
"duration":415,
"base_price":700
},

("Montreal","Casablanca"): {
"route":"YUL-CMN",
"distance_km":5600,
"duration":450,
"base_price":650
},

("London","Montreal"): {
"route":"LHR-YUL",
"distance_km":5200,
"duration":415,
"base_price":720
},

("Casablanca","Montreal"): {
"route":"CMN-YUL",
"distance_km":5600,
"duration":450,
"base_price":670
},

("Toronto","New York"): {
"route":"YYZ-JFK",
"distance_km":550,
"duration":95,
"base_price":210
}

}


# fonction recommandation
def recommendation(p):
    if p > 0.7:
        return "BUY NOW"
    elif p > 0.55:
        return "BUY"
    elif p > 0.4:
        return "WAIT"
    else:
        return "STRONG WAIT"


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/predict", response_class=HTMLResponse)
def predict(

    request: Request,

    departure_city: str = Form(...),
    arrival_city: str = Form(...),
    airline_marketing: str = Form(...),
    cabin_class: str = Form(...),
    departure_date: str = Form(...),
    departure_time: str = Form(...)
):

    try:
        # 🔥 nettoyage des inputs (CRITIQUE)
        departure_city = departure_city.strip().title()
        arrival_city = arrival_city.strip().title()
        airline_marketing = airline_marketing.strip()
        cabin_class = cabin_class.strip()

        print("DEBUG INPUT:", departure_city, arrival_city)

        # 🔒 vérification route
        route_key = (departure_city, arrival_city)

        if route_key not in routes:
            return templates.TemplateResponse(
                "index.html",
                {
                    "request": request,
                    "error": f"Route {departure_city} → {arrival_city} not available"
                }
            )

        today = datetime.now()
        dep_date = datetime.strptime(departure_date, "%Y-%m-%d")

        days_to_departure = max((dep_date - today).days, 0)

        route_info = routes[route_key]

        route = route_info["route"]
        distance = route_info["distance_km"]
        duration = route_info["duration"]
        base_price = route_info["base_price"]

        # bucket heure
        hour = int(departure_time.split(":")[0])

        if hour < 12:
            departure_bucket = "Morning"
        elif hour < 17:
            departure_bucket = "Afternoon"
        elif hour < 21:
            departure_bucket = "Evening"
        else:
            departure_bucket = "Night"

        # données modèle
        data = {

            "distance_km": distance,
            "flight_duration_min": duration,
            "number_of_segments": 1,
            "number_of_stops": 0,
            "competition_level": 3,
            "route_popularity": 5,
            "business_route_flag": 0,
            "days_to_departure": days_to_departure,
            "month": dep_date.month,
            "week_of_year": dep_date.isocalendar()[1],
            "day_of_week": dep_date.weekday(),
            "is_weekend": 1 if dep_date.weekday() >= 5 else 0,
            "season": 1,
            "seats_available": 25,
            "route": route,
            "airline_marketing": airline_marketing,
            "cabin_class": cabin_class,
            "departure_time_bucket": departure_bucket,
            "price_change_1d": 0.02,
            "rolling_mean_price": base_price,
            "price_volatility": 0.1,
            "price_momentum": 0.03

        }

        df = pd.DataFrame([data])

        # encodage
        for col in categorical_cols:

            le = label_encoders[col]

            df[col] = df[col].astype(str)

            df[col] = df[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )

        X = df[features]

        # prédiction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]

        decision = recommendation(probability)

        # prix
        price_eur = base_price * (1 + probability * 0.3)

        # conversion USD
        usd_rate = 1.59
        price_usd = price_eur * usd_rate

        return templates.TemplateResponse(

            "index.html",

            {
                "request": request,
                "price": round(price_eur, 2),
                "price_usd": round(price_usd, 2),
                "probability": round(probability, 2),
                "decision": decision
            }

        )

    except Exception as e:
        print("ERROR:", str(e))

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "error": "Internal error - check logs"
            }
        )