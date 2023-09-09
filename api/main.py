import joblib
import pandas as pd
import requests

from fastapi import FastAPI, HTTPException


app = FastAPI(
    title='Artifitial Moon Object Survival Prediction',
    docs_url='/',
)


@app.post('/predict')
def prediction(
    country: str,
    year: int,
    mass: int
):
    # ambil list negara
    response = requests.get('https://restcountries.com/v3.1/all')
    response_json = response.json()
    countries = [country['name']['common'].lower() for country in response_json]

    country = country.lower()

    # input validation
    if country not in countries:
        raise HTTPException(
            status_code=422,
            detail='Harap periksa nama negara pengirim'
        )

    if year < 1950 or year > 2100:
        raise HTTPException(
            status_code=422,
            detail='Harap masukkan tahun peluncuran antara 1959-2023'
        )

    if mass <= 0:
        raise HTTPException(
            status_code=422,
            detail='Harap masukkan berat objek'
        )

    # data pipeline
    experienced_countries = ['united states', 'russia' , 'china', 'japan', 'india']
    if country not in experienced_countries:
        country = 'other'

    # mengelompokkan setiap 10 tahun dan kurangi decade tahun pertama (1959)
    decade = year // 10 - 195

    data = {
        'Decade': [decade],
        'Mass': [mass]
    }

    # convert dalam bentuk dataframe
    df = pd.DataFrame(data)

    for country_col in (experienced_countries + ['other']):
        df[country_col] = 1 if country == country_col else 0

    # urutkan feature sesuai dengan saat training
    df = df[['china', 'india', 'japan', 'other', 'russia', 'united states', 'Decade', 'Mass']]
    print(df)

    # load model
    model = joblib.load('model/knn_model.pkl')

    # prediksi
    prediction = model.predict(df)
    result = prediction.tolist()[0]

    return result
