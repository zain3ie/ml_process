import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


def country_mapping(country):
    if country == 'Soviet Union':
        return 'russia'
    elif country == 'European Space Agency':
        return 'other'
    elif country == 'Israel':
        return 'other'
    elif country == 'Luxembourg':
        return 'other'
    elif country == 'United Arab Emirates':
        return 'other'
    else:
        return country.lower()


def load_data(path):
    '''
    function to load artifitial object data

    parameters
    ----------
    path : str
        path of artifitial object data (.csv)

    returns
    -------
    df : pandas dataframe
        artifitial object data
    '''

    # load data
    df = pd.read_csv(path, index_col=0)

    # baris terkahir bukan merupakan data artifitial object
    df = df[:-1]

    print('original data shape:', df.shape)

    # dalam project ini kita akan menggunakan berat dengan satuan kg
    # data yang beratnya berupa perkiraan (< atau >), kita gunakan berat perkiraannya
    # data pada id 70, jika konversi lb ke kg, maka data yang benar data yang diluar kurung
    df['Mass'] = df['Mass (kg)'].str.replace('<|>|,|\[76\]', '', regex=True)

    # ubah tipe data column 'Mass' menjadi int
    df['Mass'] = df.Mass.astype(int)

    # country mapping
    df['Country'] = df.Country.apply(lambda x: country_mapping(x))
    contry = df.Country.str.get_dummies()

    # mengelompokkan setiap 10 tahun dan menjadikan decade petama = 0
    df['Decade'] = df.Year.apply(lambda x: int(x) // 10 - 195)

    # membuat data 'survival' berdasarkan data 'status'
    df['Survival'] = df.Status.apply(lambda x: 1-int('crashed' in str(x).lower()))

    df = contry.join(df[['Decade', 'Mass', 'Survival']])
    print('final data shape:', df.shape)

    # generate feature and label
    y = df['Survival']
    X = df.drop('Survival', axis=1)

    return X, y


def search_best_model(X, y, model, param_grid, cv, scoring):
    # buat objek gridsearchcv dengan model, parameter grid, dan metrik evaluasi
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring)

    # lakukan pencarian parameter pada data dengan validasi silang
    grid_search.fit(X, y)

    # tampilkan parameter terbaik
    print('parameter terbaik:', grid_search.best_params_)

    # hitung akurasi model dengan cross_val_score menggunakan parameter terbaik
    best_model = grid_search.best_estimator_
    accuracy_scores = cross_val_score(best_model, X, y, cv=cv, scoring=scoring)

    # tampilkan akurasi rata-rata dari cross_val_score
    print('akurasi rata-rata:', accuracy_scores.mean())

    return best_model


def rf_classifier_modelling(X, y):
    print('========================')
    print('Random forest classifier')
    model = RandomForestClassifier()

    # daftar parameter
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20]
    }

    # mencari model terbaik
    best_model = search_best_model(X, y, model, param_grid, cv=5, scoring='accuracy')

    # save model
    joblib.dump(best_model, 'model/rfc_model.pkl')


def knn_classifier_modelling(X, y):
    print('=========================')
    print('k-NN classifier')
    model = KNeighborsClassifier()

    # daftar parameter
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance']
    }

    # mencari model terbaik
    best_model = search_best_model(X, y, model, param_grid, cv=5, scoring='accuracy')

    # save model
    joblib.dump(best_model, 'model/knn_model.pkl')


def main():
    X, y = load_data('data/List of Artificial Objects on the Moon.csv')
    rf_classifier_modelling(X, y)
    knn_classifier_modelling(X, y)

if __name__ == '__main__':
    main()
