import flask
from flask import Flask, request, render_template
from sklearn.externals import joblib
import numpy as np

from utils import onehotCategorical

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    return flask.render_template('index.html')


@app.route('/predict', methods=['POST'])
def make_prediction():
    if request.method=='POST':

        entered_li = []

        # ========== Part 2.3 ==========
        # YOUR CODE START HERE

        # get request values
        store = int(request.form['store'])
        month = int(request.form['month'])
        promo = int(request.form['promo'])
        state_holiday = int(request.form['state_holiday'])
        assortment = int(request.form['assortment'])
        day_of_the_week = int(request.form['day_of_the_week'])
        promo2 = int(request.form['promo2'])
        school_holiday = int(request.form['school_holiday'])
        store_type = int(request.form['store_type'])

        # one-hot encode categorical variables
        onehotStore = onehotCategorical(store, 945, 1)
        onehotStore_type = onehotCategorical(store_type, 4)
        onehotAssortment = onehotCategorical(assortment, 3)
        onehotState_holiday = onehotCategorical(state_holiday, 4)

        # manually specify competition distance
        comp_dist = 5458.1


        # build 1 observation for prediction
        entered_li.extend(onehotStore)
        entered_li.append(day_of_the_week)
        entered_li.append(promo)
        entered_li.extend(onehotState_holiday)
        entered_li.append(school_holiday)
        entered_li.extend(onehotStore_type)
        entered_li.extend(onehotAssortment)
        entered_li.append(comp_dist)
        entered_li.append(promo2)
        entered_li.append(month)

        # ========== End of Part 2.3 ==========

        # make prediction
        prediction = model.predict(np.array(entered_li).reshape(1, -1))
        label = str(np.squeeze(prediction.round(2)))

        return render_template('index.html', label=label)

if __name__ == '__main__':
    # load ML model
    # ========== Part 2.2 ==========
    # YOUR CODE START HERE
    # unzip rm.pkl.zip before running, was too big for github
    model = joblib.load('rm.pkl')
    # ========== End of Part 2.2 ==========
    # start API
    app.run(host='0.0.0.0', port=8000, debug=True)
