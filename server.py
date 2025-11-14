from flask import Flask, render_template, request
from eval import main as eval_main

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    # simple index with a dropdown of known traffic types (eval will validate)
    return render_template('index.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/model')
def model():
    return render_template('model.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        t = request.form.get('traffic_type', '')
        expected = t
        t = t.strip()
        # call eval.main which returns (pred_index, prob_list, predsvm, probsvm)
        try:
            pred, prob, predsvm, probsvm, classes = eval_main(t)
        except Exception as e:
            return render_template('index.html', error=str(e))

        # build dictionaries for template (align keys to sample app's expectations)
        # classes is a list of class names in the model order
        # Build a mapping of class name -> probability
        prob_map = {classes[i]: float(prob[i]) for i in range(len(classes))}

        # For compatibility with the example app, create dict entries for each class name.
        # If a class is missing, value will be 0.0
        d = {
            'expected': expected,
            'predictions': classes[pred] if 0 <= pred < len(classes) else str(pred),
        }
        # add probabilities for each known class
        for cls in classes:
            # normalize key (no spaces)
            key = cls.replace(' ', '_')
            d[key] = prob_map.get(cls, 0.0)

        # For compatibility with the sample, create a second dict for 'svm' (same values)
        dsvm = {'expected': expected, 'predictionssvm': d['predictions']}
        for cls in classes:
            key = cls.replace(' ', '_')
            dsvm[key] = prob_map.get(cls, 0.0)

        return render_template('result.html', dict=d, dictsvm=dsvm, classes=classes)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
