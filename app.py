from flask import Flask, request, jsonify, send_file
import pandas as pd
from pycaret.classification import setup, compare_models, tune_model, predict_model
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, tune_model as reg_tune_model, predict_model as reg_predict_model
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Detectar tipo de problema: Classificação ou Regressão
def detect_problem_type(df):
    target_col = df.columns[-1]
    target_values = df[target_col].unique()
    if len(target_values) <= 10 and df[target_col].dtype in ['object', 'int', 'bool']:
        return 'classification'
    else:
        return 'regression'

# Função para treinar o modelo e salvar previsões e importância de features
def train_model(df, problem_type):
    target_col = df.columns[-1]
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

    if problem_type == 'classification':
        clf_setup = setup(data=train_df, target=target_col, verbose=False, fix_imbalance=True, feature_selection=True, remove_multicollinearity=True)
        best_model = tune_model(compare_models(sort='F1', n_select=1))
        predictions = predict_model(best_model, data=test_df)
    else:
        reg_setup = reg_setup(data=train_df, target=target_col, verbose=False, feature_selection=True, remove_multicollinearity=True)
        best_model = reg_tune_model(reg_compare_models(sort='R2', n_select=1))
        predictions = reg_predict_model(best_model, data=test_df)
    
    # Salvar previsões em CSV
    predictions.to_csv('previsoes_modelo.csv', index=False)
    
    # Salvar importância das features em CSV (se disponível)
    try:
        feature_importance = pd.DataFrame(best_model.feature_importances_, index=train_df.drop(columns=[target_col]).columns, columns=['Importance'])
        feature_importance.to_csv('importancia_features.csv')
    except AttributeError:
        print("O modelo selecionado não fornece importância das features.")
    
    return predictions

# Rota para receber o CSV e treinar o modelo
@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({"message": "Nenhum arquivo enviado"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "Nenhum arquivo selecionado"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return jsonify({"message": f"Erro ao processar o arquivo: {str(e)}"}), 400

    problem_type = detect_problem_type(df)
    train_model(df, problem_type)
    return jsonify({"message": "Modelo treinado, previsões e importância das features geradas"}), 200

# Rota para retornar as previsões em CSV
@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    if os.path.exists('previsoes_modelo.csv'):
        return send_file('previsoes_modelo.csv', as_attachment=True)
    else:
        return jsonify({"message": "Previsões não encontradas. Envie o CSV para treinar o modelo primeiro."}), 404

# Rota para retornar a importância das features em CSV
@app.route('/get_feature_importance', methods=['GET'])
def get_feature_importance():
    if os.path.exists('importancia_features.csv'):
        return send_file('importancia_features.csv', as_attachment=True)
    else:
        return jsonify({"message": "Importância das features não encontrada. Envie o CSV para treinar o modelo primeiro."}), 404

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)

