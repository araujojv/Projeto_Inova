from flask import Flask, request, jsonify, send_file, render_template, redirect, url_for, flash
import pandas as pd
from pycaret.classification import setup, compare_models, tune_model, predict_model
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, tune_model as reg_tune_model, predict_model as reg_predict_model
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from datetime import datetime

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Necessário para autenticação de sessão
app.config['UPLOAD_FOLDER'] = 'uploads'

# Configuração do banco de dados PostgreSQL
DB_CONFIG = {
    "dbname": "modelo_classificacao",
    "user": "usuario_teste",
    "password": "senha123",
    "host": "localhost"
}

# Função para conectar ao banco de dados
def get_db_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    return conn

# Rota para renderizar o front-end
@app.route('/')
def index():
    return render_template('index.html')

# Rota para login (ainda mantida para testar a tela de login)
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("senha")

        # Conectar ao banco e verificar o usuário
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM usuarios WHERE username = %s", (username,))
        user = cursor.fetchone()
        conn.close()

        if user and user["senha"] == password:
            session["user_id"] = user["id"]
            session.modified = True
            flash("Login realizado com sucesso!")
            return redirect(url_for("index"))
        else:
            flash("Usuário ou senha incorretos.")

    return render_template("login.html")

# Detectar tipo de problema: Classificação ou Regressão
def detect_problem_type(df):
    target_col = df.columns[-1]
    target_values = df[target_col].unique()
    if len(target_values) <= 10 and df[target_col].dtype in ['object', 'int', 'bool']:
        return 'classification'
    else:
        return 'regression'

# Função para treinar o modelo e salvar o caminho no banco
def train_model(df, problem_type):
    target_col = df.columns[-1]
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

    if problem_type == 'classification':
        clf_setup = setup(data=train_df, target=target_col, verbose=False, fix_imbalance=True, feature_selection=True, remove_multicollinearity=True, session_id=123)
        best_model = tune_model(compare_models(sort='F1', n_select=1))
        predictions = predict_model(best_model, data=test_df)
    else:
        reg_setup = reg_setup(data=train_df, target=target_col, verbose=False, feature_selection=True, remove_multicollinearity=True, session_id=123)
        best_model = reg_tune_model(reg_compare_models(sort='R2', n_select=1))
        predictions = reg_predict_model(best_model, data=test_df)

    # Salva as previsões e a importância das features em CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    predictions_file = os.path.join(app.config['UPLOAD_FOLDER'], f'previsoes_{timestamp}.csv')
    predictions.to_csv(predictions_file, index=False)

    # Salva a importância das features se disponível
    feature_file = None
    try:
        feature_importance = pd.DataFrame(best_model.feature_importances_, index=train_df.drop(columns=[target_col]).columns, columns=['Importance'])
        feature_file = os.path.join(app.config['UPLOAD_FOLDER'], f'importancia_features_{timestamp}.csv')
        feature_importance.to_csv(feature_file)
    except AttributeError:
        print("O modelo selecionado não fornece importância das features.")
    
    # Salvar no banco de dados
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO modelos (user_id, nome_modelo, caminho_arquivo, data_treinamento)
        VALUES (%s, %s, %s, %s)
    """, (1, f"modelo_{timestamp}", predictions_file, datetime.now()))  # user_id fixo para teste
    conn.commit()
    conn.close()
    
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
    train_model(df, problem_type)  # user_id fixo removido
    return jsonify({"message": "Modelo treinado, previsões e importância das features geradas"}), 200

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=5000)
