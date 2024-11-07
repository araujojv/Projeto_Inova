from flask import Flask, render_template, request, redirect, url_for, flash, session, send_file
import requests
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Necessário para exibir mensagens flash e autenticação de sessão
API_URL = "http://127.0.0.1:5000"  # URL da sua API backend

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

# Rota para login
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        senha = request.form.get("senha")  # Corrigido para "senha" ao invés de "password"

        # Conectar ao banco e verificar o usuário
        conn = get_db_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT * FROM usuarios WHERE username = %s", (username,))
        user = cursor.fetchone()
        conn.close()

        # Verifique se o usuário existe e a senha está correta
        if user and user["senha"] == senha:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash("Login realizado com sucesso!")
            return redirect(url_for("index"))
        else:
            flash("Usuário ou senha incorretos.")
            return redirect(url_for("login"))

    return render_template("login.html")

# Rota para logout
@app.route("/logout")
def logout():
    session.clear()
    flash("Você saiu da sessão.")
    return redirect(url_for("login"))

# Decorador para verificar se o usuário está logado
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "user_id" not in session:
            flash("Por favor, faça login primeiro.")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrap

# Rota principal para o formulário de upload
@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        # Verifica se o arquivo foi enviado
        if "file" not in request.files or request.files["file"].filename == "":
            flash("Nenhum arquivo selecionado!")
            return redirect(url_for("index"))

        file = request.files["file"]
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        filepath = os.path.join("uploads", filename)
        file.save(filepath)

        # Envia o arquivo para a API /upload_csv
        response = requests.post(f"{API_URL}/upload_csv", files={"file": open(filepath, "rb")})
        if response.status_code == 200:
            flash("Arquivo enviado e modelo treinado com sucesso!")

            # Registra o modelo no banco de dados
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO modelos (user_id, nome_modelo, caminho_arquivo) VALUES (%s, %s, %s)",
                (session["user_id"], filename, filepath)
            )
            conn.commit()
            conn.close()
        else:
            flash("Falha ao treinar o modelo.")
        return redirect(url_for("index"))

    # Obter os modelos treinados pelo usuário logado
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT * FROM modelos WHERE user_id = %s", (session["user_id"],))
    modelos = cursor.fetchall()
    conn.close()

    return render_template("index.html", modelos=modelos)

# Rota para baixar o arquivo de um modelo específico
@app.route("/download_modelo/<int:modelo_id>")
@login_required
def download_modelo(modelo_id):
    # Obter o modelo do banco de dados
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT * FROM modelos WHERE id = %s AND user_id = %s", (modelo_id, session["user_id"]))
    modelo = cursor.fetchone()
    conn.close()

    if modelo:
        return send_file(modelo["caminho_arquivo"], as_attachment=True)
    else:
        flash("Modelo não encontrado.")
        return redirect(url_for("index"))

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(port=5001, debug=True)
