from flask import Flask, request, render_template, redirect, url_for, flash, send_file
from pycaret.classification import (
    setup as clf_setup, compare_models as clf_compare_models,
    save_model as clf_save_model, load_model as clf_load_model,
    predict_model as clf_predict_model
)
from pycaret.regression import (
    setup as reg_setup, compare_models as reg_compare_models,
    save_model as reg_save_model, load_model as reg_load_model,
    predict_model as reg_predict_model
)
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'modelos_salvos'
app.config['PREDICTION_FOLDER'] = 'predicoes'
app.config['STATIC_FOLDER'] = 'static'

# Garantir que as pastas necessárias existam
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PREDICTION_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

def sanitize_model_name(model_name):
    """Sanitiza o nome do arquivo do modelo."""
    if model_name.endswith('.pkl.pkl'):
        return model_name.replace('.pkl.pkl', '.pkl')
    return model_name

@app.route('/')
def index():
    """Página inicial para treinar modelos."""
    try:
        modelos_salvos = os.listdir(app.config['UPLOAD_FOLDER'])
    except Exception as e:
        flash(f"Erro ao listar os modelos salvos: {e}")
        modelos_salvos = []
    return render_template('index.html', modelos=modelos_salvos)

@app.route('/train_model', methods=['POST'])
def train_model():
    """Treina um modelo baseado em um arquivo CSV enviado pelo cliente."""
    model_type = request.form.get("model_type")
    metric = request.form.get("metrics")
    csv_file = request.files.get("file")

    if not csv_file:
        flash("Por favor, envie um arquivo CSV para treinar o modelo.")
        return redirect(url_for("index"))

    try:
        df = pd.read_csv(csv_file)
        if 'target' not in df.columns:
            flash("O arquivo CSV deve conter uma coluna chamada 'target' (variável dependente).")
            return redirect(url_for("index"))

        # Configuração e treinamento do modelo
        if model_type == "classification":
            clf_setup(data=df, target='target', verbose=False)
            best_model = clf_compare_models(sort=metric)
        elif model_type == "regression":
            reg_setup(data=df, target='target', verbose=False)
            best_model = reg_compare_models(sort=metric)
        else:
            flash("Tipo de modelo inválido.")
            return redirect(url_for("index"))

        # Salvamento do modelo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"{model_type}_model_{timestamp}.pkl"
        if model_type == "classification":
            clf_save_model(best_model, os.path.join(app.config['UPLOAD_FOLDER'], model_name.replace('.pkl', '')))
        elif model_type == "regression":
            reg_save_model(best_model, os.path.join(app.config['UPLOAD_FOLDER'], model_name.replace('.pkl', '')))

        flash(f"Modelo treinado e salvo com sucesso! Arquivo: {model_name}")
    except Exception as e:
        flash(f"Erro durante o treinamento do modelo: {e}")

    return redirect(url_for("index"))

@app.route('/delete_model/<path:model_name>')
def delete_model(model_name):
    """Permite a exclusão de um modelo salvo."""
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_name)
    try:
        if os.path.exists(model_path):
            os.remove(model_path)
            flash(f"Modelo '{model_name}' excluído com sucesso!")
        else:
            flash(f"Modelo '{model_name}' não encontrado.")
    except Exception as e:
        flash(f"Erro ao excluir o modelo: {e}")
    return redirect(url_for("index"))

@app.route('/download_model/<path:model_name>')
def download_model(model_name):
    """Permite o download de um modelo salvo."""
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_name)
    try:
        if os.path.exists(model_path):
            return send_file(model_path, as_attachment=True)
        else:
            flash(f"Modelo '{model_name}' não encontrado.")
            return redirect(url_for("index"))
    except Exception as e:
        flash(f"Erro ao tentar baixar o modelo: {e}")
        return redirect(url_for("index"))

@app.route('/predict')
def predict():
    """Página para realizar previsões."""
    try:
        modelos_salvos = os.listdir(app.config['UPLOAD_FOLDER'])
        prediction_files = os.listdir(app.config['PREDICTION_FOLDER'])
        return render_template('predict.html', modelos=modelos_salvos, prediction_files=prediction_files)
    except Exception as e:
        flash(f"Erro ao carregar modelos ou previsões: {e}")
        return redirect(url_for("index"))

@app.route('/run_prediction', methods=['POST'])
def run_prediction():
    """Realiza previsões e salva o resultado em um arquivo CSV."""
    model_name = sanitize_model_name(request.form.get('model_name'))
    csv_file = request.files.get('file')

    if not model_name or not csv_file:
        flash("Selecione um modelo e envie um arquivo CSV para realizar a previsão.")
        return redirect(url_for("predict"))

    model_path = os.path.join(app.config['UPLOAD_FOLDER'], model_name)

    try:
        # Carregar o modelo correto
        if "classification" in model_name:
            model = clf_load_model(model_path.replace('.pkl', ''))
        elif "regression" in model_name:
            model = reg_load_model(model_path.replace('.pkl', ''))
        else:
            flash("Modelo inválido.")
            return redirect(url_for("predict"))

        # Ler o CSV de entrada
        df = pd.read_csv(csv_file)

        # Fazer previsões com o PyCaret
        if "classification" in model_name:
            predictions = clf_predict_model(model, data=df)
        elif "regression" in model_name:
            predictions = reg_predict_model(model, data=df)

        # Salvar os resultados da previsão
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prediction_filename = f"prediction_{timestamp}.csv"
        prediction_file = os.path.join(app.config['PREDICTION_FOLDER'], prediction_filename)

        predictions.to_csv(prediction_file, index=False)

        flash(f"Previsão concluída! Arquivo salvo como {prediction_filename}.")
        return redirect(url_for("predict", model_name=model_name))
    except Exception as e:
        flash(f"Erro ao realizar previsão: {e}")
        return redirect(url_for("predict"))

@app.route('/download_prediction/<path:filename>')
def download_prediction(filename):
    """Permite o download de um arquivo de previsão."""
    file_path = os.path.join(app.config['PREDICTION_FOLDER'], filename)
    try:
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            flash("Arquivo de previsão não encontrado.")
            return redirect(url_for("predict"))
    except Exception as e:
        flash(f"Erro ao tentar baixar a previsão: {e}")
        return redirect(url_for("predict"))

@app.route('/visualize/<filename>')
def visualize(filename):
    """Visualiza os dados de previsão com gráficos."""
    file_path = os.path.join(app.config['PREDICTION_FOLDER'], filename)
    try:
        df = pd.read_csv(file_path)

        # Resumir a tabela para mostrar apenas as 5 primeiras linhas
        table_data = df.head(5).to_html(classes='table table-striped table-bordered', index=False)

        # Gerar gráfico estático com Seaborn
        static_plot_path = os.path.join(app.config['STATIC_FOLDER'], 'static_plot.png')
        plt.figure(figsize=(6, 4))  # Tamanho ajustado
        if 'Label' in df.columns or 'prediction_label' in df.columns:
            label_col = 'Label' if 'Label' in df.columns else 'prediction_label'
            sns.countplot(x=label_col, data=df, palette='viridis')
            plt.title('Distribuição das Previsões (Labels)')
        else:
            sns.histplot(df.iloc[:, 0], kde=True, color='blue')
            plt.title('Histograma dos Dados Previstos')
        plt.savefig(static_plot_path)
        plt.close()

        # Gerar gráfico de pizza com Matplotlib
        pie_plot_path = os.path.join(app.config['STATIC_FOLDER'], 'pie_chart.png')
        if 'Label' in df.columns or 'prediction_label' in df.columns:
            label_col = 'Label' if 'Label' in df.columns else 'prediction_label'
            label_counts = df[label_col].value_counts()
            plt.figure(figsize=(6, 4))  # Tamanho ajustado
            plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', colors=['#4CAF50', '#FFC107'])
            plt.title('Distribuição de Labels')
            plt.savefig(pie_plot_path)
            plt.close()

        # Gerar gráfico interativo com Plotly
        if 'Label' in df.columns or 'prediction_label' in df.columns:
            label_col = 'Label' if 'Label' in df.columns else 'prediction_label'
            fig = px.bar(df, x=label_col, y=df.index, title='Gráfico de Barras - Previsões', height=300)
        else:
            fig = px.line(df, y=df.iloc[:, 0], title='Gráfico de Linhas - Dados Previstos', height=300)

        interactive_plot = fig.to_html(full_html=False)

        return render_template(
            'visualize.html',
            table=table_data,
            static_plot_path=static_plot_path,
            pie_plot_path=pie_plot_path,
            interactive_plot=interactive_plot
        )
    except Exception as e:
        flash(f"Erro ao visualizar o arquivo: {e}")
        return redirect(url_for('predict'))
    
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
