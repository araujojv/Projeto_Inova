import pandas as pd
from pycaret.classification import setup, compare_models, tune_model, save_model, predict_model
from pycaret.regression import setup as reg_setup, compare_models as reg_compare_models, tune_model as reg_tune_model, predict_model as reg_predict_model
from sklearn.model_selection import train_test_split

# Função para detectar automaticamente o tipo de problema: Classificação ou Regressão
def detect_problem_type(df):
    target_col = df.columns[-1]  # Última coluna como target
    target_values = df[target_col].unique()

    # Se o target for binário ou categórico, assumimos que é um problema de classificação
    if len(target_values) <= 10 and df[target_col].dtype in ['object', 'int', 'bool']:
        return 'classification'
    else:
        return 'regression'

# Função para treinar o modelo automaticamente e salvar as previsões em um CSV
def train_and_save_predictions(df, problem_type):
    target_col = df.columns[-1]
    
    # Dividir o conjunto de dados em treino (75%) e teste (25%)
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=42)

    # Verificação de dados
    print(f"Tamanho dos dados de treino: {train_df.shape}")
    print(f"Tamanho dos dados de teste: {test_df.shape}")

    if problem_type == 'classification':
        print("Iniciando setup de classificação...")
        clf_setup = setup(
            data=train_df, target=target_col, verbose=True,
            fix_imbalance=True,  # Ajusta o desbalanceamento de classes
            feature_selection=True,  # Ativa a seleção automática de features
            remove_multicollinearity=True,  # Remove colinearidade
        )
        best_model = tune_model(compare_models(sort='F1', n_select=1))  # Seleciona o melhor modelo com F1-score
    else:
        print("Iniciando setup de regressão...")
        reg_setup = reg_setup(
            data=train_df, target=target_col, verbose=True,
            feature_selection=True,  # Ativa a seleção automática de features
            remove_multicollinearity=True,  # Remove colinearidade
        )
        best_model = reg_tune_model(reg_compare_models(sort='R2', n_select=1))  # Seleciona o melhor modelo com R2

    # Gerar previsões com o modelo
    if problem_type == 'classification':
        predictions = predict_model(best_model, data=test_df)  # Previsões com o modelo
    else:
        predictions = reg_predict_model(best_model, data=test_df)  # Previsões com o modelo

    # Salvar o modelo treinado
    save_model(best_model, 'best_model')
    
    # Salvar as previsões em um arquivo CSV
    predictions.to_csv('previsoes_modelo.csv', index=False)
    print(f"As previsões foram salvas em 'previsoes_modelo.csv'")

# Função principal para rodar localmente
def main():
    # Pedir ao usuário o caminho do arquivo CSV
    csv_path = input("Insira o caminho do arquivo CSV: ").strip()  # Remover espaços extras

   
    print(f"Tentando abrir o arquivo: {csv_path}")

    # Tentar ler o arquivo CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {csv_path}")
        return
    except pd.errors.EmptyDataError:
        print("O arquivo está vazio.")
        return
    except pd.errors.ParserError:
        print("Erro ao processar o arquivo CSV.")
        return

    # Detectar o tipo de problema
    problem_type = detect_problem_type(df)
    print(f"Tipo de problema detectado: {problem_type}")

    # Treinar o modelo e salvar previsões em CSV
    train_and_save_predictions(df, problem_type)

if __name__ == "__main__":
    main()
