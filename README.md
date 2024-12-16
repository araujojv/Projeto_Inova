

```markdown
# Visualização de Previsões com Flask e PyCaret

Este projeto utiliza Flask, PyCaret, Matplotlib, Seaborn e Plotly para realizar treinamento de modelos de machine learning e gerar previsões. Além disso, exibe gráficos interativos e estáticos baseados nos resultados.

## Funcionalidades

- Treinar modelos de classificação e regressão com PyCaret.
- Realizar previsões usando modelos treinados.
- Visualizar os resultados das previsões com tabelas e gráficos:
  - Gráfico de barras (estático).
  - Gráfico de pizza (estático).
  - Gráfico interativo (gerado com Plotly).
- Download dos arquivos de previsão gerados.

---

## Tecnologias Utilizadas

- Python
- Flask
- PyCaret
- Matplotlib
- Seaborn
- Plotly
- Bootstrap (para estilização)

---

## Pré-requisitos

1. **Python 3.8 ou superior**.
2. **Bibliotecas necessárias**: listadas no arquivo `requirements.txt`.

---

## Instalação e Configuração

1. **Clone o repositório**:
   ```bash
   git clone https://github.com/araujojv/Projeto_Inova.git
   cd seu-repositorio
   ```

2. **Crie um ambiente virtual (opcional, mas recomendado)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Instale as dependências**:
   Certifique-se de ter o arquivo `requirements.txt` na raiz do projeto.
   ```bash
   pip install -r requirements.txt
   ```

---

## Como Usar

1. **Execute o aplicativo Flask**:
   ```bash
   python app.py
   ```

2. **Acesse o aplicativo no navegador**:
   O servidor estará disponível em: [http://127.0.0.1:5000](http://127.0.0.1:5000)

3. **Passos no aplicativo**:
   - **Treinar Modelos**: Faça o upload de um arquivo CSV com a coluna `target` para treinar um modelo.
   - **Fazer Previsões**:
     1. Escolha um modelo treinado.
     2. Faça o upload de um arquivo CSV (sem a coluna `target`).
   - **Visualizar Previsões**:
     - Acesse a página de visualização para ver os resultados das previsões em tabelas e gráficos.
   - **Download**:
     - Faça o download dos arquivos gerados na seção de previsões disponíveis.

---

## Estrutura do Projeto

```plaintext
.
├── app.py                  # Arquivo principal com o código do servidor Flask
├── requirements.txt        # Dependências do projeto
├── templates/              # Arquivos HTML
│   ├── index.html          # Página inicial
│   ├── predict.html        # Página para fazer previsões
│   └── visualize.html      # Página para visualizar previsões
├── static/                 # Arquivos estáticos (gráficos gerados)
│   ├── static_plot.png     # Gráfico de barras estático
│   ├── pie_chart.png       # Gráfico de pizza
└── modelos_salvos/         # Modelos treinados salvos
└── predicoes/              # Arquivos de previsão gerados
```

---

## Exemplo de Arquivo CSV

### Para Treinar Modelos:
- **Formato do CSV**: Deve conter uma coluna chamada `target` (variável dependente).
```csv
feature1,feature2,feature3,target
0.64,1.1,A,Yes
0.03,7.5,B,No
0.28,7.1,B,Yes
```

### Para Fazer Previsões:
- **Formato do CSV**: Sem a coluna `target`.
```csv
feature1,feature2,feature3
0.64,1.1,A
0.03,7.5,B
0.28,7.1,B
```

---



