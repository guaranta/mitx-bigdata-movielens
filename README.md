# mitx-bigdata-movielens

**MITx xPro — Data Science and Big Data Analytics**

Big Data na prática: filtragem colaborativa (MovieLens 100k) e redução de dimensionalidade com PCA/k-means (caso genômico).

| Módulo | Conteúdo | Comando |
|--------|----------|---------|
| `movielens/` | User-based collaborative filtering | `python movielens/run.py` |
| `pca_lda/` | k-mer frequencies → PCA → k-means | `python pca_lda/run.py` |
| `evaluation/` | RMSE, MAE, silhouette, ARI | `from evaluation.metrics import ...` |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# MovieLens (obrigatório para movielens/)
# Ver data/README.md para download

python pca_lda/run.py          # funciona sem dados externos
python movielens/run.py        # requer data/ml-100k/u.data
```

## Origem acadêmica

- **MovieLens:** evolução de `teste_padas.py` do curso xPro
- **PCA/k-means:** port Python de `CalcFreq.m`, `PCAFreq.m`, `ClustFreq.m` (Week 1 Case Study)

## Portfólio

- [Portfolio AI Engineer / CTO](https://portfolio-ai-cto-guaranta.netlify.app)
- [Métricas RMSE e laboratório](docs/portfolio-link.md)

## Autor

**Guarantã Almeida** — [github.com/guaranta](https://github.com/guaranta)
