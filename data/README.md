# Data — MovieLens 100k

MovieLens 100k **não está incluído** neste repositório (restrição de redistribuição GroupLens).

## Download

1. Baixe de [grouplens.org/datasets/movielens/100k](https://grouplens.org/datasets/movielens/100k/)
2. Extraia em `data/ml-100k/`
3. Arquivos necessários: `u.data`, `u.user`, `u.item`

```bash
mkdir -p data/ml-100k
# após download:
unzip ml-100k.zip -d data/
```

## Genomics (opcional)

Para o módulo `pca_lda/` com DNA real, copie `ccrescentus.fa` do curso MITx Week1 para `data/ccrescentus.fa`.
