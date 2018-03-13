
**********************************
CI172

Processamento de Imagens Biomédicas

Primeiro semestre de 2017

1º trabalho - Top-hat  
**********************************

**Base:**
  - http://web.inf.ufpr.br/vri/alumni/2015-LuizaDriBagesteiro-Msc

**Dados:**
  - http://web.inf.ufpr.br/lferrari/ci172-2017-1/lung_blocks_exemplo_dcm.tar.gz

**Extrator de características:**
- Top-Hat


**Comentários:**

- White top-hat: imagemOriginal - abertura(imagemOriginal)
- Black top-hat: fechamento(imagemOriginal) - imagemOriginal

**Características:**
- Cada pixel: C(x,y) = max{ W(x,y), 0 }
- Cada pixel: C(x,y) = max{ 0, B(x,y) }
    

+ 4 valores de caracteristicas: media white top-hat, desvio padrao white top-hat, media black top-hat, desvio padrao black top-hat

** https://www.osapublishing.org/view_article.cfm?gotourl=https%3A%2F%2Fwww%2Eosapublishing%2Eorg%2FDirectPDFAccess%2F8CEF7AF3-B49B-5518-DE9B37C8AEB19528_253926%2Fao-52-16-3777%2Epdf%3Fda%3D1%26id%3D253926%26seq%3D0%26mobile%3Dno&org=Universidade%20Federal%20do%20Parana
??

***************************************************************

**Descricao arquivos:**
  - projeto/percorre.sh: Roda o executavel build/projeto para todos os arquivos .dcm dentro do diretorio passado como parametro
    - Ex.: bash percorre.sh /nobackup/ibm/cdp13/cdp13/nobackup/7-semestre/pib/lung_blocks_exemplo_dcm/
    - ( obs.: passar caminho completo para o diretorio com as imagens )



****************************************************************

**O que foi utilizado?**
  
  - Extratores: White Top-hat (WTH) e Black Top-hat (BTH)
    - Vetor de características: Media WTH, Desvio padrão WTH, Media BTH, Desvio padrão BTH

  - Classificadores: KNN, Arvore de decisão, Naive Bayes Gaussiano, SVM, Logistic Reqression Ovr, Logistic Regression Mul, LDA



