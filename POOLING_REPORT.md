# Alternativas ao mean-pooling — relatório

**Data:** 2026-09-30
**Motivação:** três medições independentes convergiram para o mesmo diagnóstico — o
classificador *localiza* os resíduos catalíticos, mas a agregação por média descarta
essa informação antes de ela chegar à cabeça.

---

## 1. A evidência que motivou a mudança

### 1.1 Validação externa (18.276 sequências, 6 grupos, nenhuma homóloga ao treino)

| | MCC interno | MCC externo (macro) |
|---|---|---|
| LoRA r=8 | 0,955 | **0,186** |
| frozen + MLP | 0,704 | **0,056** |

O colapso é seletivo: nos grupos de **mutantes pontuais** (Parkin, PLpro, BRCA1, E4B) o
MCC vai a 0 — o modelo prediz uma classe constante. Nos grupos de **sequências
desenhadas** (chorismate mutase) ele funciona (MCC 0,38). Varrer a cabeça de 128 a
6×2560 neurônios não mudou nada, em nenhum dos dois backbones.

### 1.2 Geometria dos embeddings

Medido sobre os embeddings mean-pooled do benchmark:

| grupo | dispersão interna | separação ativo/inativo |
|---|---|---|
| Parkin (mutantes pontuais) | 0,064 | 0,025 |
| PLpro (mutantes pontuais) | 0,108 | 0,021 |
| Chorismate mutase (desenhadas) | **1,706** | **0,796** |
| **distância entre proteínas diferentes** | **3,388** | |

Uma mutação pontual desloca o vetor agregado em ~0,06; proteínas diferentes distam 3,39
— **fator ~50×**. A fronteira de decisão foi aprendida na escala maior; o sinal dos
mutantes fica abaixo da sua resolução.

### 1.3 Atribuição por resíduo (200 enzimas SwissProt, 313 sítios catalíticos)

| medida | valor | leitura |
|---|---|---|
| enriquecimento no top-10% (camada 33) | **7,41×** | o modelo **localiza** os sítios catalíticos |
| controle: mesmo aminoácido, catalítico vs não | **+0,29 a +0,41** de rank | não é viés de composição — usa contexto |
| fração do logit vinda dos sítios catalíticos | **0,73%** | mas eles quase não pesam |
| razão massa/proporção | 2,1× | cada sítio pesa só 2× um resíduo médio |

**Esta é a medida decisiva.** O enriquecimento de 7,41× é *ordinal* (está no topo do
ranking); os 0,73% são *cardinais* (quanto do sinal ele carrega). Sob
`pooled = (1/N)·Σ h[l]`, cada resíduo entra com peso fixo 1/465 **independentemente de
importar ou não**. O máximo que o modelo consegue é alinhar melhor o `h[l]` daquele
resíduo com o gradiente — o fator 2,1× medido. E 2,1 × (1/465) continua sendo ~0,5% do
total.

O controle por aminoácido foi necessário porque His, Cys e Asp são 62% dos sítios
catalíticos mas apenas 9,5% do fundo: ranquear só por identidade de aminoácido já dá
6,17× de enriquecimento. Comparando **dentro do mesmo aminoácido**, o confundidor
desaparece — e a separação persiste, então o sinal é genuinamente contextual.

---

## 2. O que foi implementado

Quatro modos de agregação em `model/lora_classifier.py`, selecionáveis por
`cfg.head.pooling` ou `--pooling`:

| modo | fórmula | parâmetros extras | entrada da cabeça |
|---|---|---|---|
| **`mean`** (padrão) | `Σ (1/N)·h[l]` | 0 | 1280 |
| **`attention`** | `Σ w[l]·h[l]`, `w = softmax(Linear(h))` | **1.281** | 1280 |
| **`max`** | `max_l h[l]` por dimensão | 0 | 1280 |
| **`mean_max`** | `concat(mean, max)` | 0 | **2560** |

### Por que estes quatro

- **`attention`** ataca diretamente o mecanismo medido: troca o peso fixo 1/N por um peso
  aprendido. Se o modelo já identifica o sítio catalítico — e a evidência diz que sim —
  ele **pode** atribuir peso 0,3 em vez de 0,002. Custa 1.281 parâmetros (0,0002% do modelo).
- **`max`** preserva o pico local sem nenhum parâmetro: um resíduo com valor extremo numa
  dimensão sobrevive à agregação em vez de ser dissolvido.
- **`mean_max`** combina contexto global (média) e pico local (máximo). É uma prática comum
  em agregação de sequências e serve de ablação intermediária entre as duas.
- **`mean`** permanece o padrão para que tudo já treinado continue reproduzível.

### Detalhes de implementação

**Máscara de resíduos.** Extraí `residue_mask()` como função única, usada pelos quatro
modos: zera `<cls>`, `<eos>` e padding. Antes essa lógica estava embutida no
`mean_pool_residues`; agora é compartilhada, garantindo que os modos novos excluam
exatamente os mesmos tokens que os runs anteriores.

**Segurança numérica em fp16.** O mascaramento usa `torch.finfo(dtype).min` em vez de
`-inf`. Sob autocast fp16, `-inf` no softmax ou no max propaga NaN. Verificado: nenhum NaN
nos dois modos com tensores half.

**Dimensão da cabeça.** `mean_max` concatena dois vetores, então a primeira camada da
cabeça passa a receber 2560 em vez de 1280. Isso é resolvido no construtor
(`head_in = hidden_size * 2 if pooling == "mean_max"`), não pelo usuário.

**O pooler treina junto com a cabeça.** Em `param_groups()`, os parâmetros do
`AttentionPool` entram no grupo da cabeça — ou seja, com `lr_head` (1e-3) e
`weight_decay_head` (0,05), não com a taxa menor dos adapters.

**Inspeção.** `AttentionPool.weights()` devolve a distribuição de atenção. Isso permite
medir diretamente, depois do treino, se o peso se concentrou nos sítios catalíticos —
a mesma pergunta do XAI, mas respondida pelo próprio mecanismo de pooling em vez de por
gradiente.

### Retrocompatibilidade — verificada

| verificação | resultado |
|---|---|
| default de `cfg.head.pooling` | `"mean"` |
| `lora_r8/config.json` e `lora_r8_posw/config.json` têm a chave? | não → caem no default |
| chaves de pooler nos checkpoints antigos | nenhuma (correto) |

Os checkpoints existentes continuam carregando sem alteração.

### Validação numérica

```
pesos de atenção somam 1                        OK  [1.0, 1.0]
peso em <cls>/<eos>/padding                     OK  0.00e+00
peso máximo possível com mean pooling           0,1000  (fixo, = 1/N)
peso máximo com attention                       1,0000  (pode concentrar)
NaN em fp16 (attention / max)                   OK  não
```

A penúltima linha é o ponto todo: o mean-pooling **não consegue** dar mais de 1/N a
nenhum resíduo; a atenção consegue.

---

## 3. Como rodar

```bash
cd ~/ESM2LoRa   # ou /scratch/izar/josartor/ESM2LoRa

sbatch submit_izar_lora.slurm lora_attnpool  --pooling attention
sbatch submit_izar_lora.slurm lora_meanmax   --pooling mean_max
sbatch submit_izar_lora.slurm lora_maxpool   --pooling max
```

Arquivos a enviar:

```bash
rsync -avz ESM2LoRa/model/lora_classifier.py ESM2LoRa/config.py ESM2LoRa/train.py \
           ESM2LoRa/POOLING_REPORT.md \
    josartor@izar.epfl.ch:/scratch/izar/josartor/ESM2LoRa/
```

Custo idêntico ao `lora_r8` (~6–7 h): o gargalo é o backbone de 650M, e o pooler
acrescenta 1.281 parâmetros.

Depois, avaliar no benchmark externo:

```bash
sbatch submit_izar_external_all.slurm --add-lora lora_attnpool --add-lora lora_meanmax
```

**Só o `ESM2LoRa` suporta isto.** O caminho frozen guarda o vetor já colapsado de 1280
dims; para usar pooling alternativo teria que armazenar `(L, 1280)` por sequência —
~400× mais disco, centenas de GB para 413k sequências. Na prática, pooling aprendido
exige treino end-to-end.

---

## 4. O que esperar, e como interpretar

**Hipótese:** se a diluição for o gargalo, `attention` deve melhorar principalmente nos
grupos de **mutantes pontuais** (Parkin, PLpro, BRCA1, E4B — hoje com MCC 0), e pouco nos
grupos de sequências desenhadas (chorismate mutase, que já funciona).

**Falsificação:** se o MCC continuar ~0 nos grupos DMS mesmo com atenção, a diluição não
era o gargalo, e a causa está nos rótulos de treino — os negativos são decoys gerados
computacionalmente, não perdas de função reais. Nesse caso a intervenção seguinte são os
decoys `_wt` (mesma casca redesenhada, resíduos catalíticos intactos), que já existem em
`Decoys_datasets/` e foram deliberadamente excluídos na construção da v2.

**Diagnóstico direto, independente do resultado:** após o treino, aplicar
`AttentionPool.weights()` às 200 enzimas do XAI e medir o enriquecimento dos pesos de
atenção nos sítios catalíticos. Se o modelo aprendeu a concentrar onde deve, isso aparece
sem intermediação de gradiente.

### Ressalvas

1. **Uma semente por configuração.** Diferenças de ±0,02 em MCC estão dentro do ruído
   observado entre runs.
2. **A atribuição é de primeira ordem**, não causal. O teste causal seria ablação: mutar o
   sítio catalítico e medir a queda em P(ativa) — factível com os decoys `_mut`, que são
   exatamente isso.
3. **O enriquecimento por camada não é monotônico** (5,30× na camada 0, cai até 0,58 na 31,
   salta para 7,41 na 33). Só a camada 33 é decomposição exata; as demais são primeira
   ordem. Confirmar a 33 com `--method intgrad` seria prudente antes de publicar.

---

## 5. Bibliografia

> Referências dadas de memória: confira autores, ano e venue antes de citar formalmente.

### Agregação por atenção

1. **Bahdanau, Cho & Bengio (2015).** *Neural Machine Translation by Jointly Learning to
   Align and Translate.* ICLR. — origem do mecanismo de atenção aditiva; o `Linear`+softmax
   do `AttentionPool` é a sua forma mais simples.
2. **Lin et al. (2017).** *A Structured Self-attentive Sentence Embedding.* ICLR.
   — **a referência mais próxima do que implementei**: atenção aprendida para agregar
   uma sequência de vetores num único embedding, exatamente substituindo a média.
3. **Ilse, Tomczak & Welling (2018).** *Attention-based Deep Multiple Instance Learning.*
   ICML. — formaliza a agregação por atenção como *pooling* em aprendizado de múltiplas
   instâncias. É o enquadramento teórico do nosso problema: a sequência é uma sacola de
   resíduos e a etiqueta é da sacola inteira.

### Pooling em sequências

4. **Shen et al. (2018).** *Baseline Needs More Love: On Simple Word-Embedding-Based Models
   and Associated Pooling Mechanisms.* ACL. — compara mean, max e concatenações; base
   empírica do modo `mean_max`.
5. **Reimers & Gurevych (2019).** *Sentence-BERT: Sentence Embeddings using Siamese
   BERT-Networks.* EMNLP. — estudo sistemático de estratégias de pooling sobre
   transformers (mean, max, CLS).

### Atribuição (base do diagnóstico)

6. **Shrikumar, Greenside & Kundaje (2017).** *Learning Important Features Through
   Propagating Activation Differences.* ICML. — citação usual para `gradient × input`.
7. **Ancona et al. (2018).** *Towards Better Understanding of Gradient-Based Attribution
   Methods for Deep Neural Networks.* ICLR. — **a referência analítica do método usado**;
   unifica gradient×input, ε-LRP, DeepLIFT e Integrated Gradients.
8. **Sundararajan, Taly & Yan (2017).** *Axiomatic Attribution for Deep Networks.* ICML.
   — Integrated Gradients, disponível como `--method intgrad`.
9. **Adebayo et al. (2018).** *Sanity Checks for Saliency Maps.* NeurIPS. — mostra que
   mapas de saliência podem não refletir o modelo; motiva a avaliação quantitativa contra
   gabarito e o controle por aminoácido.

### Contexto

10. **Lin et al. (2023).** *Evolutionary-scale prediction of atomic-level protein structure
    with a language model.* Science. — ESM-2, o backbone.
11. **Vig et al. (2021).** *BERTology Meets Biology: Interpreting Attention in Protein
    Language Models.* ICLR. — cabeças de atenção do ESM capturam contato estrutural e
    sítios de ligação; precedente direto do nosso achado de 7,41×.
12. **Hu et al. (2022).** *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR.
13. **Geirhos et al. (2020).** *Shortcut Learning in Deep Neural Networks.* Nature Machine
    Intelligence. — a hipótese que o XAI **refutou**: o modelo não está apenas detectando
    redesenho, ele localiza os sítios catalíticos de fato.
14. **Meier et al. (2021).** *Language models enable zero-shot prediction of the effects of
    mutations on protein function.* NeurIPS. — o baseline zero-shot ainda pendente, e
    necessário para saber quanto do desempenho vem do pré-treino e não do nosso treino.
