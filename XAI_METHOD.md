# Atribuição por resíduo — método, execução e bibliografia

Responde à pergunta: **o classificador olha para os resíduos catalíticos, ou para outra coisa?**

Esquema do método: [`xai_scheme.png`](xai_scheme.png)
Código: [`explain_residues.py`](explain_residues.py) · [`plot_xai.py`](plot_xai.py) · [`submit_izar_xai.slurm`](submit_izar_xai.slurm)

---

## 1. O método em uma linha

Para cada camada *k* e cada resíduo *l*:

```
contribuição(k, l) = Σ_d  ∂logit/∂h_k[l,d] · h_k[l,d]
```

É **gradiente × ativação** (*gradient × input*, também chamado de *input × gradient*),
aplicado aos estados ocultos em vez da entrada.

## 2. Por que não é Grad-CAM literal

O Grad-CAM pondera mapas de ativação pelo gradiente **médio espacial**. Aqui o
pooling é média simples sobre resíduos:

```
pooled = (1/N) · Σ_l h₃₃[l]     ⟹     ∂logit/∂h₃₃[l] = g/N  para todo l
```

Verificado numericamente: o gradiente é idêntico (0,057223) em todos os resíduos
reais, e zero apenas em `<cls>`/`<eos>`, que a máscara exclui. **Um mapa Grad-CAM
literal sairia perfeitamente plano.** Toda a informação posicional está em `h[l]`,
e só o produto a recupera.

## 3. Exatidão

| camada | propriedade |
|---|---|
| **33 (última)** | `Σ_l contribuição(33,l) = g · pooled` — **decomposição aditiva exata** do vetor que entra na cabeça (verificado: 0,083592 vs 0,083591) |
| **k < 33** | a atenção mistura posições, o gradiente passa a variar com *l*, e a aditividade se perde → **atribuição de primeira ordem (Taylor)** |

A distinção está documentada no docstring do script e na legenda da figura. As
camadas intermediárias continuam respondendo "quais resíduos naquela profundidade
movem o logit", mas são aproximação e devem ser lidas como tal.

## 4. Custo

Uma forward (`output_hidden_states=True`) + **uma única** backward. O
`retain_grad()` nos 34 estados faz com que um só `backward()` preencha todos os
`.grad` de uma vez — não são 34 passagens. Custo ≈ 2× uma inferência.

## 5. Avaliação quantitativa

Sem gabarito, um mapa de atribuição é apenas uma figura bonita. Aqui há gabarito:
`catalytic_sites.csv` traz `ACT_SITE_list` (posições 1-based) para **79.294**
entradas UniProt, validado contra as sequências.

Por camada, os resíduos são ranqueados **dentro da própria sequência** (percentil,
para que sequências de comprimentos diferentes sejam comparáveis):

| métrica | acaso | leitura |
|---|---|---|
| `catalytic_in_top10pct` | 0,10 | fração dos catalíticos entre os 10% mais atribuídos |
| `enrichment_top10pct` | **1,0** | razão sobre o acaso — **é o número que decide** |
| `mean_pct_rank_catalytic` | 0,50 | rank médio dos catalíticos |

**Interpretação:**
- `enrichment ≈ 1,0` em todas as camadas → o modelo não localiza os sítios catalíticos
  em profundidade nenhuma. Confirma a hipótese de atalho ("detecta redesenho").
- `enrichment 3–5×` em camadas profundas → ele localiza, e o gargalo é outro
  (diluição do pooling, calibração).

## 6. Como rodar

```bash
cd ~/ESM2LoRa                       # ou /scratch/izar/$USER/ESM2LoRa
export CATALYTIC_SITES=~/Decoys/Positive_decoys/catalytic_sites.csv
mkdir -p logs

# preencher --qos / --account primeiro
sbatch submit_izar_xai.slurm lora_r8
sbatch submit_izar_xai.slurm lora_deep6x2560
```

Saída em `xai_<run>/`: `attributions.csv`, `summary.json`, `xai_summary.png`.

Arquivos a enviar:
```bash
rsync -avz ESM2LoRa/explain_residues.py ESM2LoRa/plot_xai.py \
           ESM2LoRa/submit_izar_xai.slurm ESM2LoRa/XAI_METHOD.md \
    josartor@izar.epfl.ch:/scratch/izar/josartor/ESM2LoRa/
```

---

## 7. Bibliografia

> Referências dadas de memória: confira os DOIs antes de citar formalmente.

### Fundamentos dos métodos de atribuição

1. **Simonyan, Vedaldi & Zisserman (2014).** *Deep Inside Convolutional Networks:
   Visualising Image Classification Models and Saliency Maps.* ICLR Workshop.
   — origem do saliency map por gradiente.
2. **Selvaraju et al. (2017).** *Grad-CAM: Visual Explanations from Deep Networks
   via Gradient-based Localization.* ICCV. — o método que dá nome à família; ler
   para entender por que a média espacial do gradiente **não** se aplica ao nosso caso.
3. **Sundararajan, Taly & Yan (2017).** *Axiomatic Attribution for Deep Networks.*
   ICML. — Integrated Gradients, implementado como `--method intgrad`. Traz os
   axiomas (sensibilidade, invariância de implementação) que gradiente×ativação viola parcialmente.
4. **Shrikumar, Greenside & Kundaje (2017).** *Learning Important Features Through
   Propagating Activation Differences.* ICML. — DeepLIFT; discute a relação com gradiente×entrada.
5. **Ancona et al. (2018).** *Towards Better Understanding of Gradient-Based
   Attribution Methods for Deep Neural Networks.* ICLR. — **a referência central
   para o nosso método**: unifica gradiente×entrada, DeepLIFT e LRP, e mostra
   quando são equivalentes.

### Ressalvas — leitura obrigatória

6. **Adebayo et al. (2018).** *Sanity Checks for Saliency Maps.* NeurIPS.
   — mostra que vários mapas de saliência não mudam quando os pesos são
   randomizados, ou seja, não explicam o modelo. Motiva a avaliação quantitativa
   contra gabarito em vez de inspeção visual.
7. **Geirhos et al. (2020).** *Shortcut Learning in Deep Neural Networks.*
   Nature Machine Intelligence. — descreve exatamente o fenômeno que suspeitamos:
   o modelo acerta o benchmark por uma correlação espúria e falha fora da distribuição.

### Interpretabilidade em transformers

8. **Chefer, Gur & Wolf (2021).** *Transformer Interpretability Beyond Attention
   Visualization.* CVPR. — propagação de relevância específica para transformers.
9. **Abnar & Zuidema (2020).** *Quantifying Attention Flow in Transformers.* ACL.
   — *attention rollout*, alternativa posicional que não implementamos.
10. **Jain & Wallace (2019).** *Attention is not Explanation.* NAACL. e
    **Wiegreffe & Pinter (2019).** *Attention is not not Explanation.* EMNLP.
    — o debate sobre usar pesos de atenção como explicação; a razão de termos
    escolhido gradiente×ativação em vez de atenção.

### Modelos de linguagem de proteínas

11. **Rives et al. (2021).** *Biological structure and function emerge from scaling
    unsupervised learning to 250 million protein sequences.* PNAS. — ESM-1b.
12. **Lin et al. (2023).** *Evolutionary-scale prediction of atomic-level protein
    structure with a language model.* Science. — **ESM-2**, o backbone usado aqui.
13. **Meier et al. (2021).** *Language models enable zero-shot prediction of the
    effects of mutations on protein function.* NeurIPS. — o método de
    *masked marginal* (razão de log-verossimilhança), que é o **baseline zero-shot
    que ainda não medimos**.
14. **Vig et al. (2021).** *BERTology Meets Biology: Interpreting Attention in
    Protein Language Models.* ICLR. — mostra que cabeças de atenção do ESM capturam
    contato estrutural e sítios de ligação; o precedente mais próximo do que queremos testar.

### Sondas por camada (a análise complementar)

15. **Alain & Bengio (2017).** *Understanding intermediate layers using linear
    classifier probes.* ICLR Workshop. — origem do método de sonda linear.
16. **Tenney, Das & Pavlick (2019).** *BERT Rediscovers the Classical NLP Pipeline.*
    ACL. — sondas por camada revelando hierarquia de representação.
17. **Hewitt & Liang (2019).** *Designing and Interpreting Probes with Control Tasks.*
    EMNLP. — ressalva: uma sonda expressiva pode aprender a tarefa sozinha; exige
    tarefa-controle para ser interpretável.

### Information bottleneck (se for seguir esse caminho)

18. **Tishby & Zaslavsky (2015).** *Deep Learning and the Information Bottleneck
    Principle.* IEEE ITW.
19. **Shwartz-Ziv & Tishby (2017).** *Opening the Black Box of Deep Neural Networks
    via Information.* arXiv.
20. **Saxe et al. (2018).** *On the Information Bottleneck Theory of Deep Learning.*
    ICLR. — **a crítica**: a fase de compressão depende da não-linearidade e do
    binning, podendo ser artefato do estimador. Ler antes de investir nessa análise.

### Contexto do projeto

21. **Hu et al. (2022).** *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR.
22. **Notin et al. (2023).** *ProteinGym: Large-Scale Benchmarks for Protein Fitness
    Prediction and Design.* NeurIPS. — fonte de parte do conjunto de validação externa.
