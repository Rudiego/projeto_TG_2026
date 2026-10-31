# Relatório do treinamento `train_drone_bird-9`

## 1. Resumo executivo

O treinamento `train_drone_bird-9` foi concluído com sucesso em **100 épocas**, usando o modelo **YOLO26n** para detectar duas classes: `drone` e `bird`. O melhor desempenho global ocorreu na **época 65**, com **mAP@50–95 de 41,48%**, **mAP@50 de 81,03%**, **precisão de 92,98%** e **recall de 83,17%**.

O modelo distingue bem as duas classes quando encontra os objetos: na matriz de confusão entre classes, houve **174 classificações corretas em 179 ocorrências (97,21%)**. Os cinco erros observados foram drones classificados como pássaros; nenhum pássaro foi classificado como drone.

O resultado é promissor para identificação geral, especialmente pela alta precisão. Entretanto, o recall de aproximadamente 83% mostra que ainda existem objetos não detectados, e a diferença entre mAP@50 e mAP@50–95 indica que a localização das caixas perde qualidade quando se exige maior sobreposição com a anotação real.

## 2. Configuração do experimento

| Item | Valor |
|---|---:|
| Tarefa | Detecção de objetos |
| Modelo inicial | `yolo26n.pt` (pré-treinado) |
| Classes | `drone`, `bird` |
| Épocas | 100 |
| Tamanho das imagens | 640 × 640 px |
| Batch size | 4 |
| Otimizador | Automático |
| Dispositivo | GPU 0 |
| Precisão mista (AMP) | Ativada |
| Cache | RAM |
| Seed | 42 |
| Execução determinística | Sim |
| Política de learning rate | Cossenoidal |
| Confiança usada na validação | 0,45 |
| IoU do NMS | 0,70 |
| Paciência para parada antecipada | 0 (desativada) |

As principais técnicas de aumento de dados incluíram mosaico, espelhamento horizontal, variações de matiz/saturação/brilho, translação, escala, rotação de até 5° e `mixup` de 5%. O mosaico foi desativado nas últimas 15 épocas.

## 3. Dados utilizados

O arquivo de configuração aponta para:

- **Treinamento:** 610 imagens e 610 arquivos de rótulo.
- **Avaliação:** 158 imagens e 158 arquivos de rótulo.
- **Classes:** 2 (`drone` e `bird`).

Observação metodológica importante: no arquivo de dados, tanto `val` quanto `test` apontam para `images/test`. Portanto, as métricas registradas durante o treinamento foram calculadas no conjunto chamado `test`. Esse conjunto atuou, na prática, como validação ao longo das 100 épocas e não deve ser considerado um teste final completamente independente para estimar generalização.

## 4. Resultados principais

### Melhor época global — época 65

| Métrica | Resultado |
|---|---:|
| Precisão | **92,98%** |
| Recall | **83,17%** |
| mAP@50 | **81,03%** |
| mAP@50–95 | **41,48%** |
| Perda de caixa — treino | 1,4165 |
| Perda de classe — treino | 1,0667 |
| Perda L1 — treino | 0,0196 |
| Perda de caixa — validação | 1,6011 |
| Perda de classe — validação | 0,8413 |
| Perda L1 — validação | 0,0341 |

O peso recomendado para uso é o `best.pt`, associado ao melhor desempenho combinado observado durante o treinamento.

### Outros pontos de destaque

| Indicador | Melhor valor | Época |
|---|---:|---:|
| Maior precisão | 94,83% | 36 |
| Maior recall | 86,03% | 69 |
| Maior mAP@50 | 82,06% | 67 |
| Maior mAP@50–95 | 41,48% | 65 |

Esses máximos ocorrem em épocas diferentes porque há um equilíbrio entre reduzir falsos positivos, encontrar mais objetos e posicionar melhor as caixas.

### Resultado ao final — época 100

| Métrica | Época 65 | Época 100 | Variação |
|---|---:|---:|---:|
| Precisão | 92,98% | 93,02% | +0,04 p.p. |
| Recall | 83,17% | 83,15% | −0,02 p.p. |
| mAP@50 | 81,03% | 79,83% | −1,20 p.p. |
| mAP@50–95 | 41,48% | 38,91% | −2,57 p.p. |

Após a época 65, o treinamento continuou reduzindo as perdas de treino, mas não melhorou o desempenho global de validação. A queda de 2,57 pontos percentuais no mAP@50–95 até a época 100 sugere **sobreajuste moderado**. Isso reforça a escolha de `best.pt`, e não `last.pt`, para inferência.

## 5. Matriz de confusão

| Classe verdadeira | Previsto como drone | Previsto como bird |
|---|---:|---:|
| drone | 85 | 5 |
| bird | 0 | 89 |

Considerando apenas a confusão entre as duas classes:

- Acerto para `drone`: **94,44%** (85 de 90).
- Acerto para `bird`: **100%** (89 de 89).
- Acerto total: **97,21%** (174 de 179).

Essa matriz foi gerada no modo “somente classes” e não mostra a classe de fundo. Por isso, ela avalia muito bem a troca entre `drone` e `bird`, mas não representa sozinha objetos perdidos ou detecções sem correspondência. Para isso, recall, precisão e mAP são indicadores mais completos.

## 6. Interpretação

### Pontos fortes

- **Alta precisão (92,98%)**: a maioria das detecções emitidas pelo modelo é confiável.
- **Boa separação entre classes**: foram observadas apenas cinco trocas entre `drone` e `bird`.
- **mAP@50 de 81,03%**: desempenho sólido quando o critério de sobreposição é moderado.
- **Treinamento estável e reproduzível**: seed fixa e modo determinístico ativado.

### Limitações

- **Recall de 83,17%**: aproximadamente 17% dos objetos deixam de ser recuperados sob as condições de avaliação usadas.
- **mAP@50–95 de 41,48%**: as caixas ainda precisam de maior precisão espacial em critérios de IoU mais rigorosos.
- **Sinal de sobreajuste após a época 65**: continuar até a época 100 não trouxe ganho de validação.
- **Ausência de teste independente**: `val` e `test` usam as mesmas imagens no arquivo de configuração.
- **Validação com confiança 0,45**: esse limiar é relativamente restritivo e pode reduzir o recall; a escolha ideal depende do custo de falsos positivos e falsos negativos na aplicação.

## 7. Recomendações

1. **Usar `best.pt` em produção e nos próximos testes**, pois ele representa o melhor compromisso entre as métricas.
2. **Separar três conjuntos independentes** (`train`, `val` e `test`) antes da avaliação definitiva do TCC/TG.
3. **Testar limiares de confiança**, por exemplo 0,25, 0,35 e 0,45, e escolher o ponto conforme a prioridade: maior recall ou menos falsos alarmes.
4. **Revisar os drones confundidos com pássaros**, buscando padrões de tamanho, distância, iluminação, silhueta e fundo.
5. **Aumentar exemplos difíceis e objetos pequenos**, além de revisar a precisão das caixas de anotação, para elevar o mAP@50–95.
6. **Aplicar parada antecipada ou reduzir o treinamento para cerca de 65–75 épocas** em experimentos semelhantes, economizando tempo e reduzindo sobreajuste.
7. **Executar uma avaliação final única no conjunto de teste independente**, registrando métricas gerais e por classe.

## 8. Tempo e artefatos

- Duração acumulada registrada: **5.091,76 segundos**, aproximadamente **1 h 24 min 52 s**.
- Melhor resultado alcançado por volta de **56 min 28 s**, na época 65.
- Data de conclusão dos artefatos: **5 de outubro de 2026**.
- Peso recomendado: `runs/detect/train_drone_bird-9/weights/best.pt`.
- Peso da última época: `runs/detect/train_drone_bird-9/weights/last.pt`.
- Curvas completas: `runs/detect/train_drone_bird-9/results.png`.
- Matriz de confusão: `runs/detect/train_drone_bird-9/confusion_matrix.png`.

## 9. Conclusão

O experimento `train_drone_bird-9` produziu um detector com **boa confiabilidade e forte capacidade de diferenciar drones de pássaros**, alcançando o melhor resultado na época 65. O principal espaço para evolução está na recuperação de objetos não detectados e no refinamento das caixas delimitadoras. Antes de apresentar o resultado como desempenho final do sistema, é essencial medir o modelo em um conjunto de teste que não tenha sido consultado durante o treinamento.
