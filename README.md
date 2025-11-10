# 🌪️ Analisador de Risco de Tornado com Deep Learning YOLO

## 📋 Visão Geral

Aplicativo Python completo para análise de imagens de satélite detectando padrões de tornado usando:
- **Métodos clássicos**: Detecção de círculos, espirais, funis via processamento de imagem
- **Deep Learning**: Rede neural YOLO para detecção avançada de objetos
- **Scoring inteligente**: Combina múltiplas detecções para score de risco (0-100)
- **Relatórios em PDF**: Visualizações completas com comparação de métodos

## 🎯 Características Principais

### ✨ Análise Inteligente
- ✅ Detecção de padrões circulares (tornadoes vistos de cima)
- ✅ Análise de padrões espirais (rotação)
- ✅ Identificação de formas de funil (estrutura de tornado)
- ✅ Detecção de vórtices por Hough Circles com validação física
- ✅ Análise de cores (verde/vermelho para tempestades)
- ✅ Análise de textura e turbulência
- ✅ Detecção por Deep Learning YOLO

### 🤖 Deep Learning Integration
- ✅ Modelo YOLO12n pré-treinado
- ✅ Detecção de objetos com confiança
- ✅ Filtro para formas circulares
- ✅ Confirmação mútua com métodos clássicos
- ✅ Bônus de score quando confirmado
- ✅ Fallback gracioso se não instalado

### 📊 Relatórios Profissionais
- ✅ PDF multi-página
- ✅ Página 1: Visão geral + métricas
- ✅ Página 2: Padrões detectados (circular, espiral, funil)
- ✅ Página 3: Visualizações YOLO com bounding boxes
- ✅ Página 4: Histogramas e distribuições

### 🏗️ Arquitetura
- ✅ Padrão MVC (Model-View-Controller)
- ✅ Interface Tkinter moderna com sv-ttk
- ✅ Threading para não travar UI

## 🚀 Quick Start

### 1. Instalar Dependências

```powershell
pip install -r ./requirements.txt
```

### 2. Executar Aplicação

```powershell
python main.py
```

### 3. Usar Interface

1. Clique em "Selecionar Imagem"
2. Escolha uma imagem de satélite com nuvens
3. Clique em "Analisar Imagem"
4. Aguarde análise (2-5 segundos)
5. Clique em "Gerar Relatório PDF"
6. Abra o PDF gerado

## 📁 Estrutura do Projeto

```
PDI/
├── main.py                              # Entry point
├── model.py                             # Lógica de análise (TornadoRiskAnalyzer)
├── view.py                              # Interface Tkinter (TornadoAnalyzerGUI)
├── controller.py                        # Coordenador (TornadoController)
├── Example image/                       # Imagens de teste
│   └── Example low.png
├── DOCUMENTATION/
│   ├── README.md                        # Este arquivo
│   ├── YOLO_SETUP.md                   # Guia de instalação YOLO
│   ├── YOLO_IMPLEMENTATION_SUMMARY.md  # Mudanças implementadas
│   ├── TESTING_GUIDE.md                # Guia de testes
│   └── IMPLEMENTATION_CHECKLIST.md     # Checklist de funcionalidades
└── relatorio_tornado_*.pdf             # Relatórios gerados
```

## 🎯 Métodos de Detecção

### 1. Padrões Circulares (Métodos Clássicos)

```
Entrada: Imagem de satélite
  ↓
Canny Edge Detection (3 passadas com diferentes thresholds)
  ↓
Find Contours
  ↓
Calcular Circularity = 4π·Area / Perimeter²
  ↓
Filtrar: Circularity > 0.65 (tornado) ou 0.75 (dia)
  ↓
Validar: Radius uniformity (CV < 0.2)
```

**Resultado**: Contornos circulares validados

### 2. Vórtices por Hough Circles (Métodos Clássicos)

```
Entrada: Imagem Gray
  ↓
Hough Circles (2 passadas)
  - Grande: r=30-250
  - Pequeno: r=15-120
  ↓
Validação Física:
  - Centro mais escuro que borda
  - Estrutura radial
  - Razão intensidade < 0.95
  ↓
Angular Rotation Analysis
```

**Resultado**: Vórtices confirmados com assinatura física

### 3. Padrões Espirais (Métodos Clássicos)

```
Entrada: Contornos
  ↓
Análise de Distância Radial
  ↓
Correlação entre ângulo e distância
  ↓
Filtrar: Correlação > 0.55 = espiral
```

**Resultado**: Padrões de rotação detectados

### 4. Formas de Funil (Métodos Clássicos)

```
Entrada: Contornos
  ↓
Validar Solidity: 0.3-0.65
Validar Aspect Ratio: > 1.5
  ↓
Verif icação de Afilamento:
  - Dividir em 3 segmentos
  - Cada segmento menor que anterior
  ↓
Validar Tip (ponto final)
```

**Resultado**: Formas de funil confirmadas

### 5. Deep Learning YOLO (NOVO!)

```
Entrada: Imagem RGB
  ↓
Modelo YOLOv8n
  ↓
Detecção de Objetos
  ↓
Filtrar:
  - Confiança > 0.5
  - Aspect ratio 0.7-1.4 (circular)
  ↓
Contar e armazenar confiança
```

**Resultado**: Objetos circulares detectados por rede neural

## 📊 Sistema de Scoring

### Pontuação Base (0-100)

```
1. Padrões de Rotação (0-50 pts)
   - Circulares: até 25 pts
   - Vórtices: até 25 pts
   - Espirais: até 20 pts

2. Áreas Escuras (0-25 pts)
   - Nuvens muito densas

3. Contraste Extremo (0-15 pts)
   - Turbulência/instabilidade

4. Cores de Tempestade (0-10 pts)
   - Verde/Vermelho indicativo

5. Detecção YOLO (0-15 pts) ← NOVO
   - 5 pts por objeto circular
   - Máximo 15 pts
```

### Bônus (até +50 pts)

```
- Vórtices + Escuridão: +15
- Múltiplos padrões de rotação: +10
- Tripla ameaça (contraste + escuridão + rotação): +20
- YOLO confirmando padrões: +10
```

### Normalização

```
Score final = min(pontos_totais, 100)

Classificação:
- 0-15: ⚪ Mínimo
- 15-30: 🟢 Baixo
- 30-50: 🟡 Moderado
- 50-70: 🟠 Alto
- 70-85: 🔴 Muito Alto
- 85-100: 🔴 CRÍTICO
```

## 🔧 Configuração

### Ajustar Parâmetros

Editar em `model.py`:

```python
# Linha ~500: Modelo YOLO
model = YOLO('yolo12n.pt')

# Linha ~516: Filtro de circularidade
if 0.7 < aspect_ratio < 1.4:  # ajustar intervalo

# Linha ~610: Pontos YOLO por objeto
yolo_score = min(yolo_circular * 5, 15)  # ajustar multiplicador
```

## 📈 Exemplo de Saída

### PDF Gerado

```
[Página 1] Visão Geral + Métricas
[Página 2] Padrões Detectados (4 subplots)
[Página 3] YOLO Deep Learning (4 subplots) ← NOVO
[Página 4] Histogramas (4 subplots)
```

## 🐛 Troubleshooting

### YOLO não instalado?
```
⚠ Aviso no console: "YOLO não instalado..."
✓ Aplicativo continua funcionando normalmente
✓ Usa métodos clássicos apenas
```

### YOLO não detecta nada?
```
Possíveis causas:
- Imagem muito clara (sem tempestade)
- Padrões muito sutis
- Aspecte ratio não circular

Solução:
- Use imagens com formações mais definidas
- Ajuste limiares em model.py
```

### Aplicativo lento?
```
Verificar:
- Primeira execução YOLO? (baixando modelo ~100MB)
- CPU com alta utilização?
- Imagem muito grande?

Soluções:
- Reduzir resolução de imagem
- Usar GPU NVIDIA se disponível
- Usar YOLOv8n (padrão é rápido)
```

## 🎓 Requisitos

- Python 3.10+ < 3.14
- Windows 7+, Mac OS 10.13+, Linux
- RAM: 2 GB mínimo (4 GB recomendado com YOLO)
- Espaço em disco: 500 MB (com YOLO: 1 GB)