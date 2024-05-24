# Amicus: O amigo da corte.

![Alt text](amicus.jpeg)

## Um assistente de IA para auxiliar profissionais do direito a analisar audiências judiciais.

### Modelo Base: microsoft/Phi-3-medium-128k-instruct

### Tamanho do Contexto: 128.000 tokens

### Dataset composto por:

- Transcrições de Oitivas: Geradas automaticamente a partir de gravações de áudio utilizando o modelo OpenAI Whisper.
- Resumos de Provas Orais: Extraídos de sentenças judiciais correspondentes, contendo os pontos mais relevantes da prova oral.

### Funcionalidades:

- Análise Detalhada de Oitivas: Amicus processa transcrições de oitivas, identificando trechos relevantes, contradições, inconsistências e pontos-chave.
- Comparação com Provas Orais: O modelo compara as informações da oitiva com os resumos das provas orais inseridos no contexto, auxiliando na identificação de discrepâncias e na avaliação da qualidade da prova oral.
- Auxílio na Tomada de Decisões: Amicus fornece informações valiosas para auxiliar juízes, advogados e outros profissionais do direito na tomada de decisões informadas.

### Benefícios:

- Agilidade: Amicus automatiza a análise de oitivas, economizando tempo e recursos.
- Precisão: O modelo identifica informações relevantes e inconsistências com alta precisão.

### Público-Alvo:

- Juízes
- Advogados
- Promotores
- Defensores Públicos
- Estudantes de Direito
- Pesquisadores
- Outros profissionais do direito

### Limitações:

- Falta de interpretação Subjetiva: A análise de provas e a identificação de inconsistências podem envolver interpretação subjetiva, e o modelo pode não capturar todas as nuances.

### Considerações Éticas:

- Transparência: É fundamental garantir a transparência do uso do modelo e dos seus resultados, informando as partes envolvidas sobre o uso da IA na análise das provas.
- Imparcialidade: O modelo deve ser utilizado de forma imparcial, evitando vieses e discriminação.
- Responsabilidade: Os profissionais do direito são responsáveis pela interpretação e utilização dos resultados fornecidos pelo modelo.

## Aviso Legal: Amicus é uma ferramenta de auxílio à análise de provas e não substitui a análise humana e o julgamento profissional.

## Como usar o modelo?

### Requisitos mínimos: Computador ou servidor com GPU com pelo menos 14GGB de VRAM. Há plataformas que possibilitam o uso de GPUS gratuitamente, como o Google Colab, Amazon Sagemaker e Kraggle.

Abaixo há um exemplo de código em python para o uso do modelo.

```py {"id":"01HYP5QV4QXWMHPKHBF50XHJ8A"}
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

torch.random.manual_seed(0)
model_id = "isatopus/amicus"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

hearing = "Réu: confirmo. No dia do ocorrido eu estava na rua 12 quando vi a vítima com a bolsa. Estava em uma bicicleta, passei e levei a bolsa"

messages = [
    {"role": "user", "content": "Resuma a audiência a seguir::" hearing"},
]

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

generation_args = {
    "max_new_tokens": 1000,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

output = pipe(messages, **generation_args)
print(output[0]['generated_text'])
```
